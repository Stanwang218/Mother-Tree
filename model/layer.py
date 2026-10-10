from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import (Dropout, LayerNorm, Linear, Module, Tensor,
                                          _get_activation_fn)
from typing import Optional
from torch.utils.checkpoint import checkpoint
from torch.nn import MultiheadAttention

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, TransformerEncoder


# ── SSMax / QASSMax (adapted from tabicl) ──
def _logn(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(math.log(max(n, 1)), device=device, dtype=dtype)

class SSMaxLayer(nn.Module):
    """Applies SSMax scaling to projected Q: q = q * MLP(log n)."""
    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64, elementwise: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.elementwise = elementwise
        out_dim = num_heads * head_dim if elementwise else num_heads
        self.mlp = nn.Sequential(nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, out_dim))

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        """q shape: (..., n_heads, seq_len, head_dim)"""
        logn = _logn(n, q.device, q.dtype).reshape(1, 1)
        scales = self.mlp(logn)
        if self.elementwise:
            scales = scales.view(1, self.num_heads, 1, -1)
        else:
            scales = scales.view(1, self.num_heads, 1, 1)
        return q * scales

class QASSMaxLayer(nn.Module):
    """Query-aware SSMax: q = q * base_mlp(log n) * (1 + tanh(query_mlp(q)))."""
    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64, elementwise: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.elementwise = elementwise
        base_out = num_heads * head_dim if elementwise else num_heads
        query_out = head_dim if elementwise else 1
        self.base_mlp = nn.Sequential(nn.Linear(1, n_hidden), nn.GELU(), nn.Linear(n_hidden, base_out))
        self.query_mlp = nn.Sequential(nn.Linear(head_dim, n_hidden), nn.GELU(), nn.Linear(n_hidden, query_out))
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        logn = _logn(n, q.device, q.dtype).reshape(1, 1)
        if self.elementwise:
            base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, self.head_dim)
            modulation = 1 + torch.tanh(self.query_mlp(q))
        else:
            base_scales = self.base_mlp(logn).view(1, self.num_heads, 1, 1)
            modulation = 1 + torch.tanh(self.query_mlp(q))
        return q * (base_scales * modulation)


def create_ssmax_layer(ssmax: str, num_heads: int, embed_dim: int):
    head_dim = embed_dim // num_heads
    if not ssmax or ssmax == "none":
        return None
    elif ssmax == "ssmax-mlp":
        return SSMaxLayer(num_heads, head_dim)
    elif ssmax == "ssmax-mlp-elementwise":
        return SSMaxLayer(num_heads, head_dim, elementwise=True)
    elif ssmax == "qassmax-mlp":
        return QASSMaxLayer(num_heads, head_dim)
    elif ssmax == "qassmax-mlp-elementwise":
        return QASSMaxLayer(num_heads, head_dim, elementwise=True)
    else:
        raise ValueError(f"Unknown ssmax: {ssmax}")


class _SSMaxAttentionWrapper(nn.Module): # batch first code
    """Wraps nn.MultiheadAttention to apply SSMax to Q after projection."""
    def __init__(self, mha: MultiheadAttention, ssmax_layer: Optional[nn.Module]):
        super().__init__()
        self.mha = mha
        self.ssmax_layer = ssmax_layer

    def forward(self, query, key, value, **kwargs):
        if self.ssmax_layer is None:
            return self.mha(query, key, value, **kwargs)
        bf = self.mha.batch_first
        # Normalize to batch_first internally so head-reshape logic is uniform
        if not bf:
            query = query.transpose(0, 1)   # (tgt,b,e) -> (b,tgt,e)
            key   = key.transpose(0, 1)     # (src,b,e) -> (b,src,e)
            value = value.transpose(0, 1) # (src,b,e) -> (b,src,e)

        E, H = self.mha.embed_dim, self.mha.num_heads
        hd = E // H
        W, b = self.mha.in_proj_weight, self.mha.in_proj_bias
        q = F.linear(query, W[:E], b[:E] if b is not None else None)
        k = F.linear(key,   W[E:2*E], b[E:2*E] if b is not None else None)
        v = F.linear(value, W[2*E:],  b[2*E:] if b is not None else None)
        bs, tgt = query.shape[:2]
        src = key.shape[1]
        q = q.view(bs, tgt, H, hd).transpose(1, 2)  # (b, tgt, H, hd) -> (b, H, tgt, hd)
        k = k.view(bs, src, H, hd).transpose(1, 2)  # (b, src, H, hd) -> (b, H, src, hd)
        v = v.view(bs, src, H, hd).transpose(1, 2)  # (b, src, H, hd) -> (b, H, src, hd)
        q = self.ssmax_layer(q, src)                    # scale ONCE, on projected q
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=kwargs.get('is_causal', False))
        out = attn.transpose(1,2).reshape(bs, tgt, E)  # (b, H, tgt, hd) -> (b, tgt, E)
        out = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)
        if not bf:
            out = out.transpose(0, 1)   # (b,tgt,e) -> (tgt,b,e)
        return (out, None)  # need_weights=False


class BiAttentionEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False):
        super().__init__()
        self.cross_feature_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.cross_sample_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # src_mask is in with eval position, applies only to samples
        # src comes in as samples x batch x feature x emsize
        # reshape to features x (samples * batch) x emsize for cross-feature attention
        post_feature_attention = self.cross_feature_attention(src.reshape(-1, *src.shape[2:]).transpose(0, 1), src_mask)
        # from cross-feature attention, we get features x (samples * batch) x emsize
        # reshape back to original, then reshape to samples x (batch * feature) x emsize
        reshaped = post_feature_attention.transpose(0, 1).reshape(src.shape)
        reshaped = reshaped.reshape(src.shape[0], -1, src.shape[-1])
        res = self.cross_sample_attention(reshaped, src_mask)
        return res.reshape(src.shape)


class LinearBiAttentionEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=True, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False):
        super().__init__()
        self.cross_feature_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)
        self.cross_sample_attention = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # src_mask is in with eval position, applies only to samples
        # src comes in as samples x batch x feature x emsize
        # reshape to features x (samples * batch) x emsize for cross-feature attention
        post_feature_attention = self.cross_feature_attention(src.reshape(-1, *src.shape[2:]).transpose(0, 1), src_mask)
        # from cross-feature attention, we get features x (samples * batch) x emsize
        # reshape back to original, then reshape to samples x (batch * feature) x emsize
        reshaped = post_feature_attention.transpose(0, 1).reshape(src.shape)
        reshaped = reshaped.reshape(src.shape[0], -1, src.shape[-1])
        res = self.cross_sample_attention(reshaped, src_mask)
        return res.reshape(src.shape)



class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(
        self, 
        d_model, 
        nhead, 
        dim_feedforward=2048, 
        dropout=0.1, 
        activation="relu",
        layer_norm_eps=1e-5, 
        batch_first=True, 
        pre_norm=False,
        device=None, 
        dtype=None, 
        recompute_attn=False,
        attn_name = 'default',
        norm_output = False,
        ssmax = False,
    ) -> None:
        # batch_first is set to True for using flash attention II
        # check the details of when flash attention can be triggered here: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
                d_model, 
                nhead, 
                dropout=dropout, 
                batch_first=batch_first,
                **factory_kwargs,
            )
        self.ssmax_layer = create_ssmax_layer(ssmax if isinstance(ssmax, str) else ("qassmax-mlp-elementwise" if ssmax else "none"),
                                              nhead, d_model)
        if self.ssmax_layer is not None:
            self.self_attn = _SSMaxAttentionWrapper(self.self_attn, self.ssmax_layer)
        self.attn_name = attn_name
        
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn

        self.activation = _get_activation_fn(activation)

    def forward(
        self, 
        src: Tensor, 
        src_mask: Optional[Tensor] = None, 
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src

        if isinstance(src_mask, tuple):
            return NotImplementedError
        elif isinstance(src_mask, int):
            single_eval_position = src_mask
            train_mask = None
            test_mask = None
            train_is_causal = False
            test_is_causal = False

            # split the training and testing samples
            src_train = src_[:single_eval_position]
            src_test = src_[single_eval_position:]
            # since we set batch_first = True, the shape of src_ is (batch, seq, feature)
            src_train = src_train.permute(1, 0, 2)
            src_test = src_test.permute(1, 0, 2)

            # the training samples are only attend to themselves
            src_left = self.self_attn(
                src_train, 
                src_train, 
                src_train, 
                attn_mask=train_mask,
                is_causal=train_is_causal,
                need_weights=False,
            )[0]

            # the testing samples attend to training samples
            src_right = self.self_attn(
                src_test, 
                src_train, 
                src_train,
                attn_mask=test_mask,
                is_causal=test_is_causal,
                need_weights=False,
            )[0]

            # permute them back to (seq, batch, feature)
            src_left = src_left.permute(1, 0, 2)
            src_right = src_right.permute(1, 0, 2)
            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                # this might have some problems, double check
                # https://github.com/pytorch/pytorch/issues/99282

                src2 = checkpoint(
                    self.self_attn, 
                    src_, # query: Tensor,
                    src_, # key: Tensor,
                    src_, # value: Tensor,
                    None, # key_padding_mask: Optional[Tensor] = None,
                    False, # need_weights: bool = True,
                    src_mask, # attn_mask: Optional[Tensor] = None,
                    True, # average_attn_weights: bool = True,
                    False, # is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
                    use_reentrant=True,
                )[0]
            else:
                src2 = self.self_attn(
                    query = src_, 
                    key = src_, 
                    value = src_, 
                    attn_mask = src_mask,
                    is_causal = False,
                    need_weights=False,
                )[0]
        
        # residual connection
        src = src + self.dropout1(src2)
        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        return src


class TransformerEncoderSimple(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, 
        src: Tensor, 
        mask: Optional[Tensor] = None, 
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(
                output, 
                src_mask=mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
