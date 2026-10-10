import torch
from torch import nn
import numpy as np

class LinearModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_out=10, hidden_size=1024):
        super().__init__()
        self.emsize = emsize
        self.n_out = n_out
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(nn.Linear(emsize,  hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, (emsize + 1) * n_out))

    def forward(self, x):
        return self.mlp(x.mean(0)).reshape(-1, self.emsize + 1, self.n_out)


class SummaryLayer(nn.Module):
    def __init__(self, emsize=512, embed_dim=2048, n_out=10, decoder_type='output_attention', nhead=4):
        super().__init__()
        self.emsize = emsize
        self.n_out = n_out
        self.decoder_type = decoder_type
        self.nhead = nhead

        if decoder_type == "output_attention":
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
            out_size = embed_dim
            self.output_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.nhead, kdim=emsize, vdim=emsize)
        elif decoder_type == "special_token":
            out_size = emsize
            self.output_layer = nn.MultiheadAttention(embed_dim=emsize, num_heads=self.nhead)
        elif decoder_type == "special_token_simple":
            out_size = emsize
        elif decoder_type == "class_tokens":
            out_size = emsize * n_out
        elif decoder_type == "class_average":
            out_size = emsize * n_out
        elif decoder_type == "average":
            out_size = emsize
        elif decoder_type == "feature_average":
            out_size = emsize
        else:
            raise ValueError(f"Unknown decoder_type {decoder_type}")

        self.out_size = out_size

    def forward(self, x, y_src):        # seq, batch, emsize
        if x.shape[0] != 0:
            if self.decoder_type == "output_attention":
                if x.ndim == 3:
                    res = self.output_layer(self.query.repeat(1, x.shape[1], 1), x, x, need_weights=False)[0].squeeze(0)
                elif x.ndim == 4:
                    x_flat = x.reshape(x.shape[0], -1, x.shape[3])
                    res = self.output_layer(self.query.repeat(1, x_flat.shape[1], 1), x_flat, x_flat, need_weights=False)[0]
                    res = res.reshape(x.shape[1], x.shape[2], -1)
                else:
                    raise ValueError(f"Unknown x shape: {x.shape}")
            elif self.decoder_type == "special_token":
                res = self.output_layer(x[[0]], x[1:], x[1:], need_weights=False)[0]
            elif self.decoder_type == "special_token_simple":
                res = x[0]
            elif self.decoder_type == "class_tokens":
                res = x[:10].transpose(0, 1)
            elif self.decoder_type == "class_average":
                # per-class mean
                # clamping y_src to avoid -100 which should be ignored in loss
                # fingers crossed this will not mess things up
                y_src = y_src.long().clamp(0)
                # scatter add does not broadcast so we need to expand
                if y_src.ndim == 1:
                    # batch size 1
                    y_src = y_src.unsqueeze(1)
                if x.ndim == 3:
                    indices = y_src.unsqueeze(-1).expand(-1, -1, self.emsize)
                    sums = torch.zeros(self.n_out, x.shape[1], self.emsize, device=x.device, dtype = x.dtype)
                    sums.scatter_add_(0, indices, x)
                    # create counts
                    ones = torch.ones(1, device=x.device).expand(x.shape[0], x.shape[1])
                    counts = torch.zeros(self.n_out, x.shape[1], device=x.device)
                    indices = y_src
                elif x.ndim == 4:
                    indices = y_src.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], self.emsize)
                    sums = torch.zeros(self.n_out, x.shape[1], x.shape[2], self.emsize, device=x.device)
                    sums.scatter_add_(0, indices, x)
                    # create counts
                    ones = torch.ones(1, device=x.device).expand(x.shape[0], x.shape[1], x.shape[2])
                    counts = torch.zeros(self.n_out, x.shape[1], x.shape[2], device=x.device)
                    indices = y_src.unsqueeze(-1).expand(-1, -1, x.shape[2])
                else:
                    raise ValueError(f"Unknown x shape: {x.shape}")
                counts.scatter_add_(0, indices, ones)
                counts = counts.clamp(1e-10)  # don't divide by zero
                res = (sums / counts.unsqueeze(-1)).transpose(0, 1)
            elif self.decoder_type == "average":
                res = x.mean(0)
            elif self.decoder_type == "feature_average":
                assert x.ndim == 4
                res = x.mean(0)
                pass
            else:
                raise ValueError(f"Unknown decoder_type: {self.decoder_type}")
        else:
            raise ValueError("Empty input")
        return res


def make_decoder_mlp(in_size, hidden_size, out_size, n_layers=1, activation='relu'):
    if activation == 'relu':
        activation = nn.ReLU
    elif activation == 'gelu':
        activation = nn.GELU
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return nn.Sequential(
        nn.Linear(in_size,  hidden_size), # 5120 * 2048 + 2048
        activation(),
        *sum([[nn.Linear(hidden_size, hidden_size), activation()] for _ in range(n_layers - 1)], []),
        nn.Linear(hidden_size, out_size)) # 2048 * 25738 


class TreeTransformerDecoder(nn.Module):
    def __init__(self, depth, in_size, hidden_size, inner_size, leaf_size, out_size, 
                 n_layers=1, n_estimators = 1,
                 activation='relu', n_heads=4, shared_unembedding=False):
        super().__init__()
        # print('shared_unembedding:', shared_unembedding)
        self.depth = depth
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        assert out_size == inner_size + leaf_size, "out_size must be equal to inner_size + leaf_size"
        self.n_layers = n_layers
        self.activation = activation
        self.nhead = n_heads
        self.number_of_nodes = 2 ** (depth + 1) - 1
        self.number_of_leaves = 2 ** depth
        self.n_estimators = n_estimators
        self.number_of_internal_nodes = self.number_of_nodes - self.number_of_leaves
        assert inner_size % self.number_of_internal_nodes == 0, "inner_size must be divisible by number of internal nodes"
        assert leaf_size % self.number_of_leaves == 0, "leaf_size must be divisible by number of leaves"
        self.inner_size = inner_size // self.number_of_internal_nodes
        self.leaf_size = leaf_size // self.number_of_leaves
        self.embedding = nn.Linear(in_size, hidden_size * self.number_of_nodes) 
        self.shared_unembedding = shared_unembedding
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=self.nhead, 
            dim_feedforward=hidden_size * 4, activation=activation,
            batch_first=True)
              for _ in range(n_layers)])
        if self.shared_unembedding:
            self.unembedding_inner = nn.Linear(hidden_size, self.inner_size)
            self.unembedding_leaf = nn.Linear(hidden_size, self.leaf_size)
        else:
            self.unembedding = nn.Linear(hidden_size * self.number_of_nodes, out_size)
        self.register_buffer(
            "tree_mask",
            self.generate_tree_attention_mask(depth, device="cpu")
            )
    def is_ancestor(self, i, j):
        # check if node i is an ancestor of node j in a complete binary tree
        if i == 0:
            return True
        while j > 0:
            j = (j - 1) // 2
            if j == i:
                return True
        return False
    
    def generate_tree_attention_mask(self, depth, device):
        num_nodes = 2 ** (depth + 1) - 1
        mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if j == i or self.is_ancestor(j, i):  
                    mask[i, j] = False
        return mask.to(device)
    

    def forward(self, x):
        batch_size = x.shape[0]
        mask = self.tree_mask.to(x.device)
        embed = self.embedding(x).reshape(-1, self.number_of_nodes, self.hidden_size)
        for layer in self.transformer_layers:
            embed = layer(embed, src_mask=mask)
        if self.shared_unembedding:
            inner_embed = self.unembedding_inner(embed[:,:self.number_of_internal_nodes])\
                    .reshape(batch_size, self.number_of_internal_nodes, self.n_estimators, -1).transpose(1, 2)
            leaf_embed = self.unembedding_leaf(embed[:,self.number_of_internal_nodes:])\
                .reshape(batch_size, self.number_of_leaves, self.n_estimators, -1).transpose(1, 2)
            inner_embed = inner_embed.reshape(batch_size, -1)
            leaf_embed = leaf_embed.reshape(batch_size, -1)
            return torch.cat([inner_embed, leaf_embed], dim=1)
        else:
            return self.unembedding(embed.reshape(-1, self.number_of_nodes * self.hidden_size))  # should be (batch_size, out_size)

class MLPModelDecoder(nn.Module):
    def __init__(self, emsize=512, n_out=10, hidden_size=1024, decoder_type='output_attention', predicted_hidden_layer_size=None, embed_dim=2048,
                 decoder_hidden_layers=1, nhead=4, predicted_hidden_layers=1, weight_embedding_rank=None, low_rank_weights=False, decoder_activation='relu',
                 in_size=100):
        super().__init__()
        self.emsize = emsize
        self.embed_dim = embed_dim
        self.n_out = n_out
        self.hidden_size = hidden_size
        self.decoder_type = decoder_type
        self.predicted_hidden_layer_size = predicted_hidden_layer_size or emsize # 512 hidden size for predictor
        self.in_size = in_size # number of features for predictor
        self.nhead = nhead
        self.weight_embedding_rank = weight_embedding_rank if low_rank_weights else None

        self.predicted_hidden_layers = predicted_hidden_layers
        self.summary_layer = SummaryLayer(emsize=emsize, n_out=n_out, decoder_type=decoder_type, embed_dim=embed_dim, nhead=nhead)
        mlp_in_size = self.summary_layer.out_size
        if self.predicted_hidden_layers == 0:
            if decoder_type in ["class_tokens", "class_average"]:
                # class convolutional
                self.num_output_layer_weights = 1 + self.in_size
                mlp_in_size = emsize
            else:
                self.num_output_layer_weights =  n_out * (1 + self.in_size)
        elif self.weight_embedding_rank is None:
            self.num_output_layer_weights = (self.predicted_hidden_layer_size + 1) * n_out + (self.in_size + 1) * self.predicted_hidden_layer_size # output lyer + input layer
            if self.predicted_hidden_layers > 1:
                self.num_output_layer_weights += (self.predicted_hidden_layers - 1) * (self.predicted_hidden_layer_size ** 2 + self.predicted_hidden_layer_size)
        else:
            self.num_output_layer_weights = (self.predicted_hidden_layer_size + 1) * n_out + self.in_size * \
                self.weight_embedding_rank + self.predicted_hidden_layer_size # 513 * 10 + 100 * 32 + 512 = 8842
            if self.predicted_hidden_layers > 1:
                self.num_output_layer_weights += (self.predicted_hidden_layers - 1) * (self.predicted_hidden_layer_size *
                                                                                       self.weight_embedding_rank + self.predicted_hidden_layer_size) # 16896 = 512 * 33
            self.shared_weights = nn.ParameterList([nn.Parameter(torch.randn(
                self.weight_embedding_rank, self.predicted_hidden_layer_size) / self.weight_embedding_rank) for _ in range(self.predicted_hidden_layers)]) # project back to original space
        self.mlp = make_decoder_mlp(mlp_in_size, hidden_size, self.num_output_layer_weights, n_layers=decoder_hidden_layers,
                                    activation=decoder_activation)

    def forward(self, x, y_src):
        # x is samples x batch x emsize
        hidden_size = self.predicted_hidden_layer_size
        # summary layer goes from per-sample to per-dataset representations
        x_summary = self.summary_layer(x, y_src)
        if self.predicted_hidden_layers != 0 or self.decoder_type != "class_average":
            x_summary = x_summary.reshape(x.shape[1], self.summary_layer.out_size)
        res = self.mlp(x_summary)
        assert res.shape[-1] == self.num_output_layer_weights

        # let's confuse ourselves by storing them in the opposite order!
        def take_weights(res, shape):
            if len(shape) == 1:
                size = shape[0]
            elif len(shape) == 2:
                size = shape[0] * shape[1]
            else:
                raise ValueError("Only 1D and 2D shapes are supported")
            return res[:, :size].reshape(-1, *shape), res[:, size:]

        if self.predicted_hidden_layers == 0:
            if self.decoder_type == "class_average":
                w = res[:, :, :-1]
                w = w.transpose(1, 2)
                b = res[:, :, -1]
                return (b, w),
            else:
                w, next_res = take_weights(res, (self.in_size, self.n_out))
                b, next_res = take_weights(next_res, (self.n_out,))
                assert next_res.shape[1] == 0
                return (b, w), 
        if self.weight_embedding_rank is not None:
            second_shape = self.weight_embedding_rank
        else:
            second_shape = hidden_size # 512

        # output layer
        w2, next_res = take_weights(res, (hidden_size, self.n_out))
        b2, next_res = take_weights(next_res, (self.n_out,))

        # input layer
        w1, next_res = take_weights(next_res, (self.in_size, second_shape))
        b1, next_res = take_weights(next_res, (hidden_size,))
        result = [(b1, w1)]
        for _ in range(self.predicted_hidden_layers - 1):
            w, next_res = take_weights(next_res, (hidden_size, second_shape))
            b, next_res = take_weights(next_res, (hidden_size,))
            result.append((b, w))
        assert next_res.shape[1] == 0
        result.append((b2, w2))

        return result
    
class TreeDecoder(nn.Module):
    def __init__(self, depth=2, emsize=512, n_out=10, in_size=100, decoder_type='class_average', decoder_activation='relu',
                 decoder_hidden_layers=1, hidden_size=1024, tree_type='poetree', n_estimators=1, shared_unembedding=False,
                 weight_embedding_rank=None, low_rank_weights=False, constant_flag=False, tree_decoder_type='default'):
        super().__init__()
        self.depth = depth
        self.emsize = emsize
        self.n_out = n_out
        self.insize = in_size

        # By default, it is a complete tree
        self.num_leaf_nodes = 2 ** depth
        self.num_inner_nodes = self.num_leaf_nodes - 1
        self.num_nodes = self.num_leaf_nodes + self.num_inner_nodes

        self.summary_layer = SummaryLayer(emsize=emsize, n_out=n_out, decoder_type=decoder_type)

        mlp_in_size = self.summary_layer.out_size
        self.tree_type = tree_type
        self.n_estimators = n_estimators
        self.weight_embedding_rank = weight_embedding_rank if low_rank_weights else None
        self.constant_flag = constant_flag
        self.tree_decoder_type = tree_decoder_type
        self.shared_unembedding = shared_unembedding

        
        if self.tree_type == 'grad_tree':
            self.input_feature = self.insize
            self.num_inner_parameters = self.n_estimators * self.num_inner_nodes * self.insize * 2 # feature selector and threshold, matrix I + T
            self.num_inner_nodes_to_generate = self.n_estimators * self.num_inner_nodes * 2
            self.num_leaf_parameters = self.n_estimators * self.n_out * self.num_leaf_nodes
            self.num_output_layer_weights = self.num_inner_parameters + self.num_leaf_parameters
            self.per_node_parameter = self.insize * 2 + self.n_out # parameters per node, for value transformer            
        else:
            raise ValueError(f"Unknown tree type: {tree_type}")
        

        # if low_rank_weights, we will predict a low rank factorization of the weights, otherwise we will predict the full weights
        self.weight_embedding_rank = weight_embedding_rank if low_rank_weights else None
        if self.weight_embedding_rank is not None:
            # generate parameters for input layers:
            if self.tree_type == 'grad_tree':
                self.num_input_parameters = self.insize * self.weight_embedding_rank # input layer
            else:
                raise ValueError(f"Unknown tree type: {tree_type}")

            self.shared_weights = nn.ParameterList([nn.Parameter(torch.randn(
                self.weight_embedding_rank, self.num_inner_nodes_to_generate) / self.weight_embedding_rank)]) # project back to original space
            # right now only one layer
            self.num_output_layer_weights = self.num_input_parameters + self.num_leaf_parameters
        
        if self.tree_decoder_type == 'default':
            self.mlp = make_decoder_mlp(mlp_in_size, hidden_size, self.num_output_layer_weights, n_layers=decoder_hidden_layers,
                                activation=decoder_activation)
        elif self.tree_decoder_type == 'transformer':
            self.mlp = TreeTransformerDecoder(depth=depth, in_size=mlp_in_size, hidden_size=32, 
                                              inner_size=self.num_inner_parameters, leaf_size=self.num_leaf_parameters, 
                                              out_size=self.num_output_layer_weights, n_estimators=self.n_estimators,
                                              n_layers=8, activation=decoder_activation, n_heads=4, shared_unembedding=self.shared_unembedding)
                
        # print('Constant flag:', self.constant_flag)
        if self.constant_flag:
            self.constant_parameter = nn.Parameter(torch.randn(self.num_output_layer_weights))
        else:
            self.constant_parameter = None
        num_param_decoder = 0        
        for param in self.mlp.parameters():
            num_param_decoder += param.numel()
        # print(f"Number of parameters in tree decoder {self.tree_decoder_type}: {num_param_decoder/1e6:.2f}M")
        # print('Using tree decoder {}'.format(self.tree_decoder_type))

    def forward(self, x, y_src):
        """
        x is sequence first
        """
        x_summary: torch.Tensor = self.summary_layer(x, y_src) # batch_size, #classs, embed_dim
        x_summary = x_summary.reshape(x.shape[1], self.summary_layer.out_size)
        res: torch.Tensor = self.mlp(x_summary) # batch, num_output_layer_weights
        if self.constant_flag:
            res = res + self.constant_parameter

        # print(res.shape, self.num_output_layer_weights)
        assert res.shape[-1] == self.num_output_layer_weights # or res.shape[-1] == self.per_node_parameter

        def take_weights(res, shape):
            if len(shape) == 1:
                size = shape[0]
            elif len(shape) == 2:
                size = shape[0] * shape[1]
            elif len(shape) == 3:
                size = shape[0] * shape[1] * shape[2]
            else:
                raise ValueError("Only 1D and 2D shapes are supported")
            return res[:, :size].reshape(-1, *shape), res[:, size:]
        
        ret_list = []
        
        if self.weight_embedding_rank is None:
            if self.tree_type == 'grad_tree':
                feature_selector, next_res = take_weights(res, (self.n_estimators, self.num_inner_nodes, self.insize))
                threshold, next_res = take_weights(next_res, (self.n_estimators, self.num_inner_nodes, self.insize))
                leaf_node_weights, next_res = take_weights(next_res, (self.n_estimators, self.num_leaf_nodes, self.n_out))
                ret_list.extend([feature_selector, threshold, leaf_node_weights])
            else:
                raise ValueError(f"Unknown tree type: {self.tree_type}")
        else: #low-rank deprecated, not sure if we want to keep it
            if self.tree_type == 'grad_tree':
                low_rank_input, next_res = take_weights(res, (self.insize, self.weight_embedding_rank)) 
                leaf_node_weights, next_res = take_weights(next_res, (self.n_estimators, self.num_leaf_nodes, self.n_out))
                ret_list.extend([low_rank_input, leaf_node_weights])
            else:
                raise ValueError(f"Unknown tree type: {self.tree_type}")
        # if self.tree_decoder_type != 'value_transformer': # there will be parameters left for value transformer
        assert next_res.shape[1] == 0
        return *ret_list,

