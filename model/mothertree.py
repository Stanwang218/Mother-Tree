import torch, wandb
import torch.nn as nn
from torch.nn import TransformerEncoder

from model.encoders import OneHotAndLinear, Linear
from model.decoders import MLPModelDecoder, TreeDecoder
from model.tree import SoftDecisionTreeNeural
from model.layer import TransformerEncoderLayer, TransformerEncoderSimple, _SSMaxAttentionWrapper
from model.utils_torch import SeqBN, get_init_method


class MLPModelPredictor(nn.Module):
    def extract_low_rank_poetree_parameters(self, output, y, single_eval_pos=None):
        low_rank_input, leaf_node_weights = self.decoder(output, y[:single_eval_pos])
        inner_node_weights = torch.einsum('bfr, rj -> bfj', low_rank_input, self.decoder.shared_weights[0]) # batch, features, inner_nodes_parameters
        inner_node_weights = inner_node_weights.transpose(1, 2).reshape(inner_node_weights.shape[0], -1, self.max_features + 1) # batch, inner_nodes_parameters, features
        return inner_node_weights, leaf_node_weights

    def extract_low_rank_poetree_value_transformer_parameters(self, output, y, single_eval_pos=None):
        low_rank_input, leaf_node_weights = self.decoder(output, y[:single_eval_pos]) # batch, #nodes, rank
        inner_node_weights = torch.einsum('bfr, rj -> bfj', low_rank_input, self.decoder.shared_weights[0]) # batch, features, inner_nodes_parameters
        leaf_node_weights = torch.einsum('bfr, rj -> bfj', leaf_node_weights, self.decoder.shared_weights[0]) # batch, features, leaf_nodes_parameters
        # print(inner_node_weights.shape, leaf_node_weights[:, :, self.max_features + 1: ].shape)
        return inner_node_weights[:, :, :self.max_features + 1], leaf_node_weights[:, :, self.max_features + 1: ].transpose(1, 2)

    def extract_low_rank_grad_tree_parameters(self, output, y, single_eval_pos=None):
        low_rank_input, leaf_weights = self.decoder(output, y[:single_eval_pos])
        original_space_input = torch.einsum('bfr, rj -> bfj', low_rank_input, self.decoder.shared_weights[0]) # batch, features, inner_nodes_parameters
        feature_selector, thresholds = original_space_input.split(
            [self.n_estimators * self.num_inner_nodes, self.n_estimators * self.num_inner_nodes], dim=-1)
        assert feature_selector.shape == thresholds.shape
        feature_selector = feature_selector.transpose(1, 2).reshape(
            feature_selector.shape[0], self.n_estimators, self.num_inner_nodes, self.max_features
            ) # batch, n_estimators, num of inner nodes, features
        thresholds= thresholds.transpose(1, 2).reshape(feature_selector.shape[0], self.n_estimators, self.num_inner_nodes, self.max_features) # batch, inner_nodes_parameters, features
        return feature_selector, thresholds, leaf_weights
    

    def forward(self, src, single_eval_pos=None, temperature=1.0):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'
        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        _, x, y = src
        x_enc = self.encoder(x)
        if self.y_encoder is None:
            enc_train = x_enc[:single_eval_pos]
        else:
            mask = y == -100
            y[mask] = 0
            y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=self.n_out).float()

            if y_one_hot.ndim == 3:
                y_one_hot[:, :, 0][mask] = 0
            else:
                y_one_hot[:, 0][mask] = 0
                y_one_hot = y_one_hot.unsqueeze(1)

            dummy_predict = torch.ones_like(y_one_hot) / self.n_out
            y_enc = self.y_encoder(y_one_hot - dummy_predict, boosting=True)
            enc_train = x_enc[:single_eval_pos] + y_enc[:single_eval_pos]
        if self.decoder_type in ["special_token", "special_token_simple"]:
            enc_train = torch.cat([self.token_embedding.repeat(1, enc_train.shape[1], 1), enc_train], 0)
        elif self.decoder_type == "class_tokens":
            if not isinstance(self.y_encoder, OneHotAndLinear):
                raise ValueError("class_tokens decoder type is only supported with OneHotAndLinear y_encoder")
            repeated_class_tokens = self.y_encoder.weight.T.unsqueeze(1).repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat([repeated_class_tokens, enc_train], 0)

        output = self.inner_forward(enc_train) # transformer encoder output
        x_test_nona = torch.nan_to_num(x[single_eval_pos:], nan=0) # seq, batch, features
        self.num_feature = x_test_nona.shape[-1]
        # print(x_test_nona.shape)
        if isinstance(self.decoder, TreeDecoder):
            assert hasattr(self, "tree_predictor"), "tree_predictor attribute not found. Make sure to initialize the MotherNet with predictor_type='Tree' to use TreeDecoder."
            assert isinstance(self.tree_predictor, SoftDecisionTreeNeural), "tree_predictor is not an instance of SoftDecisionTreeNeural. Make sure to initialize the MotherNet with predictor_type='Tree' to use TreeDecoder."
            if self.tree_type == 'poetree':
                if self.decoder.weight_embedding_rank is not None:
                    if self.decoder.tree_decoder_type == 'value_transformer':
                        self.inner_node_weights, self.leaf_node_weights = self.extract_low_rank_poetree_value_transformer_parameters(output, y, single_eval_pos)
                    else:
                        self.inner_node_weights, self.leaf_node_weights = self.extract_low_rank_poetree_parameters(output, y, single_eval_pos)
                    # low_rank_input, self.leaf_node_weights = self.decoder(output, y[:single_eval_pos])
                    # low_rank_input = low_rank_input.unsqueeze(0) # batch, insize, rank
                    # self.inner_node_weights = torch.einsum('bfr, rj -> bfj', low_rank_input, self.decoder.shared_weights[0]) # batch, features, inner_nodes_parameters
                    # self.inner_node_weights = self.inner_node_weights.transpose(1, 2).reshape(self.inner_node_weights.shape[0], -1, num_feature) # batch, inner_nodes_parameters, features
                else:
                    self.inner_node_weights, self.leaf_node_weights = self.decoder(output, y[:single_eval_pos])
                    # self.tree_predictor._init_node_weight(inner_node_weights, leaf_node_weights)
                d = {
                    'inner_node_weights': self.inner_node_weights,
                    'leaf_node_weights': self.leaf_node_weights,
                    'temperature': temperature
                }
                h = self.tree_predictor(x_test_nona, **d)
                del self.inner_node_weights
                del self.leaf_node_weights
            elif self.tree_type == 'grad_tree':
                if self.decoder.weight_embedding_rank is not None:
                    self.feature_selector, self.thresholds, self.leaf_weights = self.extract_low_rank_grad_tree_parameters(output, y, single_eval_pos)
                else:
                    self.feature_selector, self.thresholds, self.leaf_weights = self.decoder(output, y[:single_eval_pos])
                d = {
                    'feature_selector': self.feature_selector,
                    'thresholds': self.thresholds,
                    'leaf_weights': self.leaf_weights,
                    'temperature': temperature
                }
                # print(self.feature_selector.shape, self.thresholds.shape, self.leaf_weights.shape)

                # batch_size, # ensemblers, num_inner_nodes, insize
                h = self.tree_predictor(x_test_nona, **d)
                del self.feature_selector
                del self.thresholds
                del self.leaf_weights
            else:
                raise ValueError(f"Unsupported tree type: {self.tree_type}")
            
            if h['logits'].isnan().all():
                print("NAN")
                raise ValueError("NAN")
            return h

        elif isinstance(self.decoder, MLPModelDecoder):
            (b1, w1), *layers = self.decoder(output, y[:single_eval_pos])
            # seq, batch, 100 and batch, 100, low_rank
            # seq, batch, 100, 1 and 1, batch, 100, low_rank

            h = (x_test_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2) # seq_len, batch, low_rank

            if self.decoder.weight_embedding_rank is not None and len(layers):
                h = torch.matmul(h, self.decoder.shared_weights[0])
            h = h + b1

            for i, (b, w) in enumerate(layers):
                if self.predicted_activation == "relu":
                    h = torch.relu(h)
                elif self.predicted_activation == "gelu":
                    h = torch.nn.functional.gelu(h)
                else:
                    raise ValueError(f"Unsupported predicted activation: {self.predicted_activation}")
                h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                if self.decoder.weight_embedding_rank is not None and i != len(layers) - 1:
                    # last layer has no shared weights
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
                h = h + b

            if h.isnan().all():
                print("NAN")
                raise ValueError("NAN")
            h = h.transpose(0, 1) # batch, seq_len, n_out
            return h


class MotherTree(MLPModelPredictor):
    def __init__(self, *, n_out, emsize, nhead, nhid_factor, nlayers, n_features, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_type="output_attention", predicted_hidden_layer_size=None,
                 decoder_embed_dim=2048, classification_task=True,
                 decoder_hidden_layers=1, decoder_hidden_size=None, predicted_hidden_layers=1, weight_embedding_rank=None, y_encoder=None,
                 low_rank_weights=False, tabpfn_zero_weights=True, decoder_activation="relu", predicted_activation="relu",
                 predictor_type = 'MLP', shared_unembedding=False, icl_ssmax = False, **kwargs):
        super().__init__()
        self.classification_task = classification_task
        # decoder activation = "relu" is legacy behavior
        nhid = emsize * nhid_factor
        # mothernet has batch_first=False, unlike all the other models.
        def encoder_layer_creator(): return TransformerEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, batch_first=False, ssmax = icl_ssmax)
        self.transformer_encoder = TransformerEncoderSimple(encoder_layer_creator, nlayers)
        backbone_size = sum(p.numel() for p in self.transformer_encoder.parameters())
        if wandb.run: wandb.log({"backbone_size": backbone_size})
        # print("Number of parameters in backbone: ", backbone_size)

        self.decoder_activation = decoder_activation
        self.emsize = emsize
        self.max_features = n_features
        self.encoder = Linear(n_features, emsize, replace_nan_by_zero=True)
        self.y_encoder = y_encoder_layer
        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_out = n_out
        self.nhid = nhid
        self.decoder_type = decoder_type
        self.inner_prediction = True if kwargs.get('tree_decoder', None) == 'value_transformer' else False
        decoder_hidden_size = decoder_hidden_size or nhid
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.predicted_activation = predicted_activation
        self.constant_flag = kwargs.get('constant_param', False)
        assert predictor_type in ['MLP', 'Tree'], "Unsupported predictor type. Supported types are 'MLP' and 'Tree'."
        if predictor_type == 'MLP':
            self.decoder = MLPModelDecoder(emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out, decoder_type=self.decoder_type,
                                       predicted_hidden_layer_size=predicted_hidden_layer_size, embed_dim=decoder_embed_dim,
                                       decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, predicted_hidden_layers=predicted_hidden_layers,
                                       weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights, decoder_activation=decoder_activation,
                                       in_size=n_features)
        elif predictor_type == 'Tree':
            if kwargs.get('depth', None) is None:
                kwargs["depth"] = 5  # default depth for tree predictor
            if kwargs.get('lambda_', None) is None:
                kwargs['lambda_'] = 0.001  # default lambda for tree predictor
            if kwargs.get('prediction', None) is None:
                kwargs['prediction'] = 'mean'
            if kwargs.get('tree_type', None) is None:
                kwargs['tree_type'] = 'poetree'
            if kwargs.get('n_estimators', None) is None:
                kwargs['n_estimators'] = 1
            if kwargs.get('gumbel_softmax', None) is None:
                kwargs['gumbel_softmax'] = True
            if kwargs.get('hard_gumbel', None) is None:
                kwargs['hard_gumbel'] = False
            if kwargs.get('constant_param', None) is None:
                kwargs['constant_param'] = False
            if kwargs.get('logistic_temperature', None) is None:
                kwargs['logistic_temperature'] = 1.0
            if kwargs.get('tree_decoder', None) is None:
                kwargs['tree_decoder'] = 'default'

            self.num_inner_nodes = 2 ** kwargs["depth"] - 1
            self.n_estimators = kwargs['n_estimators']
            # print('Gumbel Softmax', kwargs.get('gumbel_softmax', None), 'Hard Gumbel', kwargs.get('hard_gumbel', None))


            self.tree_type = kwargs['tree_type']
            self.decoder = TreeDecoder(depth=kwargs["depth"], emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out, decoder_type=self.decoder_type,
                                       decoder_hidden_layers=decoder_hidden_layers, in_size=n_features, shared_unembedding=shared_unembedding,
                                       tree_type=kwargs['tree_type'], n_estimators=kwargs['n_estimators'], 
                                       low_rank_weights=low_rank_weights, weight_embedding_rank=weight_embedding_rank, 
                                       constant_flag=kwargs['constant_param'], tree_decoder_type=kwargs['tree_decoder'])
            
            self.tree_predictor = SoftDecisionTreeNeural(input_dim=n_features, 
                                                         max_depth=kwargs["depth"],
                                                         output_dim=n_out,
                                                         grow_incrementally=False,
                                                         n_estimators=kwargs['n_estimators'],
                                                         gumbel_softmax=kwargs['gumbel_softmax'],
                                                         gumbel_hard=kwargs['hard_gumbel'],
                                                         inner_prediction=self.inner_prediction)
            # num_param = 0
            # for param in self.decoder.parameters():
            #     num_param += param.numel()
            # print(f"Number of parameters in decoder: {num_param}")
            # for name, param in self.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.shape)
            # breakpoint()
        else:
            raise ValueError(f"Unsupported predictor type: {predictor_type}")
        
        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))

        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for layer in self.transformer_encoder.layers:
                nn.init.zeros_(layer.linear2.weight)
                nn.init.zeros_(layer.linear2.bias)
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    if isinstance(attn, _SSMaxAttentionWrapper):
                        nn.init.zeros_(attn.mha.out_proj.weight)
                        nn.init.zeros_(attn.mha.out_proj.bias)
                    else:
                        nn.init.zeros_(attn.out_proj.weight)
                        nn.init.zeros_(attn.out_proj.bias)

    def inner_forward(self, train_x):
        return self.transformer_encoder(train_x)
    def forward(self, src, single_eval_pos=None, temperature=1):
        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        _, x, y = src
        assert x.ndim == 3, "Input x should have 3 dimensions (seq_len, batch_size, num_features)"

        x = x.unsqueeze(-1) # add a dimension for the linear encoder

        x_enc = self.encoder(x) # seq_len, batch_size, feature, embed_dim
            
        # Add positional embeddings
        if self.positional_encoding == 'subspace':
            feature_positional_embeddings = self.feature_positional_embedding_embeddings(x_enc[..., :self.emsize // 4]) # seq_len, batch_size, feature, embed_dim
            embs = torch.randn(
                (x_enc.shape[2], x_enc.shape[3] // 4),
                device=x.device,
                dtype=x.dtype,
            )
            # Random numbers vary between different devices, even with a fixed seed.
            # Thus we use the pregenerated column embeddings where possible to ensure
            # they are consistent between pretraining and inference.
            if embs.shape[1] == 48 and self.random_embedding_seed == 42:  # 192 // 4
                embs[:2000] = self.pre_generated_column_embeddings[: embs.shape[0]].to(
                    device=embs.device, dtype=embs.dtype
                )
            embs = self.feature_positional_embedding_embeddings(embs) # n_features, embed_dim
            embs = embs.unsqueeze(0).unsqueeze(0) # 1, 1, n_features, embed_dim
            x_enc = x_enc + embs
        
        enc_train = x_enc[:single_eval_pos]
        if self.y_encoder is not None:
            y_enc = self.y_encoder(y.unsqueeze(-1) if len(y.shape) < len(x.shape) else y) # seq, bz, embed
            y_enc = y_enc.unsqueeze(-2) 
        enc_train = torch.cat([enc_train, y_enc[:single_eval_pos]], dim=-2)
        output = self.inner_forward(enc_train) # transformer encoder output
        
        x_test_nona = torch.nan_to_num(x[single_eval_pos:], nan=0).squeeze() # seq, batch, features
        self.num_feature = x_test_nona.shape[-1]
        # print(x_test_nona.shape)
        if isinstance(self.decoder, TreeDecoder):
            assert hasattr(self, "tree_predictor"), "tree_predictor attribute not found. Make sure to initialize the MotherNet with predictor_type='Tree' to use TreeDecoder."
            assert isinstance(self.tree_predictor, SoftDecisionTreeNeural), "tree_predictor is not an instance of SoftDecisionTreeNeural. Make sure to initialize the MotherNet with predictor_type='Tree' to use TreeDecoder."
            if self.tree_type == 'grad_tree':
                if self.decoder.weight_embedding_rank is not None:
                    self.feature_selector, self.thresholds, self.leaf_weights = self.extract_low_rank_grad_tree_parameters(output, y, single_eval_pos)
                else:
                    self.feature_selector, self.thresholds, self.leaf_weights = self.decoder(output, y[:single_eval_pos])
                d = {
                    'feature_selector': self.feature_selector,
                    'thresholds': self.thresholds,
                    'leaf_weights': self.leaf_weights,
                    'temperature': temperature
                }
                # print(self.feature_selector.shape, self.thresholds.shape, self.leaf_weights.shape)

                # batch_size, # ensemblers, num_inner_nodes, insize
                h = self.tree_predictor(x_test_nona, **d)
                del self.feature_selector
                del self.thresholds
                del self.leaf_weights
            else:
                raise ValueError(f"Unsupported tree type: {self.tree_type}")
            
            if h['logits'].isnan().all():
                print("NAN")
                raise ValueError("NAN")
            return h

        elif isinstance(self.decoder, MLPModelDecoder):
            (b1, w1), *layers = self.decoder(output, y[:single_eval_pos])
            # seq, batch, 100 and batch, 100, low_rank
            # seq, batch, 100, 1 and 1, batch, 100, low_rank

            h = (x_test_nona.unsqueeze(-1) * w1.unsqueeze(0)).sum(2) # seq_len, batch, low_rank

            if self.decoder.weight_embedding_rank is not None and len(layers):
                h = torch.matmul(h, self.decoder.shared_weights[0])
            h = h + b1

            for i, (b, w) in enumerate(layers):
                if self.predicted_activation == "relu":
                    h = torch.relu(h)
                elif self.predicted_activation == "gelu":
                    h = torch.nn.functional.gelu(h)
                else:
                    raise ValueError(f"Unsupported predicted activation: {self.predicted_activation}")
                h = (h.unsqueeze(-1) * w.unsqueeze(0)).sum(2)
                if self.decoder.weight_embedding_rank is not None and i != len(layers) - 1:
                    # last layer has no shared weights
                    h = torch.matmul(h, self.decoder.shared_weights[i + 1])
                h = h + b

            if h.isnan().all():
                print("NAN")
                raise ValueError("NAN")
            h = h.transpose(0, 1) # batch, seq_len, n_out
            return h


        return super().forward(src, single_eval_pos, temperature)