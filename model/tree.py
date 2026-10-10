import torch
import torch.nn as nn
import numpy as np
import math
import graphviz
from typing import List, Dict, Optional
from scipy.special import softmax
from torch.nn.functional import sigmoid, softsign, gumbel_softmax
from typing import Optional
from entmax import sparsemax, entmax15, entmax_bisect

ALL_NODE_MODULES = [
    'inner_nodes',
    'leaf_nodes',
    'history_leaf_nodes',
    'observation_leaf_nodes'
]



def create_decision_tree(depth):
    tree = create_decision_stump()
    while (
        tree.depth < depth or
        len(tree.inner_nodes) < 2 ** depth - 1
    ):
        leaf_node_indices = list(tree.leaf_nodes)
        leaf_node_indices.sort()  # Sort from smallest to largest
        tree = tree.split_node(index=leaf_node_indices[0])
    return tree


def create_decision_stump():
    root = Node(index=0)
    stump = Tree(nodes=[root])
    return stump.split_node(index=0)


class Node:
    def __init__(
        self,
        index,
        parent=None,
        left_child=None,
        right_child=None
    ):
        self.index = index
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child

    def remove_child(self, child):
        if child == self.left_child:
            self.left_child = None
        elif child == self.right_child:
            self.right_child = None
        else:
            raise ValueError(
                f"Child node with index {child.index} is not a child "
                f"of node with index {self.index}."
            )

    def remove_children(self):
        self.left_child = None
        self.right_child = None
    
    def add_children(self):
        left_child = Node(
            index=2 * self.index + 1,
            parent=self,
        )
        right_child = Node(
            index=2 * self.index + 2,
            parent=self,
        )
        self.left_child = left_child
        self.right_child = right_child
    
    @property
    def layer_index(self):
        return int(math.log2(self.index + 1)) # [2 ** d + 1, 2 ** (d + 1) - 2]

    @property
    def is_suboptimal(self):
        return getattr(
            self, '_is_suboptimal', self.is_leaf
        )
    
    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def children(self):
        return [child for child in [self.left_child, self.right_child] if child is not None]

class Tree:
    def __init__(self, nodes:List[Node]=[]):
        self.nodes : dict[int, Node] = {node.index: node for node in nodes}
        self._update()

    def split_node(self, index):
        node : Node = self.nodes[index]
        assert node.index == index
        node.add_children()
        for child in node.children:
            self.nodes[child.index] = child
        self._update()
        return self

    def unsplit_node(self, index):
        node:Node = self.nodes[index]
        assert node.index == index
        for child in node.children:
            self.nodes.pop(child.index)
        node.remove_children()
        node._is_suboptimal = False
        self._update()
        return self

    def remove_leaf_node(self, index):
        node = self.nodes[index]
        assert node.index == index
        assert node.is_leaf
        self.nodes.pop(index)
        node.parent.remove_child(node)
        self._update()
        return self

    def _update_index_of_descendants(self, node):
        if node.left_child is not None:
            node.left_child.index = 2 * node.index + 1
            self._update_index_of_descendants(node.left_child)
        if node.right_child is not None:
            node.right_child.index = 2 * node.index + 2
            self._update_index_of_descendants(node.right_child)
        return self

    def remove_single_child_nodes(self):
        # Check which nodes are to be removed.
        removed_inner_node_indices = []
        for node in self.nodes.values():
            if len(node.children) == 1:
                assert not node.is_leaf
                removed_inner_node_indices.append(self.inner_node_index(node))

        # Remove nodes and update indices and connections.
        sorted_nodes = list(self.sorted_nodes.items())
        for index, node in sorted_nodes[::-1]:  # Iterate from leaves to root
            children = node.children
            
            if len(children) == 1:
                child = children.pop()
                parent = node.parent

                # Connect child and parent.
                child.parent = parent
                if parent and parent.left_child == node:
                    parent.left_child = child
                elif parent and parent.right_child == node:
                    parent.right_child = child

                # Update index of child and its descendants.
                child.index = node.index
                self._update_index_of_descendants(child)
                
                # Remove node.
                self.nodes.pop(index)
        
        self.nodes = {
            node.index: node for node in self.nodes.values()
        }
        self._update()
        
        return removed_inner_node_indices
    
    def inner_node_index(self, node):
        assert not node.is_leaf
        return list(self.sorted_inner_nodes).index(node.index)

    def leaf_node_index(self, node):
        assert node.is_leaf
        return list(self.sorted_leaf_nodes).index(node.index)

    def get_sorted_inner_nodes_in_layer(self, layer_index):
        out = {}
        for index, node in self.sorted_inner_nodes.items():
            if node.layer_index == layer_index:
                out[index] = node
        return out
    
    @property
    def suboptimal_leaves(self):
        leaf_nodes = list(self.leaf_nodes.values())
        suboptimal_leaves = [
            leaf for leaf in leaf_nodes if leaf.is_suboptimal
        ]
        # We want to sort from largest to smallest index; note that
        # `self.sorted_leaf_nodes` contains the nodes sorted from
        # left to right.
        suboptimal_leaves.sort(key=lambda leaf: leaf.index, reverse=True)
        return suboptimal_leaves

    @property
    def inner_nodes(self) -> Dict[int, Node]:
        return self.sorted_inner_nodes
    
    @property
    def leaf_nodes(self):
        return self.sorted_leaf_nodes

    def _update_sorted_nodes(self):
        self.sorted_nodes = dict(sorted(self.nodes.items()))
    
    def _update_sorted_inner_nodes(self):
        self.sorted_inner_nodes = {
            index: node for index, node in self.sorted_nodes.items()
            if not node.is_leaf
        }
    
    def _update_depth(self):
        if len(self.inner_nodes) == 0:
            self.depth = 0
        else:
            self.depth = max(
                [node.layer_index for node in self.inner_nodes.values()]
            ) + 1
        
    def _update_sorted_leaf_nodes(self):
        def sorter(item):
            index, node = item
            if node.layer_index == self.depth:
                return index
            else:
                # Propagate to the bottom of the tree and return the
                # left-most descendant of the node.
                return get_leaf_node_indices(self.depth, index)[0]
        leaf_nodes = {
            index: node for index, node in self.nodes.items() if node.is_leaf
        }
        self.sorted_leaf_nodes = dict(sorted(leaf_nodes.items(), key=sorter))
    
    def _update(self):
        self._update_sorted_nodes()
        self._update_sorted_inner_nodes()  # Requires `self.sorted_nodes`
        self._update_depth()  # Requires `self.sorted_inner_nodes`
        self._update_sorted_leaf_nodes()  # Requires `self.depth`
    
        self.tree_aligned_inner_node_indices_per_layer = {}
        self.layer_aligned_inner_node_indices_per_layer = {}

        full_inner_node_indices_per_layer : dict[int, List] = \
            get_node_indices_per_layer(self.depth) # indices for complete binary tree

        for i_layer in range(self.depth):
            full_inner_node_indices = full_inner_node_indices_per_layer[i_layer]
            inner_nodes = self.get_sorted_inner_nodes_in_layer(i_layer) # retrieve the inner nodes in a layer
            self.tree_aligned_inner_node_indices_per_layer[i_layer] = [
                self.inner_node_index(node) for node in inner_nodes.values()
            ]
            self.layer_aligned_inner_node_indices_per_layer[i_layer] = [
                index - full_inner_node_indices[0] for index in inner_nodes.keys()
            ] # indices in the complete tree but for current tree 
            # real index - the left index in complete tree
            # eliminate the intercept for the layer
        self.bottom_layer_aligned_leaf_node_indices = [ 
            get_leaf_node_indices(self.depth, index)[0]  # Select the index of the left-most leaf descendant
            if node.layer_index < self.depth else index 
            # if node.layer_index < self.depth, the node is terminated earlier
            for index, node in self.leaf_nodes.items()
        ]
        self.bottom_layer_aligned_leaf_node_indices.sort()
        # The index of the left-most leaf node always comes first
        # so we can shift the indices accordingly.
        shift = self.bottom_layer_aligned_leaf_node_indices[0]
        self.bottom_layer_aligned_leaf_node_indices = \
            [i - shift for i in self.bottom_layer_aligned_leaf_node_indices]

class LinearFromWeights(nn.Linear):
    def __init__(self, weights):
        super(LinearFromWeights, self).__init__(
            in_features=weights.shape[1],
            out_features=weights.shape[0],
            bias=False,
            device=weights.device,
            dtype=weights.dtype
        )
        with torch.no_grad():
            self.weight.copy_(weights)

class UnidimensionalInnerNodes(nn.Module):
    def __init__(self, weight, input_dim, hidden_dim=0):
        super(UnidimensionalInnerNodes, self).__init__()
        self.weight = weight # out_features, in_features
        self.input_dim = input_dim + 1 # Add bias
        self._split_sizes = [1, hidden_dim, input_dim]
    
    def forward(self, x):
        # X has shape (batch size, seq_num, num_features)
        x = x.reshape(-1, self.input_dim) # (batch size * seq_num, num_features)
        # Split the weights.
        w = torch.t(self.weight) # input_dim + 1, num_inner_nodes
        b, wh, wz = torch.split(w, self._split_sizes, dim=0)

        # Split the input.
        _, h, z = torch.split(x, self._split_sizes, dim=1) # 
        
        # Marginalize out the history into the bias term.
        #
        # If there is no hidden dimension, torch.mm(h, wh) returns
        # a zero tensor of shape (batch size, # inner nodes).
        bh = b + torch.mm(h, wh)  # Shape (batch size, # inner nodes)

        # Select the largest weight for each inner node.
        wmax, max_indices = torch.max(wz, dim=0)  # Shape (# inner nodes,)
        
        # Select the observations corresponding to the largest weights.
        zmax = z[:, max_indices]  # Shape (batch size, # inner nodes)

        # Compute the thresholds for each input data.
        thresholds = -(bh / wmax) # inner nodes (batch size, # inner nodes)

        # For each sample in the batch, compute
        # zmax > -(bh / wmax) if wmax > 0
        # zmax < -(bh / wmax) if wmax < 0
        condition = wmax > 0 # if the weight is large than zero
        input = zmax > thresholds
        other = zmax < thresholds
        return (
            torch.where(condition, input, other).type(torch.float32), # data to go to each inner node
            thresholds
        )



def get_node_indices_per_layer(depth)-> Dict[int, List]:
    node_indices_per_layer = {}
    i_start, i_stop = 0, 1
    for i_layer in range(depth):
        node_indices_per_layer[i_layer] = \
            list(range(i_start, i_stop))
        i_start = i_stop
        i_stop = i_start + 2 ** (i_layer + 1)
    return node_indices_per_layer


def get_leaf_node_indices(depth, index):
    result = []
    
    num_inner_nodes = 2 ** depth - 1
    num_leaf_nodes = 2 ** depth
    num_total_nodes = num_inner_nodes + num_leaf_nodes

    if index >= num_inner_nodes:
        return result

    def traverse(node_index, current_depth):
        if current_depth == (depth + 1):
            result.append(node_index)
        else:
            left_child_index = 2 * node_index + 1
            right_child_index = 2 * node_index + 2
            if left_child_index < num_total_nodes:
                traverse(left_child_index, current_depth + 1)
            if right_child_index < num_total_nodes:
                traverse(right_child_index, current_depth + 1)
    
    current_depth = int(math.log2(index + 1)) + 1

    traverse(index, current_depth)

    return result


def make_input_table(samples, feature_names):
    header = "".join(f"<TD><B>{name}</B></TD>" for name in feature_names)
    rows = f"<TR>{header}</TR>"

    for sample in samples:
        row = "".join(f"<TD>{value}</TD>" for value in sample)
        rows += f"<TR>{row}</TR>"

    table_html = (
        f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
        f"<TR><TD COLSPAN='{len(feature_names)}'><B>Input Data</B></TD></TR>"
        f"{rows}"
        f"</TABLE>>"
    )

    return table_html

def draw_tree(tree, inner_node_labels=None, leaf_node_labels=None, edge_attrs=None):
    graph = graphviz.Digraph()
    for node in tree.nodes.values():
        if node.is_leaf:
            if leaf_node_labels:
                label = leaf_node_labels[node.index]
            else:
                label = str(node.index)
            shape = 'circle'
        else:
            if inner_node_labels:
                label = inner_node_labels[node.index]
            else:
                label = str(node.index)
            shape = 'box'
        if shape == 'box':
            graph.node(str(node.index), label, shape=shape)
        else:
            graph.node(str(node.index), label, shape=shape,
                        width='0.5',
                        height='0.8',
                        fontsize='8',
                        fixedsize='true')
        if node.parent is not None:
            label = 'Y' if node.parent.right_child == node else 'N'
            edge_attr = edge_attrs[node.index] if edge_attrs is not None else {}
            if edge_attr.get('label'):
                label += '\n' + edge_attr.pop('label')
            graph.edge(str(node.parent.index), str(node.index), label=label, **edge_attr)
        
    return graph


# def trace_graph(fn, indent=0, depth=0):
#     if fn is None or depth > 10:
#         return
#     print(" " * indent, fn)
#     for next_fn, _ in fn.next_functions:
#         trace_graph(next_fn, indent + 2, depth=depth+1)


class GradTree(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        tree: Tree,
        n_estimators = 1,
        gumbel_softmax = False,
        hard_gumbel = False,
        inner_prediction = False,
        split_max_features = None,
    ):
        super(GradTree, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        self.tree: Tree = tree
        self.depth = tree.depth
        self.num_inner_nodes = len(tree.inner_nodes)
        self.num_leaf_nodes = len(tree.leaf_nodes)
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_hard = hard_gumbel
        self.inner_prediction = inner_prediction
        self.split_max_features = split_max_features  # None → argmax, int/str → multinomial sample

        self.feature_selector = nn.Parameter(
            torch.randn(self.n_estimators, self.num_inner_nodes, self.input_dim)
        ) # matrix I

        self.thresholds = nn.Parameter(
            torch.randn(self.n_estimators, self.num_inner_nodes, self.input_dim)
        ) # matrix T

        if self.inner_prediction:
            self.leaf_nodes = nn.Parameter(
                torch.randn(self.n_estimators, self.num_inner_nodes + self.num_leaf_nodes, self.output_dim)
            )
        else:
            self.leaf_nodes = nn.Parameter(
                torch.randn(self.n_estimators, self.num_leaf_nodes, self.output_dim)
            )

        # Prepare the tree structure for efficient computation.
        leaf_index = torch.arange(self.num_leaf_nodes)
        current_depth = torch.arange(1, self.depth + 1)   

        leaf_index = leaf_index.view(-1, 1)               # (l, 1)
        current_depth = current_depth.view(1, -1)         # (1, d)

        self.path_identifier = torch.floor(
                leaf_index / (2 ** (self.depth - current_depth))
            ) % 2

        self.path_identifier = self.path_identifier.float()         # (l, d)

        self.internal_node_index = (
                2 ** (current_depth - 1)
                + torch.floor(
                    leaf_index / (2 ** (self.depth - (current_depth - 1)))
                )
                - 1
            )
        # indexes paths from internal nodes to leaf nodes, shape (l, d)
        self.internal_node_index = self.internal_node_index.long()  # (l, d)

    def _resolve_split_max_features(self):
        """Resolve split_max_features to an integer, or None for argmax."""
        n = self.input_dim
        if self.split_max_features is None:
            return None
        if isinstance(self.split_max_features, int):
            return min(self.split_max_features, n)
        if self.split_max_features == "sqrt":
            return max(1, int(n ** 0.5))
        if self.split_max_features == "log2":
            return max(1, int(np.log2(n)))
        if isinstance(self.split_max_features, float):
            return max(1, int(self.split_max_features * n))
        return None

    def forward(self, X, **kwargs):
        feature_selector = kwargs.get('feature_selector', None)
        thresholds = kwargs.get('thresholds', None)
        leaf_weights = kwargs.get('leaf_weights', None)
        temperature = kwargs.get('temperature', None)

        if X.ndim == 2: # (batch_size, input_dim)
            X = X.unsqueeze(0) # Add sequence dimension, (seq, batch_size, input_dim)

        if X.ndim == 3: # (seq, batch_size, input_dim)
            X = X.transpose(0,1) # batch first
            batch_size = X.size(0)
            seq_len = X.size(1)

        path_probas, penalty, all_path_probas = self._forward(X, 
            feature_selector=feature_selector, thresholds=thresholds, temperature=temperature)
        path_probas = path_probas.view(batch_size, seq_len, self.n_estimators, -1) # batch_size, seq_len, num of estimators, num_leaf_nodes
        # bz * seq_len, n_estimators, num_leaf_nodes
        # n_estimators, num_leaf_nodes, output_dim
        effective_leaf_nodes = self.leaf_nodes if leaf_weights is None else leaf_weights
        if effective_leaf_nodes.device != path_probas.device:
            effective_leaf_nodes = effective_leaf_nodes.to(path_probas.device)

        if effective_leaf_nodes.ndim == 3:
            # Inference mode: shared tree parameters, leaf-only prediction
            if self.inner_prediction:
                effective_leaf_nodes = effective_leaf_nodes[:, self.num_inner_nodes:, :]  # (e, l, o)
            logits = torch.einsum("edo,bsed->bso", effective_leaf_nodes, path_probas)

        elif effective_leaf_nodes.ndim == 4:
            # Training mode: per-dataset tree parameters
            if effective_leaf_nodes.shape[0] != path_probas.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: leaf_weights has batch {effective_leaf_nodes.shape[0]}, "
                    f"but path probabilities have batch {path_probas.shape[0]}."
                )

            if self.inner_prediction:
                # effective_leaf_nodes: (b, e, num_inner_nodes + num_leaf_nodes, o)
                # Aggregate inner node + leaf node predictions, weighted by update counts
                leaf_index = path_probas.argmax(dim=-1)                  # (b, s, e)
                device = path_probas.device
                self.internal_node_index = self.internal_node_index.to(device)
                route = self.internal_node_index[leaf_index]             # (b, s, e, d)
                b_range = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)  # (b, 1, 1, 1)
                e_range = torch.arange(self.n_estimators, device=device).view(1, 1, -1, 1)  # (1, 1, e, 1)

                # Gather inner node predictions along the route: (b, s, e, d, o)
                inner_logits = effective_leaf_nodes[b_range, e_range, route, :]

                # Gather leaf prediction: (b, s, e, o)
                leaf_offset_idx = self.num_inner_nodes + leaf_index      # offset into leaf section
                leaf_logits = effective_leaf_nodes[b_range.squeeze(-1), e_range.squeeze(-1), leaf_offset_idx, :]

                # Weight by inverse update counts: leaf=1, parent=2, ..., root=2^D
                update_counts = 2.0 ** (self.depth - torch.arange(self.depth, device=device))
                update_counts = torch.cat([update_counts, torch.ones(1, device=device)])
                depth_weights = (1.0 / update_counts)
                depth_weights = depth_weights / depth_weights.sum()      # → [1,2,4,8]/15 for D=3

                all_logits = torch.cat([inner_logits, leaf_logits.unsqueeze(-2)], dim=-2)  # (b, s, e, d+1, o)
                logits = torch.einsum("bsedo,d->bso", all_logits, depth_weights)
            else:
                # Decoder may generate logits for all nodes (inner + leaf);
                # when inner_prediction is off, take only the leaf portion.
                if effective_leaf_nodes.shape[-2] > self.num_leaf_nodes:
                    effective_leaf_nodes = effective_leaf_nodes[:, :, self.num_inner_nodes:, :]
                logits = torch.einsum("bedo,bsed->bso", effective_leaf_nodes, path_probas)
        else:
            raise ValueError(f"`leaf_weights` must be 3D (e,l,o) or 4D (b,e,l,o), got shape {tuple(effective_leaf_nodes.shape)}.")

        return {
            'probas': (path_probas, all_path_probas),
            'penalties': (penalty,),
            'predictions': (logits,),
        }

    def _forward(self, X, feature_selector=None, thresholds=None, temperature=1.0):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        device = X.device
        # X = X.transpose(0,1)
        
        batch_size = X.size()[0]
        seq_len = X.size()[1]

        if (feature_selector is None) != (thresholds is None):
            raise ValueError("`feature_selector` and `thresholds` must be provided together.")
        effective_feature_selector = self.feature_selector if feature_selector is None else feature_selector
        effective_thresholds = self.thresholds if thresholds is None else thresholds
        if effective_feature_selector.device != X.device:
            effective_feature_selector = effective_feature_selector.to(X.device)
        if effective_thresholds.device != X.device:
            effective_thresholds = effective_thresholds.to(X.device)
        if effective_feature_selector.ndim != effective_thresholds.ndim:
            raise ValueError(
                f"`feature_selector` and `thresholds` rank mismatch: "
                f"{effective_feature_selector.ndim} vs {effective_thresholds.ndim}."
            )

        # X in the shape of 
        # (batch size, seq_len, input_dim) * (n_estimators, num_inner_nodes, input_dim) -> (batch size, seq_len, n_estimators, num_inner_nodes)
        # (n_estimators, num_inner_nodes, input_dim)
        # 
        if self.gumbel_softmax:
            feature_selector = gumbel_softmax(effective_feature_selector, tau=temperature, hard=self.gumbel_hard, dim=-1)
        else:
            feature_selector = entmax15(effective_feature_selector, dim=-1) # sparsify the feature selector
            k = self._resolve_split_max_features()
            if k is not None and k < self.input_dim:
                # ── Multinomial feature sampling (sklearn RF style) ──
                # Sample k features per (estimator, inner_node), then pick the
                # best among them.  Provides per-split diversity without needing
                # external column subsampling.
                orig_shape = feature_selector.shape  # (e,i,n) or (b,e,i,n)
                flat_probs = feature_selector.reshape(-1, self.input_dim)  # (*, n)
                sample_indices = torch.multinomial(flat_probs, num_samples=k, replacement=False)  # (*, k)
                mask = torch.zeros_like(flat_probs)
                mask.scatter_(-1, sample_indices, 1.0)
                masked_probs = flat_probs * mask
                max_in_sampled = masked_probs.argmax(dim=-1, keepdim=True)  # (*, 1)
                hard = torch.zeros_like(flat_probs)
                hard.scatter_(-1, max_in_sampled, 1.0)
                hard = hard.view(orig_shape)
            else:
                # ── Original argmax (no subsampling) ──
                hard = torch.zeros_like(feature_selector) #batch_size, n_estimators, num_inner_nodes, input_dim
                max_indices = feature_selector.argmax(dim=-1, keepdim=True) # n_estimators, num_inner_nodes, 1
                hard.scatter_(-1, max_indices, 1.0)
            feature_selector = feature_selector - (feature_selector - hard).detach()# straight through estimator
        if feature_selector.ndim == 3:
            # (e, i, n) -> (e, l, d, n)
            split_index_array_selected = feature_selector[:, self.internal_node_index, :] 
            # (n_estimators, num_leaf_nodes, depth, input_dim)
            # for each leaf node, the selected feature index in the internal node
            split_values_selected = effective_thresholds[:, self.internal_node_index, :]
            s1_sum = torch.einsum("eldn,eldn->eld", split_values_selected, split_index_array_selected).unsqueeze(0).unsqueeze(0) # add batch and sequence dimension
            s2_sum = torch.einsum("bsn,eldn->bseld", X, split_index_array_selected)
        elif feature_selector.ndim == 4:
            # (b, e, i, n) -> (b, e, l, d, n)
            # if feature_selector.shape[0] == 1 and X.shape[0] > 1:
            #     feature_selector = feature_selector.expand(X.shape[0], -1, -1, -1)
            #     effective_thresholds = effective_thresholds.expand(X.shape[0], -1, -1, -1)
            # if feature_selector.shape[0] != X.shape[0]:
            #     raise ValueError(
            #         f"Batch size mismatch: feature_selector has batch {feature_selector.shape[0]}, "
            #         f"but X has batch {X.shape[0]}."
            #     )
            split_index_array_selected = feature_selector[:, :, self.internal_node_index, :]
            split_values_selected = effective_thresholds[:, :, self.internal_node_index, :]
            s1_sum = torch.einsum("beldn,beldn->beld", split_values_selected, split_index_array_selected).unsqueeze(1)
            s2_sum = torch.einsum("bsn,beldn->bseld", X, split_index_array_selected)
        else:
            raise ValueError(
                f"`feature_selector` must be 3D (e,i,n) or 4D (b,e,i,n), got shape {tuple(feature_selector.shape)}."
            )

        node_result = (softsign(s1_sum - s2_sum) + 1) / 2   # (b, s, e, l, d)
        node_result_corrected = node_result - (node_result - torch.round(node_result)).detach() # left is one, right is one
        self.path_identifier = self.path_identifier.to(device)
        p = torch.prod(
                (1 - self.path_identifier) * node_result_corrected + # layers for going left
                self.path_identifier * (1 - node_result_corrected), # layers for going right
                dim=-1
            )   # (b, s, e, l), which leaf node is reached by each sample through each estimator
        
        return p, torch.tensor(0.0).to(device), None

    def _draw_tree(self, features, classes, edge_attrs=None):
        # For simplicity, we only visualize the first estimator.
        feature_selector = self.feature_selector[0].detach().cpu().numpy()
        thresholds = self.thresholds[0].detach().cpu().numpy()
        leaf_node_weights = self.leaf_nodes[0].detach().cpu().numpy()

        leaf_node_probas = softmax(leaf_node_weights, axis=1)
        leaf_node_preds = np.argmax(leaf_node_weights, axis=1).astype(int)
        
        inner_node_labels = {}
        leaf_node_labels = {}
        for node in self.tree.nodes.values():
            if node.is_leaf:
                index = self.tree.leaf_node_index(node)
                pred_class = leaf_node_preds[index]
                class_proba = leaf_node_probas[index, pred_class]
                label = f'{classes[pred_class]} ({class_proba:.2f})'
                leaf_node_labels[node.index] = label
            else:
                index = self.tree.inner_node_index(node)
                max_feature = feature_selector[index].argmax().item()
                if thresholds is not None:
                    label = features[max_feature] + f' > {thresholds[index, max_feature]:.2f}'
                else:
                    label = features[max_feature]
                inner_node_labels[node.index] = label
            
        return draw_tree(self.tree, inner_node_labels, leaf_node_labels, edge_attrs)



class SoftDecisionTreeNeural(nn.Module):
    def __init__(
        self,
        input_dim=100,
        output_dim=10,
        prediction='mean',
        grow_incrementally=False,
        max_depth=5,
        initial_depth=2,
        lambda_=0.001,
        n_estimators=1,
        gumbel_softmax = False,
        gumbel_hard = False,
        inner_prediction = False,
        split_max_features = None,
    ):
        super(SoftDecisionTreeNeural, self).__init__()
        assert prediction in ['mean', 'max']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_estimators = n_estimators
        self.lambda_ = lambda_
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_hard = gumbel_hard
        self.split_max_features = split_max_features
        # print('Gumbel Softmax parameters, softmax:', gumbel_softmax, 'hard:', gumbel_hard)
        if grow_incrementally:
            self.tree = create_decision_tree(initial_depth)
        else:
            self.tree = create_decision_tree(max_depth)

        self.num_inner_nodes = len(self.tree.inner_nodes)
        self.num_leaf_nodes = len(self.tree.leaf_nodes)



        self.sdt = GradTree(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tree=self.tree,
            n_estimators=self.n_estimators,
            gumbel_softmax=self.gumbel_softmax,
            hard_gumbel=self.gumbel_hard,
            inner_prediction=inner_prediction,
            split_max_features=self.split_max_features,
        )
    

    def _init_node_weight(self, param_list):
        with torch.no_grad():
            feature_selector = param_list[0]
            thresholds = param_list[1]
            leaf_weights = param_list[2]
            if isinstance(feature_selector, np.ndarray):
                feature_selector = torch.from_numpy(feature_selector)
            if isinstance(thresholds, np.ndarray):
                thresholds = torch.from_numpy(thresholds)
            if isinstance(leaf_weights, np.ndarray):
                leaf_weights = torch.from_numpy(leaf_weights)
            self.sdt.feature_selector.copy_(feature_selector)
            self.sdt.thresholds.copy_(thresholds)
            self.sdt.leaf_nodes.copy_(leaf_weights)

            
    def _align_axes(self):
        self.sdt._align_axes()

    def draw_tree(self, features, classes, thresholds=None, edge_attrs=None, input_data=None, probas=None):
        g = self.sdt._draw_tree(features, classes)
        
        table_html = make_input_table(np.round(input_data[: 5], 2), features)
        if input_data is not None:
            g.node(
                "input_data",
                label=table_html,
                shape="plaintext"
            )

            with g.subgraph() as s:
                s.attr(rank='min')
                s.node("input_data")

            g.edge("input_data", "0", style="invis")
        
        if probas is not None:
            leaf_probas = probas[0].squeeze().sum(0)
            attrs = {}
            for node in self.tree.nodes.values():
                if node.is_leaf:
                    original_index = node.index 
                    index = self.tree.leaf_node_index(node)
                    # print(index, leaf_probas[index])
                    if leaf_probas[index] == 0.0: 
                        continue
                    attrs['color'] = 'red'
                    g.node(str(original_index), **attrs)
                    
        return g

    def examine_uninformative_paths(self):
        feature_selector = self.sdt.feature_selector.detach().cpu().numpy()
        thresholds = self.sdt.thresholds.detach().cpu().numpy()
        assert feature_selector.ndim == 3, "feature_selector should be 3D (n_estimators, num_inner_nodes, input_dim)"
        assert thresholds.ndim == 3, "thresholds should be 3D (n_estimators, num_inner_nodes, input_dim)"
        uninformative_paths = []
        path_internal_node_index = self.sdt.internal_node_index.detach().cpu().numpy() # (num_leaf_nodes, depth)
        internal_node_index = self.sdt.internal_node_index.detach().cpu().numpy()
        path_identifier = self.sdt.path_identifier.detach().cpu().numpy() 
        # (num_leaf_nodes, depth), left is 0, right is 1, if go right, feature is less than threshold
        for est in range(feature_selector.shape[0]):
            selected_features_idx = np.argmax(feature_selector[est], axis=1) # num_inner_nodes
            path_selected_features_idx = selected_features_idx[internal_node_index] # (num_leaf_nodes, depth)
            selected_thresholds = thresholds[est, np.arange(thresholds.shape[1]), selected_features_idx] # num_inner_nodes
            selected_thresholds_per_path = selected_thresholds[internal_node_index] # (num_leaf_nodes, depth)
            num_leaf_nodes, depth = path_selected_features_idx.shape
            for leaf_idx in range(num_leaf_nodes):
                path_features = path_selected_features_idx[leaf_idx] # (depth,)
                path_thresholds = selected_thresholds_per_path[leaf_idx] # (depth,)
                path_directions = path_identifier[leaf_idx] # (depth,)
                feature_dict = dict()
                for i in range(depth):
                    feature_i = path_features[i]
                    threshold_i = path_thresholds[i]
                    direction_i = path_directions[i]
                    if feature_i not in feature_dict:
                        feature_dict[feature_i] = [-np.inf, np.inf] # [min_threshold, max_threshold]
                    if direction_i == 0: # x < threshold
                        feature_dict[feature_i][1] = min(feature_dict[feature_i][1], threshold_i)
                    elif direction_i == 1: # x >= threshold
                        feature_dict[feature_i][0] = max(feature_dict[feature_i][0], threshold_i)
                    if feature_dict[feature_i][0] > feature_dict[feature_i][1]:
                        uninformative_paths.append((est, leaf_idx))
                        break
            return uninformative_paths

    def compute_leaf_occupancy(self, X):
        """Route samples to leaves and return occupancy-based structure metrics.

        Returns a dict (averaged over estimators) with:
          - leaf_utilization: fraction of leaves visited by >=1 sample
          - occupancy_entropy: normalized entropy of the leaf sample distribution
          - max_leaf_fraction: largest fraction of samples in a single leaf
        """

        X = torch.as_tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            path_probas = self.forward(X)['probas'][0]  # (batch, seq, n_estimators, num_leaf_nodes)
        if path_probas.ndim == 4:
            path_probas = path_probas.squeeze(1)  # (batch, n_estimators, num_leaf_nodes)
        num_leaf_nodes = path_probas.shape[-1]
        n_estimators = path_probas.shape[1]
        utils = []
        entropies = []
        max_fracs = []
        total_counts = np.zeros(num_leaf_nodes, dtype=float)
        for e in range(n_estimators):
            leaf_idx = path_probas[:, e, :].argmax(dim=-1).cpu().numpy()  # (batch,)
            counts = np.bincount(leaf_idx, minlength=num_leaf_nodes).astype(float)
            total_counts += counts
            n = counts.sum()
            if n <= 0:
                continue
            utils.append(float((counts > 0).sum()) / num_leaf_nodes)
            p = counts / n
            ent = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
            entropies.append(ent / np.log(num_leaf_nodes))
            max_fracs.append(float(p.max()))
        return {
            'leaf_utilization': float(np.mean(utils)) if utils else 0.0,
            'occupancy_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'max_leaf_fraction': float(np.mean(max_fracs)) if max_fracs else 0.0,
            'per_leaf_counts': total_counts,
        }

    def compute_split_feature_counts(self):
        """Count how many inner nodes select each feature (argmax of feature_selector).

        Returns np.ndarray of shape (input_dim,), averaged over estimators.
        """

        feature_selector = self.sdt.feature_selector.detach().cpu().numpy()  # (e, i, n)
        n_features = feature_selector.shape[-1]
        counts = np.zeros(n_features, dtype=float)
        for est in range(feature_selector.shape[0]):
            selected = feature_selector[est].argmax(axis=-1)  # (i,)
            counts += np.bincount(selected, minlength=n_features)
        return counts / feature_selector.shape[0]

    def compute_feature_importance(self, X, y, criterion='gini', normalize=True):
        """Compute per-feature importance via impurity decrease (MDI),
        matching sklearn's ``feature_importances_`` semantics.

        Routes the training data through the hardened tree (argmax over
        ``feature_selector``) and attributes each node's impurity decrease to
        the feature selected at that node.

        :param X: preprocessed training features (n_samples, n_features).
            Must match the tree's ``input_dim`` (i.e. the post-encoding /
            post-scaling features the tree was fitted on).
        :param y: label-encoded targets (n_samples,).
        :param criterion: 'gini' or 'entropy'.
        :param normalize: if True, rescale importances to sum to 1.
        :return: np.ndarray of shape (n_features,).
        """

        if criterion not in ('gini', 'entropy'):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")

        feature_selector = self.sdt.feature_selector.detach().cpu().numpy()  # (e, i, n)
        thresholds = self.sdt.thresholds.detach().cpu().numpy()              # (e, i, n)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]
        n_estimators, num_inner_nodes, n_features = feature_selector.shape
        depth = self.tree.depth

        def _impurity(labels):
            _, counts = np.unique(labels, return_counts=True)
            p = counts / counts.sum()
            if criterion == 'gini':
                return 1.0 - float(np.sum(p ** 2)) # p(1 - p)
            p = p[p > 0]
            return -float(np.sum(p * np.log(p)))

        total_importance = np.zeros(n_features, dtype=np.float64)

        for est in range(n_estimators):
            selected_feat = np.argmax(feature_selector[est], axis=1)                     # (i,)
            selected_thr = thresholds[est, np.arange(num_inner_nodes), selected_feat]    # (i,)

            est_importance = np.zeros(n_features, dtype=np.float64)

            # Route samples through the complete binary tree, level by level.
            # Node indices follow the heap layout: children of i are 2i+1, 2i+2.
            node_samples = {0: np.arange(n_samples)}
            for level in range(depth):
                start = 2 ** level - 1 # leftmost node index at this level
                for node_idx in range(start, start + 2 ** level):
                    samples = node_samples.get(node_idx)
                    if samples is None or len(samples) == 0:
                        continue
                    feat = selected_feat[node_idx]
                    thr = selected_thr[node_idx]
                    # path_identifier 0 = left branch = x < threshold
                    go_left = X[samples, feat] < thr
                    left_samples = samples[go_left]
                    right_samples = samples[~go_left]

                    if len(left_samples) > 0 and len(right_samples) > 0:
                        n_node = len(samples)
                        imp_parent = _impurity(y[samples])
                        imp_left = _impurity(y[left_samples])
                        imp_right = _impurity(y[right_samples])
                        decrease = imp_parent - (
                            (len(left_samples) / n_node) * imp_left
                            + (len(right_samples) / n_node) * imp_right
                        )
                        # sklearn-style node sample weighting
                        est_importance[feat] += (n_node / n_samples) * decrease

                    node_samples[2 * node_idx + 1] = left_samples
                    node_samples[2 * node_idx + 2] = right_samples

            total_importance += est_importance

        importance = total_importance / n_estimators
        if normalize:
            total = importance.sum()
            if total != 0:
                importance = importance / total
        return importance

    def forward(self, X, **kwargs):
        feature_selector = kwargs.get('feature_selector')
        thresholds = kwargs.get('thresholds')
        leaf_weights = kwargs.get('leaf_weights')
        temperature = kwargs.get('temperature', 1.0)
        output = self.sdt(X, feature_selector=feature_selector, thresholds=thresholds, leaf_weights=leaf_weights, temperature=temperature)
        logits = output['predictions'][0]
        penalty = output['penalties'][0]
        thresholds = output.get('other', (None,))[0]
        probas = output['probas']
        return {
            'logits': logits,
            'penalty': self.lambda_ * penalty,
            'other':thresholds,
            'probas': probas,
        }
