import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import copy
import torch
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
import pandas as pd
import category_encoders as ce
from scipy.special import softmax
from model.utils import load_model
from model.tree import SoftDecisionTreeNeural
from model.utils_torch import normalize_data, normalize_by_used_features_f

def extract_tree_model(model, config, X_train, y_train, device="cpu", inference_device="cpu", scale=True ):
    if "cuda" in inference_device and device == "cpu":
        raise ValueError("Cannot run inference on cuda when model is on cpu")
    try:
        max_features = config['prior']['num_features']
    except KeyError:
        max_features = 100
    eval_position = X_train.shape[0]
    n_classes = len(np.unique(y_train))
    if X_train.shape[1] > max_features:
        raise ValueError(f"Cannot run inference on data with more than {max_features} features")

    n_features = X_train.shape[1]
    if torch.is_tensor(X_train):
        xs = X_train.to(device)
    else:
        xs = torch.Tensor(X_train.astype(float)).to(device)
    if torch.is_tensor(y_train):
        ys = y_train.to(device)
    else:
        ys = torch.Tensor(y_train.astype(float)).to(device)

    if scale:
        eval_xs_ = normalize_data(xs, eval_position)
    else:
        eval_xs_ = torch.clip(xs, min=-100, max=100)

    eval_xs = eval_xs_

    x_all_torch = torch.concat([eval_xs, torch.zeros((X_train.shape[0], max_features - X_train.shape[1]), device=device)], axis=1)


    x_src = model.encoder(x_all_torch.unsqueeze(1))

    if model.y_encoder is not None:
        y_src = model.y_encoder(ys.unsqueeze(1).unsqueeze(-1))
        train_x = x_src + y_src
    else:
        train_x = x_src

    if hasattr(model, "transformer_encoder"):
        # tabpfn mlp model maker
        output = model.transformer_encoder(train_x)
    else:
        raise ValueError("Tree predictor type only supported with transformer encoder model maker")
    
    ret_list = []

    if model.decoder.weight_embedding_rank is not None:
        feature_selector, thresholds, leaf_weights = model.extract_low_rank_grad_tree_parameters(output, ys, single_eval_pos=eval_position)
    else:
        feature_selector, thresholds, leaf_weights = model.decoder(output, ys)[0], model.decoder(output, ys)[1], model.decoder(output, ys)[2]
    feature_selector = feature_selector.squeeze(0)
    thresholds = thresholds.squeeze(0)
    leaf_weights = leaf_weights.squeeze(0)

    feature_selector = feature_selector[:, :, :n_features]  # /  (n_features / max_features)
    thresholds = thresholds[:, :, :n_features].clone() # * (n_features / max_features) # For motherTabPFN, it seems no normalization
    # if normalized_by_used_features:
    #     thresholds = thresholds * (n_features / max_features)
    
    leaf_weights = leaf_weights[:, :, :n_classes ]
    ret_list = [feature_selector, thresholds, leaf_weights]

    if inference_device == "cpu":
        def detach(x):
            return x.detach().cpu().numpy()
    else:
        def detach(x):
            return x.detach()
    ret_list = [detach(tensor) for tensor in ret_list]
    return *ret_list,

def predict_with_tree_model(tree, train_mean, train_std, X_test, scale=True, inference_device="cpu", config=None, logit_as_output=False, 
                            draw_tree=False, dataset_info=None):

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    # For gradtree, extra preprocessing is needed.
    # ── Category encoding (only if grad_tree_process was on during fit) ──
    if 'LeaveOneOutEncoder' in config and 'OneHotEncoder' in config:
        encoder_loo = config['LeaveOneOutEncoder']
        encoder_ohe = config['OneHotEncoder']
        meidian_train = config.get('median')

        X_test = encoder_loo.transform(X_test)
        X_test = encoder_ohe.transform(X_test)
        X_test = X_test.fillna(meidian_train)

    # ── QuantileTransformer (applied independently) ──
    if 'QuantileTransformer' in config:
        normalizer = config['QuantileTransformer']
        X_test = normalizer.transform(X_test.values.astype(np.float64) if isinstance(X_test, pd.DataFrame) else X_test)

    # ── SelectKBest (only if >100 features after preprocessing) ──
    if 'selectK' in config:
        X_test = np.nan_to_num(X_test, nan=0.0)
        X_test = config['selectK'].transform(X_test)

    # Verify feature count matches tree input_dim after all preprocessing
    if 'LeaveOneOutEncoder' in config or 'QuantileTransformer' in config:
        if hasattr(tree, 'input_dim') and X_test.shape[1] != tree.input_dim:
            print(f"[predict_tree] WARNING: feature mismatch after preprocessing: "
                    f"got {X_test.shape[1]}, tree expects {tree.input_dim}")
            if X_test.shape[1] < tree.input_dim:
                pad = np.zeros((X_test.shape[0], tree.input_dim - X_test.shape[1]))
                X_test = np.concatenate([X_test, pad], axis=1)
            else:
                X_test = X_test[:, :tree.input_dim]
    if inference_device == "cpu":
        X_test = np.array(X_test, dtype=float)
        # FIXME replacing nan with 0 as in TabPFN
        X_test = np.nan_to_num(X_test, 0)
        if scale:
            X_test_scaled = (X_test - train_mean) / train_std
        else:
            X_test_scaled = X_test
        out = np.clip(X_test_scaled, a_min=-100, a_max=100)
        out = torch.from_numpy(out).float()
        tree._last_tree_input = X_test_scaled[:500]
        tree_out = tree(out)
        tree.uninformative_paths = tree_out.get('uninformative_paths', None)
        out = tree_out['logits'].detach().numpy()
        threshold = tree_out.get('other', None)
        if len(out.shape) == 3:
            out = out.squeeze()
        if out.ndim == 1:
            out = out[:, np.newaxis]  # restore class dim if squeeze removed it
        if np.isnan(out).any():
            print("NAN")
            import pdb
            pdb.set_trace()
        if logit_as_output:
            return out
        if draw_tree:
            split_number, dataset_name = dataset_info['split_number'], dataset_info['dataset_name']
            visualized_tree = tree.draw_tree(features=dataset_info.get('feature_names'), classes=dataset_info.get('class_names'), 
                            input_data=X_test_scaled, thresholds=threshold)  
                
            figure_path = os.path.join(dataset_info['root'], dataset_info['model_name'])
            os.makedirs(figure_path, exist_ok=True)
            if dataset_info['ensemble_number'] is not None:
                tree_path = os.path.join(figure_path, f"{dataset_info['model_name']}_{dataset_name}_{split_number}_component_{dataset_info['ensemble_number']}")
                visualized_tree.render(tree_path, view=False, cleanup=True)
            else:
                tree_path = os.path.join(figure_path, f"{dataset_info['model_name']}_{dataset_name}_{split_number}")
                visualized_tree.render(tree_path, view=False, cleanup=True)  
        # print(out.shape)
        return softmax(out / .8, axis=1)
    elif "cuda" in inference_device:
        mean = torch.Tensor(train_mean).to(inference_device)
        std = torch.Tensor(train_std).to(inference_device)
        # FIXME replacing nan with 0 as in TabPFN
        X_test = torch.Tensor(X_test.values).to(inference_device).nan_to_num(0)
        if scale:
            X_test_scaled = (X_test - mean) / std
        else:
            X_test_scaled = X_test
        out = torch.clamp(X_test_scaled, min=-100, max=100)
        tree_out = tree(out)
        # breakpoint()
        tree._last_tree_input = X_test_scaled[:500]  
        tree.uninformative_paths = tree_out.get('uninformative_paths', None)
        out = tree_out['logits']
        if len(out.shape) == 3:
            out = out.squeeze()
        if logit_as_output:
            return out.cpu().numpy()
        return torch.nn.functional.softmax(out / .8, dim=1).detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown inference_device: {inference_device}")


class MotherTree(ClassifierMixin, BaseEstimator):
    def __init__(self, path=None, device="cpu", scale=True, inference_device="cpu",
                 model=None, config=None, draw_tree=False, root=None,  grad_tree_process = False,
                   ohe=None, further_tune = False, **kwargs):
        """
        Parameters
        ----------
        path : str, optional
            Path to checkpoint file. If provided, the model will be loaded from this path.
        device : str, default="cpu"
            Device to use for training.
        scale : bool, default=True
            Whether to scale the input features, which is important for the model's performance.
        inference_device : str, default="cpu"
            Device to use for inference.  
        model : torch.nn.Module, optional
            If provided, the model will be used directly instead of loading from a checkpoint.(Never provided, old code)
        config : dict, optional
            If provided, the model configuration will be used directly instead of loading from a checkpoint. (Never provided, old code)
        draw_tree : bool, default=False
            Whether to draw the tree structure after fitting.
        root : str, optional
            Root directory for saving tree visualizations. If not provided, the current working directory will be used.
        grad_tree_process : bool, default=False
            Whether to apply gradient tree processing, which includes category encoding.
        model_name : str, optional
            Name of the model, used for saving tree visualizations. If not provided, "MotherTree" will be used.
        count_finetune : bool, default=False
            Whether to count the number of times the model is fine-tuned.(Important for the model's performance)
        """

        
        self.path = path
        self.device = device
        self.inference_device = inference_device
        self.scale = scale
        self.draw_tree = draw_tree
        self.grad_tree_process = grad_tree_process
        self.root = os.path.dirname(os.path.abspath(__file__)) if root is None else root
        if model is not None and self.path is not None:
            raise ValueError("Only one of path or model must be provided")
        if model is not None and config is None:
            raise ValueError("config must be provided if model is provided")
        self.model = model
        self.config = config

        self.model_name = 'MotherTree'
        self.ohe = ohe
        self.further_tune = further_tune
        if self.further_tune:
            default_tuning_params = self.default_tuning_parameters()
            if kwargs is not None:
                for key in default_tuning_params:
                    if key in kwargs:
                        setattr(self, key, kwargs[key])
                    else:
                        setattr(self, key, default_tuning_params[key])

    def fit(self, X, y, kwargs=None):
        self.X_train_ = X
        # Allow kwargs to override constructor params at fit time
        if kwargs is not None:
            for key in ['grad_tree_process', 'count_finetune', 'scale', 'finetune',
                        'finetune_epochs', 'finetune_lr', 'ohe', 'axis_aligned',
                        'normalized_by_used_features', 'normalized_bias', 'normalized_weight',
                        'poetree_normalized', 'gumbel_hard', 'split_max_features']:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) > 10:
            raise ValueError(f"Only 10 classes supported, found {len(le.classes_)}")
        if self.model is not None:
            model = self.model
            config = copy.deepcopy(self.config)
        else:
            model, config = load_model(self.path, device=self.device)
            config = copy.deepcopy(config)
            self.config = config
            self.model = model
        if "model_type" not in config:
            config['model_type'] = config.get("model_maker", 'tabpfn')
        if config['model_type'] not in ["mlp", "mothernet", 'la_mothernet', 'mothertabpfn', 'mothertabicl']:
            raise ValueError(f"Incompatible model_type: {config['model_type']}")
        model.to(self.device)
        n_classes = len(le.classes_)
        model.num_feature = X.shape[1]

        self.feature_names = kwargs.get('feature_names', None) if kwargs is not None else None
        self.class_names = kwargs.get('class_names', None) if kwargs is not None else None
        self.dataset_name = kwargs.get('dataset_name', None) if kwargs is not None else None
        self.split_number = kwargs.get('split_number', 1) if kwargs is not None else 1
        self.ensemble_number = kwargs.get('idx', None) if kwargs is not None else None
        self.n_features = X.shape[1]
        self.num_classes_ = n_classes
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        
        if self.class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes_)]
        
        self.dataset_info = {
            'dataset_name': self.dataset_name,
            'split_number': self.split_number,
            'ensemble_number': self.ensemble_number,
            'feature_names': self.feature_names,
            'class_names': le.classes_.tolist(),
            'root': self.root,
            'model_name': self.model_name,
        }

        if self.grad_tree_process:
            high_cardinality_indices = []
            num_columns = []
            self.cat_idx = kwargs.get('cat_idx', []) if kwargs is not None else []
            # X = smart_normalize(X, self.cat_idx)
            # print(self.cat_idx)
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            # Save original feature names for visualization
            if self.feature_names is not None and len(self.feature_names) == X.shape[1]:
                self.config['_orig_feature_names'] = list(self.feature_names)
            else:
                self.config['_orig_feature_names'] = [f"feat_{i}" for i in range(X.shape[1])]
            for column, column_index in enumerate(X.columns):
                if column_index in self.cat_idx:
                    high_cardinality_indices.append(column)
                else:
                    num_columns.append(column)
            
            self.encoder_loo = ce.LeaveOneOutEncoder(cols=high_cardinality_indices)
            self.encoder_loo.fit(X, y)  
            X = self.encoder_loo.transform(X)

            # self.encoder_ohe = ce.OneHotEncoder(cols=low_cardinality_indices, handle_unknown='value')
            # self.encoder_ohe.fit(X)
            # X = self.encoder_ohe.transform(X)
            # Map OHE feature names back to original feature names
            # raw_names = list(self.encoder_ohe.get_feature_names_out())
            # orig = self.config['_orig_feature_names']
            mapped_names = []
            # for name in raw_names:
            #     parts = str(name).split('_', 1)
            #     idx = int(parts[0])
            #     orig_name = orig[idx]
            #     if len(parts) > 1:
            #         mapped_names.append(f"{orig_name}_{parts[1]}")
            #     else:
            #         mapped_names.append(orig_name)
            self.ohe_feature_names_ = np.array(mapped_names)

            self.median_train = X.median(axis=0)
            X = X.fillna(self.median_train)

            self.cat_columns_preprocessed = []
            for column, column_index in enumerate(X.columns):
                if column not in num_columns:
                    self.cat_columns_preprocessed.append(column_index)

            _cat_config = {
                'LeaveOneOutEncoder': self.encoder_loo,
                # 'OneHotEncoder': self.encoder_ohe,
                'median': self.median_train,
            }
            self.config.update(_cat_config)

        # ── QuantileTransformer (applied regardless of grad_tree_process) ──
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.median_train = X.median(axis=0)
        X = X.fillna(self.median_train)

        quantile_noise = 1e-4
        quantile_train = np.copy(X.values).astype(np.float64)
        np.random.seed(42)
        stds = np.std(quantile_train, axis=0, keepdims=True)
        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
        quantile_train += noise_std * np.random.randn(*quantile_train.shape)

        # Handle all-NaN columns
        all_nan_mask = np.isnan(quantile_train).all(axis=0)
        if all_nan_mask.any():
            quantile_train[:, all_nan_mask] = 0
            if isinstance(X, pd.DataFrame):
                X.loc[:, all_nan_mask] = 0
            else:
                X[:, all_nan_mask] = 0

        self.normalizer = sklearn.preprocessing.QuantileTransformer(
            n_quantiles=min(quantile_train.shape[0], 1000),
            output_distribution='normal',
        )
        self.normalizer.fit(quantile_train)
        X = self.normalizer.transform(X.values if isinstance(X, pd.DataFrame) else X)
        X = np.nan_to_num(X, nan=0.0)
        self.config['QuantileTransformer'] = self.normalizer
        self.config['median'] = self.median_train

        # ── SelectKBest (still only under grad_tree_process, depends on OHE feature names) ──
        # should be changed regardless of grad_tree_process, but only if >100 features after preprocessing


        if X.shape[1] > 100:
            selector = SelectKBest(score_func=f_classif, k=100)
            X = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            self.feature_names = [str(n) for n in self.ohe_feature_names_[selected_indices]]
            self.config['selectK'] = selector
            # else:
            #     self.feature_names = [str(n) for n in self.ohe_feature_names_]
            #     self.config.pop('selectK', None)
        # Escape special chars for DOT/graphviz and update dataset_info
        self.feature_names = [n.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;') for n in self.feature_names]
        self.dataset_info['feature_names'] = self.feature_names


        feature_selector, thresholds, leaf_weights = extract_tree_model(model, config, X, y, device=self.device,
                                inference_device=self.inference_device, scale=self.scale,)

        self.parameters_ = [feature_selector, thresholds, leaf_weights]
        self.tree =  SoftDecisionTreeNeural(input_dim=X.shape[1], output_dim=n_classes, 
                                            max_depth=config['mothernet'].get('depth', 5))

            # initialize tree weights with extracted weights from mothernet
        self.tree._init_node_weight(self.parameters_)
        self.predictor_type = 'Tree'



        if self.further_tune:
            pass
        else:
            self.count_fine_tune(X, y)


        self.classes_ = le.classes_
        self.mean_ = np.nan_to_num(np.nanmean(X, axis=0), 0)
        self.std_ = np.nanstd(X, axis=0, ddof=1) + .000001
        self.std_[np.isnan(self.std_)] = 1
        self.uninformative_paths =  self.tree.examine_uninformative_paths() if self.predictor_type == 'Tree' else None
        if self.predictor_type == 'Tree':
            # X here is post-preprocessing (encoding + quantile transform);
            # y is already label-encoded. Kept for feature importance.
            self.X_train_processed_ = X
            self.y_train_encoded_ = y
            self.feature_importances_ = self.tree.compute_feature_importance(X, y)
        leaf_utilization = None
        occupancy_entropy = None
        max_leaf_fraction = None
        per_leaf_occupancy = None
        split_feature_counts = None
        if self.predictor_type == 'Tree' and getattr(self, 'tree_type', None) == 'grad_tree':
            try:
                occupancy = self.tree.compute_leaf_occupancy(X)
                leaf_utilization = occupancy['leaf_utilization']
                occupancy_entropy = occupancy['occupancy_entropy']
                max_leaf_fraction = occupancy['max_leaf_fraction']
                per_leaf_occupancy = occupancy['per_leaf_counts']
                split_feature_counts = self.tree.compute_split_feature_counts()
            except Exception:
                leaf_utilization = None
                occupancy_entropy = None
                max_leaf_fraction = None
                per_leaf_occupancy = None
                split_feature_counts = None
        self.tree_metrics_ = {
            'num_uninformative_paths': len(self.uninformative_paths) if self.uninformative_paths is not None else None,
            'leaf_utilization': leaf_utilization,
            'occupancy_entropy': occupancy_entropy,
            'max_leaf_fraction': max_leaf_fraction,
            'per_leaf_occupancy': per_leaf_occupancy,
            'split_feature_counts': split_feature_counts,
            'feature_importance': getattr(self, 'feature_importances_', None),
            'feature_names': getattr(self, 'feature_names', None),
        }
        return self

    
    def count_fine_tune(self, X, y, alpha=1.0):
        if self.predictor_type != "Tree":
            raise ValueError(
                "Fine-tuning only supported for tree predictor type"
            )

        X = torch.nan_to_num(torch.as_tensor(X, dtype=torch.float32),nan=0.0).to(self.device)
        y_tensor = torch.as_tensor(y,dtype=torch.long,device=self.device)

        self.tree.to(self.device)
        self.tree.eval()

        with torch.no_grad():
            outputs = self.tree(X)
            leaf_indices = (outputs["probas"][0].squeeze(0).argmax(dim=-1).cpu().numpy())

        # output_dist = outputs['probas'][0].argmax(-1).squeeze()
        n_estimators, n_leaf_nodes, n_classes = self.tree.sdt.leaf_nodes.shape

        # (n_estimators, n_leaf_nodes, n_classes)
        leaf_counts = np.zeros(
            (n_estimators, n_leaf_nodes, n_classes),
            dtype=np.float64
        )

        y_np = y_tensor.cpu().numpy()

        for est_idx in range(n_estimators):
            est_leaf_ids = leaf_indices[:, est_idx]
            for sample_idx, leaf_id in enumerate(est_leaf_ids):
                cls = y_np[sample_idx]
                leaf_counts[est_idx, leaf_id, cls] += 1
        leaf_totals = leaf_counts.sum(axis=-1)  # (E, L)

        visited_mask = leaf_totals > 0
        leaf_probs = (
            leaf_counts + alpha
        ) / (
            leaf_totals[..., None]
            + alpha * n_classes
        )
        leaf_logits = np.log(leaf_probs)

        leaf_logits = torch.tensor(
            leaf_logits,
            dtype=self.tree.sdt.leaf_nodes.dtype,
            device=self.tree.sdt.leaf_nodes.device
        )

        visited_mask = torch.tensor(
            visited_mask,
            dtype=torch.bool,
            device=self.tree.sdt.leaf_nodes.device
        )

        new_leaf_nodes = self.tree.sdt.leaf_nodes.data.clone()
        new_leaf_nodes[visited_mask] = leaf_logits[visited_mask]
        self.tree.sdt.leaf_nodes.data.copy_(new_leaf_nodes)

    @staticmethod
    def default_tuning_parameters():
        feature_selector_lr = [0.001,0.005,0.010,0.050]
        threshold_lr = [0.010,0.050,0.100,0.250]
        leaf_node_lr = [0.001,0.005,0.010,0.100]
        epochs = 1000
        patience = 50
        return {
            'feature_selector_lr': feature_selector_lr,
            'threshold_lr': threshold_lr,
            'leaf_node_lr': leaf_node_lr,
            'epochs': epochs,
            'patience': patience
        }

    def predict_proba(self, X, logit_as_output=False):

        proba = predict_with_tree_model(self.tree, self.mean_, self.std_, X, scale=self.scale, 
                                        inference_device=self.inference_device, config=self.config, logit_as_output=logit_as_output,
                                        draw_tree=self.draw_tree, dataset_info=self.dataset_info)
    
        occ = self.tree.compute_leaf_occupancy(self.tree._last_tree_input) if getattr(self, 'tree', None) is not None else None 
        if occ is not None and self.tree_metrics_ is not None:
            self.tree_metrics_['leaf_utilization_test'] = occ['leaf_utilization']
            self.tree_metrics_['occupancy_entropy_test'] = occ['occupancy_entropy']
            self.tree_metrics_['max_leaf_fraction_test'] = occ['max_leaf_fraction']
        return proba

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def compute_feature_importance(self, criterion='gini', normalize=True):
        """Compute per-feature importance (MDI) on the fitted tree.

        Uses the preprocessed training features stored during ``fit``.
        See ``SoftDecisionTreeNeural.compute_feature_importance``.
        """
        if self.predictor_type != 'Tree':
            raise ValueError("Feature importance is only available for tree predictors.")
        if not hasattr(self, 'X_train_processed_'):
            raise ValueError("Model must be fitted before computing feature importance.")
        return self.tree.compute_feature_importance(
            self.X_train_processed_, self.y_train_encoded_,
            criterion=criterion, normalize=normalize,
        )


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    path = '/Users/wangzi/Downloads/20260722_201828_mothernet_Tree_grad_tree_d5_b64_lr3e-05_step_155000.cpkt'
    model = MotherTree(path=path, device="cpu", scale=True, inference_device="cpu", draw_tree=False,)
    # model = DecisionTreeClassifier(max_depth=5, random_state=42)  # For testing purposes, replace with actual model if needed
    # print(model)

    # Load sample data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model.fit(X_train, y_train)
    # Predict probabilities
    prediction_probabilities = model.predict_proba(X_test)
    # Predict labels
    predictions = model.predict(X_test)
    print("Accuracy", accuracy_score(y_test, predictions))
    print("Prediction probabilities shape:", prediction_probabilities)