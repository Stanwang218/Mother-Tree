from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
from model.tree import SoftDecisionTreeNeural
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import copy
import pandas as pd
import category_encoders as ce
import sklearn
from copy import deepcopy
from typing import Optional, List, Dict, Any, Tuple


class FocalLoss(nn.Module):
    """Multiclass focal loss computed on raw logits.

    ``loss = -alpha_t * (1 - p_t)^gamma * log(p_t)``, where ``p_t`` is the
    softmax probability of the target class. Optional per-class ``weight``
    follows the same convention as ``nn.CrossEntropyLoss``.
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        loss = ((1.0 - p_t) ** self.gamma) * ce
        if self.weight is not None:
            loss = loss * self.weight[targets]
        if self.reduction == "mean":
            # Match nn.CrossEntropyLoss(weight=...) semantics: with a class
            # weight, 'mean' divides by the sum of the weights of the samples
            # in the batch (so gamma=0 collapses exactly onto weighted CE).
            if self.weight is not None:
                return loss.sum() / self.weight[targets].sum()
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)

_GRID_WORKER_STATE = {}


def _grid_search_init(trainer, train_dataset, val_dataset, class_weight_tensor):
    """Runs once per worker process: stash the task-invariant training state."""
    # One run per process: let the pool own the parallelism instead of having
    # every worker fight the others over all CPU cores.
    torch.set_num_threads(1)
    set_seed(42)
    _GRID_WORKER_STATE.update(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_weight_tensor=class_weight_tensor,
    )


def _grid_search_run(candidate):
    """Train ONE candidate in a worker process and return its result dict."""
    lr_index, lr_values, lr_leaf, use_focal = candidate
    trainer = _GRID_WORKER_STATE['trainer']

    model = trainer._make_model()
    model.load_state_dict(trainer._grid_init_state)

    best_val, best_auc, best_state, tr_loss, va_loss = trainer._train_loop(
        model,
        _GRID_WORKER_STATE['train_dataset'],
        _GRID_WORKER_STATE['val_dataset'],
        _GRID_WORKER_STATE['class_weight_tensor'],
        lr_index, lr_values, lr_leaf, focal_loss=use_focal,
    )
    return {
        'learning_rate_index': lr_index,
        'learning_rate_values': lr_values,
        'learning_rate_leaf': lr_leaf,
        'focal_loss': use_focal,
        'best_val_loss': best_val,
        'best_val_auc': best_auc,
        'best_state': best_state,
        'train_loss': tr_loss,
        'val_loss': va_loss,
    }

class GradTreeClassifier(ClassifierMixin, BaseEstimator):
    """
    Gradient-based soft decision tree classifier.
    Path: path for saving model checkpoints and visualizations. If None, defaults to "gradtree_res" in the current directory.
    """
    def __init__(self, n_estimators=1,  depth=5, batch_size = 64, 
                 learning_rate_index = 0.01, learning_rate_values = 0.01, learning_rate_leaf = 0.01,
                 epochs = 1000, patience = 50, optimizer_name = 'Adam', focal_loss = False, focal_loss_gamma = 2.0,
                 grid_search = False, 
                 grid_search_lr_index:List[float] = [0.001, 0.005, 0.010, 0.050],
                 grid_search_lr_values:List[float] = [0.010, 0.050, 0.100, 0.250],
                 grid_search_lr_leaf:List[float] = [0.001, 0.005, 0.010, 0.100],
                 grid_search_focal_loss:List[bool] = [False],
                 grid_search_workers = 4,
                 path=None, device="cpu", scale=True, 
                 inference_device="cpu",
                 draw_tree=False, overwrite=False, 
                 init_from_mothernet=False, mothernet_path=None, mothernet_kwargs=None, collect_tree_metrics=False):
        set_seed(42)
        self.learning_rate_index = learning_rate_index
        self.learning_rate_values = learning_rate_values
        self.learning_rate_leaf = learning_rate_leaf
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.grid_search = grid_search
        self.optimizer_name = optimizer_name
        self.focal_loss = focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        if self.grid_search:
            self.set_grid_search_parameters(grid_search_lr_index=grid_search_lr_index, 
                                            grid_search_lr_values=grid_search_lr_values, 
                                            grid_search_lr_leaf=grid_search_lr_leaf, 
                                            grid_search_focal_loss=grid_search_focal_loss, 
                                            grid_search_workers=grid_search_workers)
        self.path = path # path for ckpt, need to specify dataset, index, and ensemble number if applicable, it seems need to be deprecated
        self.device = device
        self.inference_device = inference_device
        self.scale = scale
        self.n_estimators = n_estimators
        self.overwrite = overwrite
        self.init_from_mothernet = init_from_mothernet
        self.mothernet_path = mothernet_path
        self.mothernet_kwargs = mothernet_kwargs or {}
        self.depth = depth
        self.model_fn = SoftDecisionTreeNeural

        self.path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradtree_res")
        self.visualization_root = os.path.join(self.path, "gradtree_visualizations")
        self.save_ckpt_path = os.path.join(self.path, "gradtree_ckpt")
        if os.makedirs(self.visualization_root, exist_ok=True):
            print(f"Created directory {self.visualization_root} for saving GradTree visualizations.")
        if os.makedirs(self.save_ckpt_path, exist_ok=True):
            print(f"Created directory {self.save_ckpt_path} for saving GradTree checkpoints.")
     
        self.ensemble_number = None
        self.split_number = 1
        self.draw_tree = draw_tree
        self.collect_tree_metrics = collect_tree_metrics

    def set_grid_search_parameters(self, **kwargs):

        # Grid search over the full Cartesian product of three log-spaced grids.
        self.grid_search = kwargs.get('grid_search', True)
        self.grid_search_lr_index = kwargs.get('grid_search_lr_index') or [0.001, 0.005, 0.010, 0.050]
        self.grid_search_lr_values = kwargs.get('grid_search_lr_values') or [0.010, 0.050, 0.100, 0.250]
        self.grid_search_lr_leaf = kwargs.get('grid_search_lr_leaf') or [0.001, 0.005, 0.010, 0.100]
        self.grid_search_workers = int(kwargs.get('grid_search_workers', 1) or 1)
        # Loss selection: plain cross-entropy (False) vs focal loss (True).
        self.focal_loss = kwargs.get('focal_loss', False)
        self.focal_loss_gamma = kwargs.get('focal_loss_gamma', 2.0)
        self.grid_search_focal_loss = kwargs.get('grid_search_focal_loss')
        if self.grid_search_focal_loss is None:
            self.grid_search_focal_loss = [False, True]
        elif not isinstance(self.grid_search_focal_loss, (list, tuple)):
            self.grid_search_focal_loss = [self.grid_search_focal_loss]


    def load_model(self, path):
        loaded = torch.load(path, map_location=self.device)
        if isinstance(loaded, (list, tuple)):
            ckpt = loaded[0]
            model_info = loaded[1]
            info = loaded[2] if len(loaded) > 2 else None
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(loaded)}")
        model = SoftDecisionTreeNeural(
            input_dim=model_info[0],
            output_dim=model_info[1],
            tree_type = 'grad_tree',
            max_depth=self.depth,
            n_estimators=self.n_estimators,
        )
        model.load_state_dict(ckpt)
        if info is not None:
            self._restore_checkpoint_info(info)
        return model

    def _restore_checkpoint_info(self, info):
        """Restore hyperparameters / grid-search results stored in a checkpoint."""
        for key in ('learning_rate_index', 'learning_rate_values', 'learning_rate_leaf', 'focal_loss', 'focal_loss_gamma', 'best_val_loss', 'best_val_auc'):
            if key in info:
                setattr(self, key, info[key])
        if 'grid_search' in info:
            self.grid_search = info['grid_search']
        if 'grid_search_results' in info:
            self.grid_search_results_ = info['grid_search_results']

    @staticmethod
    def _split_train_val(X, y, test_size=0.1, random_state=42):
        """Split off a validation set, stratified whenever the classes allow it."""
        try:
            return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        except ValueError:
            # A class may have a single sample, which stratified splitting cannot
            # handle -> fall back to a plain random split.
            return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def fit(self, X, y, kwargs=None):
        if self.init_from_mothernet:
            return self._fit_with_mothernet_init(X, y, kwargs)

        kwargs = kwargs or {}
        high_cardinality_indices = []
        num_columns = []
        self.cat_idx = kwargs.get('cat_idx', [])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        for column_name in X.columns:
            if column_name in self.cat_idx:
                high_cardinality_indices.append(column_name)
            else:
                num_columns.append(column_name)

        # Encode the labels first: the split below is stratified on them and the
        # leave-one-out encoder needs numeric targets.
        le = LabelEncoder()
        y = le.fit_transform(y)
        if len(le.classes_) > 10:
            raise ValueError(f"Only 10 classes supported, found {len(le.classes_)}")
        self.classes_ = le.classes_
        self.num_classes_ = len(self.classes_)
        self.class_names = self.classes_.tolist()  # sync with actual fitted classes

        # Split BEFORE fitting any pre-processing. Otherwise the LOO encoding,
        # the median imputation and the QuantileTransformer are all fitted on
        # rows that later serve as the early-stopping / model-selection set,
        # which leaks their labels into training and inflates `best_val_auc_`.
        X_train, X_val, y_train, y_val = self._split_train_val(X, y)

        # Categorical encoding: fitted on the training split only.
        self.encoder_loo = ce.LeaveOneOutEncoder(cols=high_cardinality_indices)
        self.encoder_loo.fit(X_train, y_train)
        # NOTE: passing y to transform() applies the leave-one-out mean (a row's
        # own target is excluded); a plain transform(X) falls back to the full
        # category mean and would leak each training row's label into its own
        # features.
        X_train = self.encoder_loo.transform(X_train, y_train)
        X_val = self.encoder_loo.transform(X_val)

        self.median_train = X_train.median(axis=0)
        X_train = X_train.fillna(self.median_train)
        X_val = X_val.fillna(self.median_train)

        self.cat_columns_preprocessed = []
        for column_name in X_train.columns:
            if column_name not in num_columns:
                self.cat_columns_preprocessed.append(column_name)

        # Quantile normalisation: fitted on the training split only. The noisy
        # copy is used for fitting, the clean data is what gets transformed.
        quantile_noise = 1e-4
        quantile_train = np.copy(X_train.values).astype(np.float64)
        rng = np.random.RandomState(42)
        stds = np.std(quantile_train, axis=0, keepdims=True)
        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
        quantile_train += noise_std * rng.randn(*quantile_train.shape)

        self.normalizer = sklearn.preprocessing.QuantileTransformer(
                                                    n_quantiles=min(quantile_train.shape[0], 1000),
                                                    output_distribution='normal',)

        self.normalizer.fit(quantile_train)
        X_train = self.normalizer.transform(X_train.values.astype(np.float64))
        X_val = self.normalizer.transform(X_val.values.astype(np.float64))

        # `X_train_` keeps its previous meaning: every row passed to fit
        # (train + val), pre-processed, and aligned with `y_train_`.
        self.X_train_ = np.vstack([X_train, X_val])
        self.y_train_ = np.concatenate([y_train, y_val])
        self.X_val_ = X_val
        self.y_val_ = y_val


        self.n_features = X_train.shape[1]
        self.feature_names = kwargs.get('feature_names', [f"feature_{i}" for i in range(self.n_features)])

        # Escape special chars for DOT/graphviz
        self.feature_names = [n.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;') for n in self.feature_names]
        self.dataset_name = kwargs.get('dataset_name', None) if kwargs is not None else None
        self.split_number = kwargs.get('split_number', 1) if kwargs is not None else 1
        self.ensemble_number = kwargs.get('idx', None) if kwargs is not None else None

        self.model = self.model_fn(input_dim=self.n_features, 
                                    output_dim=self.num_classes_,
                                    max_depth=self.depth,
                                    n_estimators=self.n_estimators,
                                    ).to(self.device)
        self.train(X_train, y_train, X_val, y_val)

        if self.collect_tree_metrics:
            self._collect_tree_metrics(self.X_train_, self.y_train_)
        return self

    def _collect_tree_metrics(self, X_processed, y_encoded):
        """Aggregate all tree-structure diagnostics into a single dict.

        New tree metrics (leaf occupancy, leaf purity, ...) should be added
        here — and only here. The evaluation pipeline only reads
        ``tree_metrics_``.
        """
        self.uninformative_paths = self.model.examine_uninformative_paths()
        self.feature_importances_ = self.model.compute_feature_importance(X_processed, y_encoded)
        occupancy = self.model.compute_leaf_occupancy(X_processed)
        self.tree_metrics_ = {
            'num_uninformative_paths': len(self.uninformative_paths) if self.uninformative_paths is not None else None,
            'leaf_utilization': occupancy['leaf_utilization'],
            'occupancy_entropy': occupancy['occupancy_entropy'],
            'max_leaf_fraction': occupancy['max_leaf_fraction'],
            'per_leaf_occupancy': occupancy['per_leaf_counts'],
            'split_feature_counts': self.model.compute_split_feature_counts(),
            'feature_importance': self.feature_importances_,
            'feature_names': self.feature_names,
        }
        return self.tree_metrics_

    
    def _fit_with_mothernet_init(self, X, y, kwargs=None):
        """Initialize GradTree from a MotherNet (grad_tree head) few-shot
        prediction, then finetune (or fully train) it on the current dataset.

        MotherNet's `grad_tree` head already outputs GradTree-compatible
        parameters (``feature_selector``, ``thresholds``, ``leaf_weights``),
        so no oblique→axis-aligned conversion is needed.  We reuse MotherNet's
        preprocessing objects so the initialized parameters and the training
        data live in the same feature space.
        """
        from prediction.mothertree import MotherTree

        mothernet_kwargs = dict(self.mothernet_kwargs)
        mothernet_kwargs.setdefault('grad_tree_process', True)

        mc = MotherTree(
            path=self.mothernet_path,
            device=self.device,
            **mothernet_kwargs,
        )
        # Merge the pipeline's kwargs (feature_names, cat_idx, dataset_name, ...)
        # with the MotherNet-specific overrides so MotherNet sees the real feature
        # names instead of falling back to feature_0..feature_n.
        mc_fit_kwargs = dict(kwargs) if kwargs is not None else {}
        mc_fit_kwargs.update(mothernet_kwargs)
        mc.fit(X, y, kwargs=mc_fit_kwargs)

        # Reuse MotherNet's preprocessing so train/predict feature spaces match.

        self.encoder_loo = getattr(mc, 'encoder_loo', None)
        self.encoder_ohe = getattr(mc, 'encoder_ohe', None)
        self.median_train = mc.median_train
        self.normalizer = mc.normalizer
        self.selector = mc.config.get('selectK', None)
        self.ohe_feature_names_ = getattr(mc, 'ohe_feature_names_', None)

        X_processed = mc.X_train_processed_
        y_encoded = mc.y_train_encoded_

        self.classes_ = mc.classes_
        self.num_classes_ = mc.num_classes_
        self.class_names = mc.class_names
        self.feature_names = mc.feature_names
        self.n_features = X_processed.shape[1]
        self.X_train_ = X_processed

        self.dataset_name = kwargs.get('dataset_name', None) if kwargs is not None else None
        self.split_number = kwargs.get('split_number', 1) if kwargs is not None else 1
        self.ensemble_number = kwargs.get('idx', None) if kwargs is not None else None

        # Align params to (n_estimators, ...) and initialize the GradTree.
        feature_selector, thresholds, leaf_weights = mc.parameters_
        feature_selector = np.asarray(feature_selector, dtype=np.float32)
        thresholds = np.asarray(thresholds, dtype=np.float32)
        leaf_weights = np.asarray(leaf_weights, dtype=np.float32)
        if feature_selector.ndim == 2:
            feature_selector = feature_selector[None]
        if thresholds.ndim == 2:
            thresholds = thresholds[None]
        if leaf_weights.ndim == 2:
            leaf_weights = leaf_weights[None]
        if feature_selector.shape[0] == 1 and self.n_estimators > 1:
            feature_selector = np.repeat(feature_selector, self.n_estimators, axis=0)
            thresholds = np.repeat(thresholds, self.n_estimators, axis=0)
            leaf_weights = np.repeat(leaf_weights, self.n_estimators, axis=0)

        self.model = self.model_fn(
            input_dim=self.n_features,
            output_dim=self.num_classes_,
            max_depth=self.depth,
            n_estimators=self.n_estimators,
        ).to(self.device)
        self.model._init_node_weight([feature_selector, thresholds, leaf_weights])

        # Split before training. `X_train_`/`y_train_` stay the full (aligned)
        # set so the tree metrics cover the same rows as before.
        self.y_train_ = y_encoded
        X_train, X_val, y_train, y_val = self._split_train_val(X_processed, y_encoded)
        self.X_val_ = X_val
        self.y_val_ = y_val
        self.train(X_train, y_train, X_val, y_val)

        if self.collect_tree_metrics:
            self._collect_tree_metrics(self.X_train_, self.y_train_)
        return self


    def predict_with_gradtree_model(self, X_test, inference_device="cpu"):
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        X_test = self.encoder_loo.transform(X_test).fillna(self.median_train)
        X_test = self.normalizer.transform(X_test.values.astype(np.float64))
        X_test = np.nan_to_num(X_test, nan=0.0)
        if getattr(self, 'selector', None) is not None:
            X_test = self.selector.transform(X_test)

        if inference_device == "cpu":
            model = self.model.cpu()
            X_test_scaled = X_test
            out = np.clip(X_test_scaled, a_min=-100, a_max=100)
            out = torch.from_numpy(out).float()
            model_out = model(out)
            thresholds = model_out['other']
            out = model_out['logits'].detach().numpy()
            if self.draw_tree:
                tree = model.draw_tree(self.feature_names, self.class_names, thresholds=thresholds, 
                                       input_data=X_test_scaled, probas=None)
                if self.ensemble_number is not None:
                    tree_path = os.path.join(self.path, f"gradtree_tree_{self.dataset_name}_{self.split_number}_component_{self.ensemble_number}")
                    tree.render(tree_path, view=False, cleanup=True)
                else:
                    tree_path = os.path.join(self.path, f"gradtree_tree_{self.dataset_name}_{self.split_number}")
                    tree.render(tree_path, view=False, cleanup=True)
                
            if len(out.shape) == 3:
                out = out.squeeze()
            if np.isnan(out).any():
                raise ValueError("NaN values found in model output. Check the input data and model parameters.")
            from scipy.special import softmax
            return softmax(out / .8, axis=1)
        elif "cuda" in inference_device:
            # FIXME replacing nan with 0 as in TabPFN
            X_test = torch.Tensor(X_test).to(inference_device).nan_to_num(0)
            model = self.model.to(inference_device)
            X_test_scaled = X_test
            out = torch.clamp(X_test_scaled, min=-100, max=100)
            out = model(out)
            if len(out.shape) == 3:
                out = out.squeeze()
            return torch.nn.functional.softmax(out['logits'] / .8, dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unknown inference_device: {inference_device}")


    def predict_proba(self, X):
        X = np.nan_to_num(X, nan=0.0)
        return self.predict_with_gradtree_model(X, inference_device=self.inference_device)
                
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
    
    def _make_model(self):
        return self.model_fn(
            input_dim=self.n_features,
            output_dim=self.num_classes_,
            max_depth=self.depth,
            n_estimators=self.n_estimators,
        ).to(self.device)

    def _run_grid_search(self, candidates, init_state, train_dataset, val_dataset, class_weight_tensor):
        """Train every candidate and return the list of per-candidate results.

        When ``grid_search_workers > 1`` the candidates are spread over worker
        processes. The heavy ``nn.Module`` is replaced by the shared
        ``init_state``, so workers only receive read-only fitting state once
        (via the pool initializer) and then hand back the results.
        """
        import concurrent.futures as cf
        import multiprocessing as mp

        workers = int(getattr(self, 'grid_search_workers', 1) or 1)
        if workers <= 1 or len(candidates) <= 1:
            return self._run_grid_search_sequential(
                candidates, init_state, train_dataset, val_dataset, class_weight_tensor)

        if str(self.device) != 'cpu':
            print(f"[grid search] WARNING: {workers} worker processes all use device={self.device!r}")

        worker_trainer = copy.copy(self)
        worker_trainer.model = None
        worker_trainer._grid_init_state = init_state

        results = [None] * len(candidates)
        with cf.ProcessPoolExecutor(
            max_workers=min(workers, len(candidates)),
            mp_context=mp.get_context('spawn'),
            initializer=_grid_search_init,
            initargs=(worker_trainer, train_dataset, val_dataset, class_weight_tensor),
        ) as pool:
            futures = {pool.submit(_grid_search_run, c): i for i, c in enumerate(candidates)}
            for future in cf.as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def _run_grid_search_sequential(self, candidates, init_state, train_dataset, val_dataset, class_weight_tensor):
        runs = []
        for lr_index, lr_values, lr_leaf, use_focal in candidates:
            model = self._make_model()
            model.load_state_dict(init_state)
            best_val, best_auc, best_state, tr_loss, va_loss = self._train_loop(
                model, train_dataset, val_dataset, class_weight_tensor,
                lr_index, lr_values, lr_leaf, focal_loss=use_focal,
            )
            runs.append({
                'learning_rate_index': lr_index,
                'learning_rate_values': lr_values,
                'learning_rate_leaf': lr_leaf,
                'focal_loss': use_focal,
                'best_val_loss': best_val,
                'best_val_auc': best_auc,
                'best_state': best_state,
                'train_loss': tr_loss,
                'val_loss': va_loss,
            })
        return runs

    def _eval_auc(self, state_dict, val_dataloader):
        """Compute ROC-AUC of a model state on the validation loader.

        Used as a loss-scale-invariant selection criterion for grid search:
        cross-entropy and focal loss have different magnitudes (focal <= CE),
        so comparing their validation *losses* would systematically favor
        focal. Validation AUC is comparable across both loss types.
        """
        from sklearn.metrics import roc_auc_score
        model = self._make_model()
        model.load_state_dict(state_dict)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.nan_to_num(0).to(self.device)
                batch_y = batch_y.to(self.device)
                logits = model(batch_X)['logits'].reshape(-1, self.num_classes_)
                preds.append(torch.softmax(logits, dim=1).cpu())
                targets.append(batch_y.cpu())
        if not preds:
            return float('nan')
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        if len(np.unique(targets)) < 2:
            return float('nan')
        try:
            if self.num_classes_ == 2:
                return float(roc_auc_score(targets, preds[:, 1]))
            return float(roc_auc_score(targets, preds, multi_class='ovo'))
        except ValueError:
            return float('nan')

    def _train_loop(self, model, train_dataset, val_dataset, class_weight_tensor,
                    lr_index, lr_values, lr_leaf, focal_loss=False):
        """Run one full training run with the given learning rates.

        No progress bar: during a grid search every candidate runs in its own
        process, so per-run bars would interleave and flood the terminal.

        Returns ``(best_val_loss, best_val_auc, best_state, training_loss_list, val_loss_list)``.
        """
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        if self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD([{"params": model.sdt.feature_selector, "lr": lr_index},
                                        {"params": model.sdt.thresholds, "lr": lr_values},
                                        {"params": model.sdt.leaf_nodes, "lr": lr_leaf},])
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam([{"params": model.sdt.feature_selector, "lr": lr_index},
                                          {"params": model.sdt.thresholds, "lr": lr_values},
                                          {"params": model.sdt.leaf_nodes, "lr": lr_leaf},])
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")



        if focal_loss:
            loss_fn = FocalLoss(gamma=self.focal_loss_gamma, weight=class_weight_tensor)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)

        best_val_loss = float("inf")
        counter = 0
        training_loss_list, val_loss_list = [], []
        global_step = 0
        for _ in range(self.epochs):
            model.train()
            total_loss = 0

            for batch_X, batch_y in train_dataloader:
                batch_X = batch_X.nan_to_num(0).to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)['logits'].reshape(-1, self.num_classes_)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                global_step += 1
            training_loss_list.append(total_loss / len(train_dataloader))


            model.eval()


            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    batch_X = batch_X.nan_to_num(0).to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_X)['logits'].reshape(-1, self.num_classes_)
                    val_loss += loss_fn(outputs, batch_y).item()
            val_loss /= len(val_dataloader)
            val_loss_list.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_state = deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= self.patience:
                    break

        best_val_auc = self._eval_auc(best_state, val_dataloader)

        return best_val_loss, best_val_auc, best_state, training_loss_list, val_loss_list

    def train(self, X, y, X_val=None, y_val=None):
        if X_val is None or y_val is None:
            # Direct call (legacy): split here. Prefer going through `fit`, which
            # splits before fitting the pre-processing.
            X, X_val, y, y_val = self._split_train_val(X, y)

        # Balanced class weights (matches official GradTree implementation)
        class_weights = sklearn.utils.class_weight.compute_class_weight(
            'balanced', classes=np.unique(y), y=y
        )
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        train_dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y).long())
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).long())


        # Candidate (lr_index, lr_values, lr_leaf, focal_loss) tuples: full
        # Cartesian product. When grid search is on, the loss is searched too.
        if self.grid_search:
            import itertools
            candidates = list(itertools.product(
                self.grid_search_lr_index,
                self.grid_search_lr_values,
                self.grid_search_lr_leaf,
                self.grid_search_focal_loss,
            ))
        else:
            candidates = [(self.learning_rate_index, self.learning_rate_values, self.learning_rate_leaf, self.focal_loss)]

        # Start every candidate from the SAME initialization. This preserves
        # init_from_mothernet: `self.model` already holds the MotherNet-initialized
        # parameters, so we must not discard them by building fresh random models.
        init_state = deepcopy(self.model.state_dict())

        runs = self._run_grid_search(candidates, init_state, train_dataset, val_dataset, class_weight_tensor)

        # Select by validation AUC (scale-invariant, fair across CE and focal).
        def _auc_key(r):
            auc = r['best_val_auc']
            if auc is None or (isinstance(auc, float) and np.isnan(auc)):
                return float('-inf')
            return auc

        best_run = max(runs, key=_auc_key)

        self.model = self._make_model()
        self.model.load_state_dict(best_run['best_state'])
        self.model = self.model.to(self.device)

        # Persist the chosen hyperparameters + grid search results.
        self.learning_rate_index = best_run['learning_rate_index']
        self.learning_rate_values = best_run['learning_rate_values']
        self.learning_rate_leaf = best_run['learning_rate_leaf']
        self.focal_loss = best_run['focal_loss']
        self.best_val_loss_ = best_run['best_val_loss']
        self.best_val_auc_ = best_run['best_val_auc']
        self.grid_search_results_ = [
            {k: r[k] for k in ('learning_rate_index', 'learning_rate_values', 'learning_rate_leaf', 'focal_loss', 'best_val_loss', 'best_val_auc')}
            for r in runs
        ]

        if self.ensemble_number is not None:
            ckpt_path = os.path.join(self.save_ckpt_path, f"gradtree_model_{self.dataset_name}_{self.split_number}_component_{self.ensemble_number}.cpkt")
            if not os.path.exists(self.save_ckpt_path):
                os.makedirs(self.save_ckpt_path)
        else:
            ckpt_path = os.path.join(self.save_ckpt_path, f"gradtree_model_{self.dataset_name}_{self.split_number}.cpkt")
            if not os.path.exists(self.save_ckpt_path):
                os.makedirs(self.save_ckpt_path)
        checkpoint_info = {
            'learning_rate_index': self.learning_rate_index,
            'learning_rate_values': self.learning_rate_values,
            'learning_rate_leaf': self.learning_rate_leaf,
            'focal_loss': self.focal_loss,
            'focal_loss_gamma': self.focal_loss_gamma,
            'best_val_loss': self.best_val_loss_,
            'best_val_auc': self.best_val_auc_,
            'grid_search': self.grid_search,
            'grid_search_results': self.grid_search_results_,
            'depth': self.depth,
            'epochs': self.epochs,
        }
        torch.save([self.model.state_dict(), [self.model.input_dim, self.model.output_dim], checkpoint_info], ckpt_path)

        plt.plot(best_run['train_loss'], label="train_loss")
        plt.plot(best_run['val_loss'], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        title = f"Training Curve for GradTree on {self.dataset_name} fold {self.split_number}"
        if self.grid_search:
            title += f" (best lr_index={self.learning_rate_index}, lr_values={self.learning_rate_values}, lr_leaf={self.learning_rate_leaf}, focal={self.focal_loss})"
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(self.visualization_root, f"gradtree_training_curve_{self.dataset_name}_{self.split_number}.png"))
        plt.clf()
        print(f"Model saved to {ckpt_path}")
        # if self.grid_search:
        #     print(f"Grid search results: {self.grid_search_results_}")
        #     print(f"Best: lr_index={self.learning_rate_index}, lr_values={self.learning_rate_values}, lr_leaf={self.learning_rate_leaf}, focal_loss={self.focal_loss} (val_loss={self.best_val_loss_:.4f})")
