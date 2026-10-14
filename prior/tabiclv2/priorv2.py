import torch
from typing import Optional, Union, Dict, Any, Tuple, Callable, List
from scipy.stats import loguniform
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from prior.tabiclv2._graph_scm import GraphSCM
import math
import threadpoolctl
from prior.tabiclv2.prior_configs import PriorConfig
from sklearn.ensemble import ExtraTreesRegressor
import multiprocessing as mp
import functools
import psutil
from torch.nested import nested_tensor



def run_parallel(func: Callable, args: List[Any], n_jobs: int = -1) -> List[Any]:
    """
    Uses a multiprocessing.Pool to evaluate func on all values in args, with n_jobs processes running in parallel.

    :param func: Function to be called.
    :param args: List of arguments to call the function with.
    :param n_jobs: Number of jobs (if -1, set to the number of physical cores).
    :return: Returns the list of values returned by the function evaluated on different values from args.
    """
    ctx = mp.get_context(method='fork')  # not sure if this is necessary
    with ctx.Pool(processes=n_jobs if n_jobs >= 1 else psutil.cpu_count(logical=False),
                 initializer=functools.partial(torch.set_num_threads, 1)) as pool:
        return pool.map(func, args)


def should_filter(
        X: torch.Tensor,
        y: torch.Tensor,
        config: PriorConfig,
        *,
        is_classif: bool,
        n_estimators: int = 25,
        n_boot: int = 200,
) -> bool:
    if not (config.remove_trivial_datasets or config.filter_unpredictable_datasets):
        return False

    if is_classif:
        y_int = y.to(torch.long)
        C = int(y_int.max().item() + 1)
        Y = F.one_hot(y_int, num_classes=C).to(torch.float32)  # (n, C)
        if C == 2:
            # drop one dimension to make it faster
            Y = Y[:, :1]
    else:
        Y = y.reshape(-1, 1).to(torch.float32)  # (n, 1)

    # sklearn needs numpy
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # do the squeeze to avoid a sklearn warning
    et = ExtraTreesRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        oob_score=True,
        n_jobs=1,
        random_state=1,  # don't set to 0 because it doesn't give OOB scores for all samples for some 133 <= n <= 257
        max_depth=6,  # make it faster
    ).fit(X_np, np.squeeze(Y_np, axis=-1) if Y_np.shape[-1] == 1 else Y_np)

    Yhat = et.oob_prediction_  # (n, d)
    if len(Yhat.shape) == 1:
        Yhat = Yhat[:, None]
    mask = ~np.isnan(Yhat).any(axis=1)  # keep valid OOB rows
    Yv, Yhatv = Y_np[mask], Yhat[mask]

    baseline = Y_np.mean(axis=0, keepdims=True)  # (1, d)
    imp = ((Yv - baseline) ** 2).sum(axis=1) - ((Yv - Yhatv) ** 2).sum(axis=1)

    baseline_mse = ((Yv - baseline) ** 2).sum(axis=1).mean()
    extratrees_mse = ((Yv - Yhatv) ** 2).sum(axis=1).mean()

    # Vectorized bootstrap (no retraining)
    rng = np.random.default_rng(0)
    n = imp.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))

    if config.remove_trivial_datasets and np.sqrt(extratrees_mse) <= config.trivial_dataset_threshold * np.sqrt(baseline_mse):
        return True
    if config.filter_unpredictable_datasets and pval >= 0.05:
        return True
    return False



class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
    ):
        self.regression = regression
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = 0 if regression else max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(min_train_size: Union[int, float], max_train_size: Union[int, float]) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(max_train_size, (int, float)):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert 0 < min_train_size < max_train_size, "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            # Allow min == max to express a fixed train fraction (e.g. the 80% used in
            # TabICLv2 pretraining stages 2 & 3); the float sampler uses np.random.uniform.
            assert 0 < min_train_size <= max_train_size < 1, "0 < min_train_size <= max_train_size < 1"
        else:
            raise ValueError("Both training sizes must be of the same type (int or float)")

    @staticmethod
    def sample_seq_len(
            min_seq_len: Optional[int], max_seq_len: int, log: bool = False, replay_small: bool = False
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(min_train_size: Union[int, float], max_train_size: Union[int, float], seq_len: int) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return max_features
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(xi[:, unique_mask], pad=(0, num_features - di_new), mode="constant", value=0)
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

    @staticmethod
    def cls_sanity_check(X: Tensor, y: Tensor, train_size: int, n_attempts: int = 10, min_classes: int = 2) -> bool:
        """
        Verifies that both train and test sets contain all classes for classification datasets.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target Targets tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return set(unique_tr.tolist()) == set(unique_te.tolist()) and len(unique_tr) >= min_classes

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class GraphPrior(Prior):
    """
    Generates synthetic datasets using Graph-based Structural Causal Models (SCM).

    Parameters
    ----------
    regression : bool, default=False
        If True, generates regression datasets. Otherwise, generates classification datasets.

    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    log_n_features : bool, default=False
        If True, sample the number of features from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    config : PriorConfig | None, default=None
        Prior configuration.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
            self,
            regression: bool = False,
            batch_size: int = 256,
            batch_size_per_gp: int = 4,
            batch_size_per_subgp: Optional[int] = None,
            min_features: int = 2,
            max_features: int = 100,
            max_classes: int = 10,
            min_seq_len: Optional[int] = None,
            max_seq_len: int = 1024,
            log_seq_len: bool = False, # False
            log_n_features: bool = False,
            seq_len_per_gp: bool = False, # True in stage one
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
            config: Optional[PriorConfig] = None,
            n_jobs: int = -1,
            num_threads_per_generate: int = 1,
            device: str = "cpu",
    ):
        super().__init__(
            regression=regression,
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.log_n_features = log_n_features
        self.config = config or PriorConfig()
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.

        This method implements a robust generation process with:
        1. Multiple retry attempts for generating valid datasets
        2. Feature filtering to remove non-informative columns
        3. Validation checks to ensure class balance in train/test splits

        Parameters
        ----------
        params : dict
            Parameters accepted by `GraphSCM` for dataset generation including:
            - regression: boolean flag for regression/classification
            - seq_len: number of samples
            - train_size: position to split train/test sets
            - num_features: number of features
            - max_features: maximum number of features
            - num_classes: number of target classes
            - device: computation device
            - config: PriorConfig object

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Targets tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """

        while True:
            X, y = GraphSCM(**params)()

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

            # Only keep valid datasets with sufficient features and balanced classes
            X, d = self.delete_unique_features(X, d)
            if not (d > 0).all():
                continue

            if not (self.regression or self.cls_sanity_check(X, y, params["train_size"])):
                continue

            if should_filter(X[0], y[0], self.config, is_classif=not self.regression):
                continue

            return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Targets tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
            )
            global_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, global_seq_len)

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len, self.max_seq_len, log=self.log_seq_len, replay_small=self.replay_small
                )
                gp_train_size = self.sample_train_size(self.min_train_size, self.max_train_size, gp_seq_len)
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(gp_seq_len, self.max_features)
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp)
                if actual_subgp_size <= 0:
                    break

                # Each subgroup shares the same number of features
                if self.log_n_features:
                    subgp_num_features = int(loguniform.rvs(self.min_features, gp_max_features))
                else:
                    subgp_num_features = round(np.random.uniform(self.min_features, gp_max_features))

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    if self.regression:
                        ds_num_classes = None
                    else:
                        # Each dataset has its own number of classes
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)

                    # Create parameters dictionary for this dataset
                    params = {
                        # -----------------
                        # Global parameters
                        # -----------------
                        "regression": self.regression,
                        "device": self.device,
                        "config": self.config,
                        # -------------------------
                        # Group-specific parameters
                        # -------------------------
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        "max_features": gp_max_features,
                        # ----------------------------
                        # Subgroup-specific parameters
                        # ----------------------------
                        "num_features": subgp_num_features,
                        # ---------------------------
                        # Dataset-specific parameters
                        # ---------------------------
                        "num_classes": ds_num_classes,
                        # ---------------------------
                        # DAG generation parameters
                        # ---------------------------
                    }

                    param_list.append(params)

        with threadpoolctl.threadpool_limits(limits=1):
            if self.n_jobs == 1:
                n_threads_old = torch.get_num_threads()
                torch.set_num_threads(1)
                results = [self.generate_dataset(params) for params in param_list]
                torch.set_num_threads(n_threads_old)
            else:
                results = run_parallel(self.generate_dataset, args=param_list, n_jobs=self.n_jobs)

        X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(self.device)  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor([params["seq_len"] for params in param_list], device=self.device, dtype=torch.long)
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list], device=self.device, dtype=torch.long
        )

        return X, y, d, seq_lens, train_sizes

