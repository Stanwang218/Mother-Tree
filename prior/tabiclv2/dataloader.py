import os
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from typing import Optional, Union, Tuple
from prior.tabiclv2.prior_configs import PriorConfig
from torch import Tensor
from prior.tabiclv2.priorv2 import GraphPrior
import argparse



class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------

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

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', or 'dummy'

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models

        2. Dummy: Randomly generated datasets for debugging

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    config : PriorConfig | None, default=None
        Prior configuration when using 'graph_scm' prior_type

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

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
            log_seq_len: bool = False,
            log_n_features: bool = False,
            seq_len_per_gp: bool = False,
            min_train_size: Union[int, float] = 0.1,
            max_train_size: Union[int, float] = 0.9,
            replay_small: bool = False,
            prior_type: str = "mlp_scm",
            # scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
            # scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
            config: Optional[PriorConfig] = None,
            n_jobs: int = -1,
            num_threads_per_generate: int = 1,
            device: str = "cpu",
    ):
        super().__init__()
        if prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
            raise NotImplementedError("SCMPrior is not implemented in this version. Please use 'graph_scm'")
        elif prior_type == "graph_scm":
            self.prior = GraphPrior(
                regression=regression,
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                log_n_features=log_n_features,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                config=config,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', 'graph_scm', or 'dummy'."
            )

        self.regression = regression
        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = 0 if regression else max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class Targets of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        return self.prior.get_batch(batch_size or self.batch_size)

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        # with DisablePrinting():
        #     return self.get_batch()
        return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  task: {'regression' if self.regression else 'classification'}\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )



# Class for loading the tabiclv2 dataset from disk in a distributed manner. 
# Each rank will read different files, and the loader can shuffle the file order if desired. 
# It supports loading batches of data and can optionally delete files after loading.
class TabICLv2DiskDataLoader:
    """Infinite iterator over pre-generated tabiclv2 .pt batches on disk.

    Each rank reads different files with DDP support:
        rank 0 → batch_000000, batch_000004, batch_000008, ...
        rank 1 → batch_000001, batch_000005, batch_000009, ...

    Parameters
    ----------
    data_dir : str
        Directory containing batch_XXXXXX.pt files.

    batch_size : int, default=512
        Number of datasets to return in each iteration.

    shuffle : bool, default=False
        Whether to shuffle file order on each pass.

    seed : int, optional
        Random seed for shuffling.

    device : str, default='cpu'
        Device to load tensors to.

    start_from : int, default=0
        Batch index to start loading from.

    max_batches : int, optional
        Maximum number of batches to load. If None, loop indefinitely.

    delete_after_load : bool, default=False
        Whether to delete files after loading.
    """

    def __init__(self, data_dir, batch_size=256, shuffle=False, seed=None,
                 device='cpu', start_from=0, max_batches=None,
                 delete_after_load=False, size_per_group=None, total_steps=None):
        import glob
        self.files = sorted(glob.glob(os.path.join(data_dir, 'batch_*.pt')))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.pt files found in {data_dir}")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.device = torch.device(device)
        self.max_batches = max_batches
        self.delete_after_load = delete_after_load
        self.batch_size_per_gp = size_per_group
        self.total_steps = total_steps  # if None, fall back to file count below

        # DDP detection at init time
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        # Each rank gets its own file list with stride
        self.my_files = self.files[self.rank + start_from :: self.world_size]
        self.my_idx = 0

        # If total_steps not given, default to number of files for this rank
        if self.total_steps is None:
            self.total_steps = len(self.my_files) if max_batches is None else max_batches

        print(f"TabICLv2DiskDataLoader: {len(self.files)} total files, "
              f"rank {self.rank}/{self.world_size} → {len(self.my_files)} files, "
              f"shuffle={self.shuffle}, device={self.device}")

    def _load_one_file(self):
        """Load a single .pt file, return (X, y, d, seq_lens, train_sizes, n)."""
        if self.max_batches is not None and self.my_idx >= len(self.my_files):
            raise StopIteration

        idx = self.my_idx % len(self.my_files)
        self.my_idx += 1
        # self.max_batches = max_batches if max_batches is not None else len(self.my_files)

        # Reshuffle when we wrap around
        if self.shuffle and idx == 0 and self.my_idx > 1:
            self.rng.shuffle(self.my_files)

        filepath = self.my_files[idx]
        batch = torch.load(filepath, map_location=self.device, weights_only=True)

        X = batch['X']
        y = batch['y']
        d = batch['d']
        seq_lens = batch['seq_lens']
        train_sizes = batch['train_sizes']
        n = batch['batch_size']

        if self.delete_after_load and os.path.exists(filepath):
            os.remove(filepath)

        return X, y, d, seq_lens, train_sizes, n

    def __iter__(self):
        return self

    def __next__(self):
        X, y, d, seq_lens, train_sizes, n = self._load_one_file()
        return X, y, d, seq_lens, train_sizes

def get_tabiclv2_dataset(dataset_config_path):
    import json
    parser = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(parser)
    args = parser.parse_args([])  
    with open(dataset_config_path, "r") as f:
        dataset_config = json.load(f)
    

    for k, v in dataset_config.items():
        setattr(args, k, v)

    dataset = PriorDataset(
                prior_type= args.prior_type,
                regression= False,
                batch_size= args.batch_size,
                batch_size_per_gp=args.batch_size_per_gp,
                min_features=args.min_features,
                max_features=args.max_features,
                max_classes=args.max_classes,
                min_seq_len=args.min_seq_len,
                max_seq_len=args.max_seq_len,
                log_seq_len=args.log_seq_len,
                # log_n_features=args.log_n_features,
                seq_len_per_gp=args.seq_len_per_gp,
                min_train_size=args.min_train_size,
                max_train_size=args.max_train_size,
                n_jobs=args.n_jobs,
        config=PriorConfig.from_args(args),
        ) # batch, seq, feature
    return dataset, dataset_config

def get_tabiclv2_dataloader(dataset_config_path):
    dataset, dataset_config = get_tabiclv2_dataset(dataset_config_path)
    dataloader = DataLoader(
            dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=16,
            prefetch_factor=1,
            pin_memory=True if dataset_config.get("prior_device") == "cpu" else False,
            pin_memory_device=dataset_config.get("device") if dataset_config.get("prior_device") == "cpu" else "",
        )
    return dataloader

def generate_one_batch_tabiclv2(batch_idx, config):
    """Generate a single batch of tabiclv2 data. Called by multiprocessing workers.

    Args:
        batch_idx: Index of this batch.
        config: Dict with keys:
            dataset_config_path, save_dir, seed, max_retries

    Returns:
        (batch_idx, batch_size) or (batch_idx, None) on failure.
    """
    import time as _time
    dataset_config_path = config['dataset_config_path']
    save_dir = config['save_dir']
    MAX_RETRIES = config.get('max_retries', 20)
    base_seed = config.get('seed', 42)

    # Each worker gets a unique seed
    worker_seed = (base_seed + batch_idx * 10007) % (2**31)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    dataset, _ = get_tabiclv2_dataset(dataset_config_path)

    for retry in range(MAX_RETRIES):
        try:
            X, y, d, seq_lens, train_sizes = dataset.get_batch()
            break
        except (ValueError, RuntimeError, AssertionError):
            if retry == MAX_RETRIES - 1:
                return batch_idx, None
            _time.sleep(0.1)
            continue

    # Save as .pt file (compatible with LoadPriorDataset format)
    batch_file = os.path.join(save_dir, f"batch_{batch_idx:06d}.pt")
    temp_file = batch_file + ".tmp"
    torch.save(
        {"X": X.cpu(), "y": y.cpu(), "d": d.cpu(), "seq_lens": seq_lens.cpu(),
         "train_sizes": train_sizes.cpu(), "batch_size": len(d)},
        temp_file,
    )
    os.replace(temp_file, batch_file)

    return batch_idx, len(d)