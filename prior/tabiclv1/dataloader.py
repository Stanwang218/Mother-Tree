from prior.tabiclv1.prior_bag import BagPrior
from prior.tabiclv1.classification_adapter import ClassificationAdapterPrior
from prior.tabiclv1.fast_gp import GPPrior
from prior.tabiclv1.mlp import MLPPrior
from prior.tabiclv1.treeSCM import TreeSCMPrior
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


def sample_seq_len(min_seq_len, max_seq_len, log=False):
    """Sample a random sequence length within the specified range.

    Args:
        min_seq_len: Minimum sequence length. If None, returns max_seq_len directly.
        max_seq_len: Maximum sequence length.
        log: If True, sample from log-uniform distribution.

    Returns:
        int: Sampled sequence length.
    """
    if min_seq_len is None:
        return max_seq_len
    if log:
        return int(np.exp(np.random.uniform(np.log(min_seq_len), np.log(max_seq_len))))
    return np.random.randint(min_seq_len, max_seq_len + 1)


def sample_train_size(min_train_size, max_train_size, seq_len):
    """Sample a random train/test split position.

    Args:
        min_train_size: If int, absolute position. If float (0, 1), ratio of seq_len.
        max_train_size: If int, absolute position. If float (0, 1), ratio of seq_len.
        seq_len: Total sequence length.

    Returns:
        int: Train size position.
    """
    if isinstance(min_train_size, int) and isinstance(max_train_size, int):
        return np.random.randint(min_train_size, max_train_size + 1)
    elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
        ratio = np.random.uniform(min_train_size, max_train_size)
        return int(seq_len * ratio)
    raise ValueError("min_train_size and max_train_size must be both int or both float")


class _PriorIterableDataset(IterableDataset):
    """Thin wrapper so PyTorch DataLoader's num_workers can be used."""
    def __init__(self, prior, total_steps, batch_size, min_eval_pos, n_samples, device, num_features,
                 random_n_samples, n_test_samples, epoch_count=0):
        self.prior = prior
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.min_eval_pos = min_eval_pos
        self.n_samples = n_samples
        self.device = device
        self.num_features = num_features
        self.random_n_samples = random_n_samples
        self.n_test_samples = n_test_samples
        self.epoch_count = epoch_count

    def __len__(self):
        return self.total_steps

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Each worker gets a unique seed to avoid identical batches
            torch.manual_seed(worker_info.seed)
            np.random.seed(worker_info.seed % 2**31)
        for _ in range(self.total_steps):
            single_eval_pos = None
            actual_n = self.n_samples
            if self.random_n_samples:
                actual_n = np.random.randint(self.min_eval_pos, self.n_samples)
                single_eval_pos = actual_n - self.n_test_samples
            if single_eval_pos is None:
                single_eval_pos = np.random.randint(self.min_eval_pos, actual_n)
            try:
                batch = self.prior.get_batch(
                    batch_size=self.batch_size,
                    n_samples=actual_n,
                    num_features=self.num_features,
                    device=self.device,
                    epoch=self.epoch_count,
                    single_eval_pos=single_eval_pos,
                )
            except AssertionError:
                continue
            x, y, info = batch if len(batch) == 3 else (batch[0], batch[1], None)
            yield (info, x, y), y, single_eval_pos

class PriorDataLoader(DataLoader):
    def __init__(
        self, 
        prior, 
        total_steps, 
        batch_size, 
        min_eval_pos, 
        n_samples, 
        device, 
        num_features,
        model = None,
        random_n_samples = False,
        n_test_samples = False,
        num_workers=4,
        **kwargs
    ):

        if (random_n_samples and not n_test_samples) or (not random_n_samples and n_test_samples):
            raise ValueError("random_n_samples and test_samples must be set together.")
        
        self.prior = prior
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.min_eval_pos = min_eval_pos
        self.random_n_samples = random_n_samples
        if random_n_samples:
            self.n_samples = np.random.randint(min_eval_pos, random_n_samples)
            self.n_test_samples = n_test_samples
        else:
            self.n_samples = n_samples
        self.device = device
        self.num_features = num_features
        self.epoch_count = 0
        self.model = model

        # Build a proper IterableDataset so num_workers works, class from MotherNet repo, should be changed in the future
        dataset = _PriorIterableDataset(
            prior=prior, total_steps=total_steps, batch_size=batch_size,
            min_eval_pos=min_eval_pos, n_samples=self.n_samples,
            device=device, num_features=num_features,
            random_n_samples=random_n_samples, n_test_samples=n_test_samples,
            epoch_count=self.epoch_count,
        )
        super().__init__(dataset, batch_size=None, num_workers=num_workers)

    @property
    def num_steps(self):
        """Backward compat alias for total_steps."""
        return self.total_steps

    @staticmethod
    def save_data(data, save_path=None): # remember to load eval_pos together with the data
        import pickle
        import os
        num = 0
        path = os.path.join(save_path, f'{num}.pkl')
        while os.path.exists(path):
            num += 1
            path = os.path.join(save_path, f'{num}.pkl')

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def gbm(self, epoch=None, single_eval_pos=None):
        # Direct batch generation (bypasses DataLoader workers, useful for eval)
        if self.random_n_samples:
            single_eval_pos = self.n_samples - self.n_test_samples
        if single_eval_pos is None:
            single_eval_pos = np.random.randint(self.min_eval_pos, self.n_samples)
        batch = self.prior.get_batch(
            batch_size=self.batch_size,
            n_samples=self.n_samples,
            num_features=self.num_features,
            device=self.device,
            epoch=epoch if epoch is not None else self.epoch_count,
            single_eval_pos=single_eval_pos,
        )
        x, y, info = batch if len(batch) == 3 else (batch[0], batch[1], None)
        return (info, x, y), y, single_eval_pos

    def get_test_batch(self):
        return self.gbm(epoch=self.epoch_count)

def generate_one_batch(batch_idx, config):
    """Generate a single batch. Called by joblib workers (must be picklable)."""
    import os, pickle
    import numpy as np
    import torch
    # from ticl.priors import ClassificationAdapterPrior, BagPrior

    save_dir = config['save_dir']
    MAX_RETRIES = config['max_retries']

    # Re-sample with a different seed per worker
    np.random.seed((config.get('seed', 42) + batch_idx * 10007) % (2**31))

    n_samples = sample_seq_len(config['min_seq_len'], config['max_seq_len'], log=config['log_seq_len'])
    single_eval_pos = sample_train_size(config['min_train_size'], config['max_train_size'], n_samples)

    # Build prior inside worker (avoids pickling issues)
    pc = config['prior_config']
    gp_flexible = ClassificationAdapterPrior(
        GPPrior(pc['gp'], batch_size_per_subgroup=config['batch_size_per_gp']),
        num_features=pc['num_features'], padding=config['padding'],
        **pc['classification'])
    mlp_flexible = ClassificationAdapterPrior(
        MLPPrior(pc['mlp'], batch_size_per_subgroup=config['batch_size_per_gp']),
        num_features=pc['num_features'], padding=config['padding'],
        **pc['classification'])

    if config['prior_type'] == 'prior_bag':
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                            prior_weights={'mlp': 0.961, 'gp': 0.039})
    elif config['prior_type'] == 'prior_bag_plus_tree':
        tree_flexible = ClassificationAdapterPrior(
            TreeSCMPrior(pc['tree'], batch_size_per_subgroup=config['batch_size_per_gp']),
            num_features=pc['num_features'], padding=config['padding'],
            **pc['classification'])
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'tree': tree_flexible},
                            prior_weights={'mlp': 0.661, 'gp': 0.039, 'tree': 0.3})
    elif config['prior_type'] == "tree_only":
        tree_flexible = ClassificationAdapterPrior(
            TreeSCMPrior(pc['tree'], batch_size_per_subgroup=config['batch_size_per_gp']),
            num_features=pc['num_features'], padding=config['padding'],
            **pc['classification'])
        prior = BagPrior(base_priors={'tree': tree_flexible}, prior_weights={'tree': 1.0})
    else:
        raise ValueError(f"Unknown prior_type: {config['prior_type']}")

    device = torch.device(config['device'])

    for retry in range(MAX_RETRIES):
        try:
            x, y, info = prior.get_batch(
                batch_size=config['batch_size'],
                n_samples=n_samples,
                num_features=config['max_features'],
                device=device,
                single_eval_pos=single_eval_pos,
            )
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError("NaN or Inf in x")
            if torch.isnan(y).any() or torch.isinf(y).any():
                raise ValueError("NaN or Inf in y")
            break
        except (AssertionError, ValueError):
            if retry == MAX_RETRIES - 1:
                return batch_idx, None  # failed
            continue

    # Use batch_idx in filename to avoid race conditions in parallel mode
    path = os.path.join(save_dir, f'{batch_idx}.pkl')

    data = {
        'x': x.cpu(),
        'y': y.cpu(),
        'single_eval_pos': single_eval_pos,
        'n_samples': n_samples,
        'info': info,
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)

    return batch_idx, tuple(x.shape)


def get_dataloader(prior_config, dataloader_config, device, model = None, padding=True):

    prior_type = prior_config['prior_type']
    batch_size_per_subgroup = prior_config.get('batch_size_per_subgroup', None)
    gp_flexible = ClassificationAdapterPrior(GPPrior(prior_config['gp'], batch_size_per_subgroup=batch_size_per_subgroup), num_features=prior_config['num_features'], padding=padding, **prior_config['classification'])
    mlp_flexible = ClassificationAdapterPrior(MLPPrior(prior_config['mlp'], batch_size_per_subgroup=batch_size_per_subgroup), num_features=prior_config['num_features'], padding=padding, **prior_config['classification'])
    tree_flexible = ClassificationAdapterPrior(TreeSCMPrior(prior_config['tree'], batch_size_per_subgroup=batch_size_per_subgroup), num_features=prior_config['num_features'], padding=padding, **prior_config['classification'])
    
    if prior_type == 'prior_bag':
        # Prior bag combines priors
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                         prior_weights={'mlp': 0.961, 'gp': 0.039})
    elif prior_type == 'prior_bag_plus_tree':
        # Prior bag combines priors
        prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'tree': tree_flexible},
                         prior_weights={'mlp': 0.661, 'gp': 0.039, 'tree': 0.3})
    # elif prior_type == "step_function":
    #     prior = priors.StepFunctionPrior(prior_config['step_function'])
    # elif prior_type == "boolean_only":
    #     prior = BooleanConjunctionPrior(hyperparameters=prior_config['boolean'])
    # elif prior_type == "bag_boolean":
    #     boolean = BooleanConjunctionPrior(hyperparameters=prior_config['boolean'])
    #     prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible, 'boolean': boolean},
    #                      prior_weights={'mlp': 0.9, 'gp': 0.02, 'boolean': 0.08})
    else:
        raise ValueError(f"Prior type {prior_type} not supported.")

    return PriorDataLoader(
        prior=prior, 
        n_samples=prior_config['n_samples'],
        device=device, 
        num_features=prior_config['num_features'], 
        model = model,
        **dataloader_config
    )
