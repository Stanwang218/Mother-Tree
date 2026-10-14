from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from prior.tabiclv2.graph_lib._base import Context, DatasetProperties
from prior.tabiclv2.graph_lib._config import PriorConfig
from prior.tabiclv2.graph_lib._properties import sample_categorical_sizes
from prior.tabiclv2.graph_lib._dataset import RandomDataset

import random
import warnings

import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F


def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Calculates the standard deviation of a tensor, ignoring NaNs, using NumPy internally.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    dim : int or tuple[int], optional
        The dimension or dimensions to reduce. Defaults to None (reduce all dimensions).

    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not. Defaults to False.

    ddof : int, optional
        Delta Degrees of Freedom.

    dtype : torch.dtype, optional
        The desired data type of returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    device = input.device
    np_input = input.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)

    return torch.from_numpy(std).to(dtype=torch.float, device=device)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardizes features by removing the mean and scaling to unit variance.

    NaNs are ignored in mean/std calculation.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H), where T is sequence length, H is features.

    clip_value : float, optional, default=100
        The value to clip the standardized input to, preventing extreme outliers.

    Returns
    -------
    Tensor
        The standardized input, clipped between -clip_value and clip_value.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std

    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamps outliers in the input tensor based on a specified number of standard deviations (threshold).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H).

    threshold : float, optional, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking -> std is NaN or 0
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


def permute_classes(input: Tensor) -> Tensor:
    """Label encoding and permute classes.

    Parameters
    ----------
    input : Tensor
        Target of shape (T,) containing class labels.

    Returns
    -------
    Tensor
        Target with potentially permuted labels (T,).
    """
    unique_vals, _ = torch.unique(input, return_inverse=True)
    num_classes = len(unique_vals)

    if num_classes <= 1:  # No permutation needed for single class
        return input

    # Ensure labels are encoded from 0 to num_classes-1
    indices = unique_vals.argsort()
    mapped = indices[torch.searchsorted(unique_vals, input)]

    # Randomly permute classes
    perm = torch.randperm(num_classes, device=input.device)
    permuted = perm[mapped]

    return permuted


class BalancedBinarize(nn.Module):
    """Binarizes the input based on its median value."""

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).

        Returns
        -------
        Tensor
            Binarized output (0 or 1) of shape (T,).
        """
        return (input > torch.median(input)).float()


class MulticlassAssigner(nn.Module):
    """Transforms the input into discrete classes using rank-based or value-based thresholding.

    Input shape: (T,) -> Output shape: (T,)
    """

    def __init__(self, num_classes: int, mode: str = "rank", ordered_prob: float = 0.2):
        """
        Initializes the MulticlassAssigner.

        Parameters
        ----------
        num_classes : int
            The target number of discrete classes to output.

        mode : str, default="rank"
            The method used to determine class boundaries:
            - "rank": Boundaries are randomly sampled from the input.
            - "value": Boundaries are randomly sampled from a normal distribution.

        ordered_prob : float, default=0.2
            Probability of keeping the natural class order.
        """
        super().__init__()
        if num_classes < 2:
            raise ValueError("The number of classes must be at least 2 for MulticlassAssigner.")

        self.num_classes = num_classes
        self.ordered_prob = ordered_prob
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor
            Input of shape (T,).

        Returns
        -------
        Tensor
            Class labels of shape (T,) with integer values [0, num_classes-1].
        """

        T = input.shape[0]
        device = input.device

        if self.mode == "rank":
            boundary_indices = torch.randint(0, T, (self.num_classes - 1,), device=device)
            boundaries = input[boundary_indices]
        elif self.mode == "value":
            boundaries = torch.randn(self.num_classes - 1, device=device)

        # Compare input tensor with boundaries and sum across the boundary dimension to get classes
        classes = (input.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

        # Permute classes
        if random.random() > self.ordered_prob:
            classes = permute_classes(classes)

        # Reverse classes
        if random.random() > 0.5:
            classes = self.num_classes - 1 - classes

        return classes


class Reg2Cls(nn.Module):
    """Transforms a single regression dataset (features X, targets y) into a classification format
    through feature processing (categorical conversion, normalization) and target transformation
    (regression-to-classification).

    Parameters
    ----------
    hyperparameters : dict
        Configuration dictionary containing settings for feature processing and
        target transformation. Expected keys include:
        - regression (bool): Whether the transformed dataset is for regression or classification.
        - num_classes (int): Number of classes for classification conversion.
        - max_features (int): Maximum number of features allowed (defines output feature dim).
        - multiclass_type (str): Strategy for multiclass conversion ('rank' or 'value').
        - balanced (bool): Whether to enforce balanced classes (currently only for binary).
        - multiclass_ordered_prob (float): Prob. of keeping natural class order.
        - cat_prob (float, optional): Probability of converting features to categorical.
        - max_categories (int, optional): Max categories for categorical conversion.
        - scale_by_max_features (bool): Whether to scale features by proportion used.
        - permute_features (bool, optional): Whether to randomly permute features. Defaults to True.
        - permute_labels (bool, optional): Whether to randomly permute final class labels. Defaults to True.

    Attributes
    ----------
    hp : dict
        Hyperparameters for the Reg2Cls module.

    class_assigner : nn.Module or None
        The module responsible for converting regression targets to class labels.
        None if num_classes is 0.
    """

    def __init__(self, hp: dict):
        super().__init__()
        self.hp = hp

        if hp["regression"]:
            self.class_assigner = None
        else:
            num_classes = self.hp["num_classes"]
            if num_classes == 2 and self.hp.get("balanced", False):
                self.class_assigner = BalancedBinarize()
            elif num_classes >= 2:
                self.class_assigner = MulticlassAssigner(
                    num_classes, mode=self.hp["multiclass_type"], ordered_prob=self.hp["multiclass_ordered_prob"]
                )
            else:
                raise ValueError(f"Invalid number of classes: {num_classes}")

    def forward(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Processes a single dataset (X, y) according to the initialized hyperparameters.

        Parameters
        ----------
        X : Tensor
            Features of shape (T, H), where H is the number of features.

        y : Tensor
            Targets of shape (T,).

        Returns
        -------
        tuple[Tensor, Tensor]
            A tuple containing:
            - Processed features of shape (T, max_features).
            - Processed targets of shape (T,).
        """
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError(f"Input shapes mismatch or incorrect dims. X: {X.shape}, y: {y.shape}")

        X = self._num2cat(X)
        X = self._process_features(X)

        y = outlier_removing(y.unsqueeze(-1), threshold=4)
        y = standard_scaling(y).squeeze(-1)
        if self.class_assigner is not None:
            y = self.class_assigner(y)
            if self.hp.get("permute_labels", True):
                y = permute_classes(y)

        return X.float(), y.float()

    def _num2cat(self, X: Tensor) -> Tensor:
        """Converts some features to categorical based on hyperparameters.

        Operates inplace conceptually, returns the modified tensor.

        Parameters
        ----------
        X : Tensor
            Feature tensor of shape (T, H).

        Returns
        -------
        Tensor
            Feature tensor with some columns potentially converted to categorical (T, H).
        """

        if random.random() < self.hp.get("cat_prob", 0.2):
            col_prob = random.random()  # Probability for converting a specific column
            max_categories = self.hp.get("max_categories", 10)
            for col in range(X.shape[1]):
                if random.random() < col_prob:
                    # Determine number of categories for this feature
                    num_cats = min(max(round(random.gammavariate(1, 10)), 2), max_categories)
                    # Use MulticlassAssigner to convert this column
                    assigner = MulticlassAssigner(num_cats, mode="rank", ordered_prob=0.3)
                    X[:, col] = assigner(X[:, col]).float()
        return X

    def _process_features(self, X: Tensor) -> Tensor:
        """Process inputs through outlier removal, shuffling, scaling, and padding to max features.

        Parameters
        ----------
        X : Tensor
            Feature tensor of shape (T, H).

        Returns
        -------
        Tensor
            Normalized feature tensor (T, H).
        """

        num_features = X.shape[1]
        max_features = self.hp["max_features"]

        X = outlier_removing(X, threshold=4)
        X = standard_scaling(X)

        # Permute features if specified
        if self.hp.get("permute_features", True):
            perm = torch.randperm(num_features, device=X.device)
            X = X[:, perm]

        # Scale by the proportion of features used relative to max features
        if self.hp.get("scale_by_max_features", False):
            scaling_factor = num_features / max_features
            X = X / scaling_factor

        # Add empty features if needed to match max features
        if num_features < max_features:
            X = F.pad(X, (0, max_features - num_features), mode="constant", value=0.0)

        return X


class GraphSCM:
    """Generates a dataset with a causal graph structure.

    Parameters
    ----------
    regression : bool, default=False
        Whether the target is for regression or classification.

    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features.

    max_features : int, default=100
        Maximum number of features allowed.

    num_classes : int, default=2
        Number of classes for categorical target

    config : PriorConfig | None, default=None
        Configuration of the prior.

    cat_modes : List, default=None
        Categorical conversion modes that control how categorical features are processed.
        Each mode determines how continuous values are discretized and represented:
        - "id": Uses original continuous values for graph computation, returns category indices
        - "disc": Uses discretized cluster centers for graph computation, returns category indices
        - "disc_func": Applies random function to discretized centers, returns category indices
        - "int": Uses category indices as float values for graph computation, returns category indices
        - "softmax_id": Uses original values with softmax-based sampling, returns category indices
        - "softmax_disc": Uses softmax embeddings for graph computation, returns category indices
        - "softmax_int": Uses category indices from softmax sampling, returns category indices

        The discretization process creates clusters by finding nearest centers (for non-softmax modes)
        or using multinomial sampling with softmax probabilities (for softmax modes).

    fn_types : List, default=None
        List of function types to use for generating node mapping functions in the causal graph.
        Available function types:
        - "gp": Gaussian Process function using random Fourier features with Cauchy kernel
        - "generalized_gp": Enhanced GP function with optional product kernel and eigen-decay matrices
        - "nn": Neural Network function with random architecture (1-4 layers, configurable activations)
        - "tree": Random Forest of oblivious decision trees with random leaf values
        - "product": Product of two cheaper random functions for interaction modeling
        - "discretization": Nearest-neighbor discretization function with random target mapping
        - "linear": Simple linear transformation using random matrices
        - "flow": Normalizing flow function with iterative transformations

        Each function type creates different non-linear relationships between parent and child nodes
        in the causal graph, affecting the complexity and structure of generated features.

    permute_features : bool, default=True
        Whether to randomly permute features

    permute_labels : bool, default=True
        Whether to randomly permute class labels

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        regression: bool = False,
        seq_len: int = 1024,
        num_features: int = 100,
        max_features: int = 100,
        num_classes: int = 2,
        permute_features: bool = True,
        permute_labels: bool = True,
        config: Optional[PriorConfig] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super(GraphSCM, self).__init__()
        self.regression = regression
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_features = max_features
        self.num_classes = num_classes
        self.permute_features = permute_features
        self.permute_labels = permute_labels
        self.config = config
        self.device = device

    def __call__(self) -> None:
        """Generate a dataset and return features and target."""

        context = Context(config=self.config, device=self.device)
        properties = DatasetProperties(
            n_train=self.seq_len,
            n_test=0,
            cat_sizes={
                "x": sample_categorical_sizes(self.num_features, context, max_cat_size=200),
                "y": [0 if self.regression else self.num_classes],
            },
        )
        ds = RandomDataset(context).sample(properties)
        data = ds.get_concat_tensors()

        X_cat = data.get("x_cat", None)
        X_num = data.get("x_num", None)

        if X_cat is not None and X_num is not None:
            X = torch.cat([X_cat, X_num], dim=-1)
        elif X_cat is not None:
            X = X_cat
        elif X_num is not None:
            X = X_num
        else:
            raise ValueError("No features found in dataset")

        X = outlier_removing(X.float(), threshold=4)
        X = standard_scaling(X)

        if self.permute_features:
            feat_perm = torch.randperm(self.num_features, device=self.device)
            X = X[..., feat_perm]

        if self.num_features < self.max_features:
            X = F.pad(
                X, (0, self.max_features - self.num_features), mode="constant", value=0.0
            )  # (seq_len, max_features)

        if self.regression:
            y = data["y_num"]  # (seq_len, 1)
            y = outlier_removing(y.float(), threshold=4)
            y = standard_scaling(y)
            y = y.view(-1)  # (seq_len,)
        else:
            y = data["y_cat"].view(-1)  # (seq_len,)
            if self.permute_labels:
                class_perm = torch.randperm(self.num_classes, device=self.device)
                y = class_perm[y.long()]

        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)

        return X, y
