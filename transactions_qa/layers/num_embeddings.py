"""On Embeddings for Numerical Features in Tabular Deep Learning."""

import math
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(n_features, *)`
    - Output: `(*, n_features, d_embedding)`
    """

    def __init__(self, n_features: int, d_embeddings: List[int]) -> None:
        """
        Args:
            n_features: the number of continous features.
            d_embeddings: the embedding size for each feature.
        """
        self.n_features = n_features
        self.d_embeddings = d_embeddings
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features=}')
        if any(x <= 0 for x in d_embeddings):
            i, value = next((i, x) for i, x in enumerate(d_embeddings) if x <= 0)
            raise ValueError(
                'd_embeddings must contain only positive values,'
                f' however: d_embeddings[{i}]={value}'
            )

        super().__init__()
        # self.embeddings = nn.ModuleList([
        #     nn.Linear(1, d_embedding) for d_embedding in d_embeddings
        # ])
        self.weight = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.empty(1, d_embedding))
                       for d_embedding in d_embeddings])
        self.bias = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.empty(1, d_embedding))
                     for d_embedding in d_embeddings])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # _ = [torch.nn.init.xavier_uniform_(emb.weight) for emb in self.embeddings]
        d_rqsrt = torch.max(torch.LongTensor(self.d_embeddings)) ** -0.5
        _ = [nn.init.uniform_(w, -d_rqsrt, d_rqsrt) for w in self.weight]
        _ = [nn.init.uniform_(b, -d_rqsrt, d_rqsrt) for b in self.bias]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        x = [x[i][..., None] * self.weight[i] for i in range(len(self.weight))]
        x = [x[i] + self.bias[i][None] for i in range(len(self.bias))]
        # x = [embedding(x[i][..., None]) for i, embedding in enumerate(self.embeddings)]
        return torch.cat(x, dim=-1)


class CategoricalEmbeddings(nn.Module):
    """
    Embeddings for categorical features.
    """

    def __init__(
            self, cardinalities: List[int], d_embeddings: List[int],
            bias: Optional[bool] = True,
            add_missing: Optional[bool] = True
    ) -> None:
        """
        Args:
            cardinalities: the number of distinct values for each feature.
            d_embeddings: the embedding sizes for each feature.
            bias: if `True`, for each feature, a trainable vector is added to the
                embedding regardless of a feature value. For each feature, a separate
                non-shared bias vector is allocated.
                In the paper, FT-Transformer uses `bias=True`.
        """
        super().__init__()
        self.add_missing = add_missing
        if not cardinalities:
            raise ValueError('cardinalities must not be empty')
        if any(x <= 0 for x in cardinalities):
            i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
            raise ValueError(
                'cardinalities must contain only positive values,'
                f' however: cardinalities[{i}]={value}'
            )
        if any(x <= 0 for x in d_embeddings):
            i, value = next((i, x) for i, x in enumerate(d_embeddings) if x <= 0)
            raise ValueError(
                'd_embeddings must contain only positive values,'
                f' however: d_embeddings[{i}]={value}'
            )
        add_missing = 1 if self.add_missing else 0
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardin + add_missing, embedding_dim=d_embedding, padding_idx=0) for cardin, d_embedding
             in zip(cardinalities, d_embeddings)]
        )
        # self.bias = (
        #     torch.nn.parameter.Parameter(torch.empty(len(cardinalities), d_embedding)) if bias else None
        # )
        if bias:
            self.bias = torch.nn.ParameterList([
                torch.nn.parameter.Parameter(torch.empty(1, d_embedding)) for d_embedding in d_embeddings]
            )
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            _ = [torch.nn.init.uniform_(emb.weight, -d_rsqrt, d_rsqrt) for emb in self.embeddings]
        if self.bias is not None:
            _ = [torch.nn.init.uniform_(b, -d_rsqrt, d_rsqrt) for b in self.bias]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        n_features = len(self.embeddings)
        if x.shape[0] != n_features:
            raise ValueError(
                'The first input dimension (the number of categorical features) must be'
                ' equal to the number of cardinalities passed to the constructor.'
                f' However: {x.shape[0]=}, len(cardinalities)={n_features}'
            )

        # x = torch.stack(
        #     [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
        # )
        x = [self.embeddings[i](x[i, ...]) for i in range(n_features)]

        if self.bias is not None:
            # x = x + self.bias
            x = [x[i] + self.bias[i] for i in range(n_features)]
        x = torch.cat(x, dim=-1)
        return x


class LinearReLUEmbeddings(nn.Sequential):
    """LR (L ~ Linear, R ~ ReLU) embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`
    """

    def __init__(self, n_features: int, d_embeddings: List[int]) -> None:
        super().__init__(
            OrderedDict(
                [
                    (
                        'linear',
                        LinearEmbeddings(n_features, d_embeddings),
                    ),
                    ('activation', nn.ReLU()),
                ]
            )
        )


class _Periodic(nn.Module):
    """
    WARNING: the direct usage of this module is discouraged
    (do this only if you understand why this warning is here).
    """

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')

        super().__init__()
        self._sigma = sigma
        self.weight = torch.nn.parameter.Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 3:
            raise ValueError(
                f'The input must have at least three dimensions, however: {x.ndim=}'
            )

        x = 2 * math.pi * self.weight * x.permute(1, 2, 0)[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1).permute(2, 0, 1, 3)
        return x


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinearMultiLayer(nn.Module):
    """N *separate* linear layers for N feature embeddings with different embeddings sizes."""

    def __init__(self, n: int, in_features: List[int], out_features: List[int]) -> None:
        super().__init__()
        self.n_features = n
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.empty(in_features[i], out_features[i]))
                       for i in range(n)])
        self.bias = torch.nn.ParameterList([torch.nn.parameter.Parameter(torch.empty(1, out_features[i]))
                     for i in range(n)])
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = torch.max(torch.LongTensor(self.out_features)) ** -0.5
        _ = [nn.init.uniform_(self.weight[i], -d_in_rsqrt, d_in_rsqrt)
             for i in range(self.n_features)]
        _ = [nn.init.uniform_(self.bias[i], -d_in_rsqrt, d_in_rsqrt)
             for i in range(self.n_features)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        # assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        # x = (x[..., None, :] @ self.weight).squeeze(-2)
        # x = x + self.bias
        x = [x[i][..., None] * self.weight[i] for i in range(self.n_features)]
        x = [x[i] + self.bias[i][None] for i in range(self.n_features)]
        return torch.cat(x, dim=-1)


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings with same embedding size."""

    def __init__(self, n: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.empty(n, in_features, out_features))
        self.bias = torch.nn.parameter.Parameter(torch.empty(n, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assert x.ndim == 3
        # assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]
        x = (x[..., None, :] @ self.weight).squeeze(-2)
        x = x + self.bias
        return x


class PeriodicEmbeddings(nn.Module):
    """PL & PLR & PLR(lite) (P ~ Periodic, L ~ Linear, R ~ ReLU) embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(n_features, *, d_embedding)`
    """

    def __init__(
            self,
            n_features: int,
            d_embeddings: List[int],
            *,
            n_frequencies: int = 48,
            frequency_init_scale: float = 0.01,
            activation: bool = True,
            lite: bool = True,
    ) -> None:
        """
        Args:
            n_features: the number of features.
            d_embeddings: the embedding size for each feature.
            n_frequencies: the number of frequencies for each feature.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**,
                see the documentation for details.
            activation: if True, the embeddings is PLR, otherwise, it is PL.
            lite: if True, the last linear layer (the "L" step)
                is shared between all features. See the README.md document for details.
        """
        super().__init__()
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinearMultiLayer]
        self.n_features = n_features
        self.lite = lite
        if lite:
            # NOTE[DIFF]
            # The PLR(lite) variation was not covered in this paper about embeddings,
            # but it was used in the paper about the TabR model.
            if not activation:
                raise ValueError('lite=True is allowed only when activation=True')
            self.linear = torch.nn.ModuleList([nn.Linear(2 * n_frequencies, d_embeddings[i])
                                               for i in range(n_features)])
        else:
            self.linear = _NLinearMultiLayer(n_features, 2 * n_frequencies, d_embeddings)
        self.activation = nn.ReLU() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )
        x = self.periodic(x)
        if self.lite:
            x = torch.cat(
                [self.linear[i](x[i]) for i in range(self.n_features)],
                dim=-1)
        else:
            x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _check_bins(bins: List[torch.Tensor]) -> None:
    if bins is None:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, torch.Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        if len(feature_bins) == 2:
            warnings.warn(
                f'The {i}-th feature has just two bin edges, which means only one bin.'
                ' Strictly speaking, using a single bin for the'
                ' piecewise-linear encoding should not break anything,'
                ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
            )


def compute_bins(
        X: torch.Tensor,
        n_bins: int = 48,
        *,
        tree_kwargs: Optional[Dict[str, Any]] = None,
        y: Optional[torch.Tensor] = None,
        regression: Optional[bool] = None,
        verbose: bool = False,
) -> List[torch.Tensor]:
    """Compute bin edges for `PiecewiseLinearEmbeddings`.

    **Usage**

    Computing the quantile-based bins (Section 3.2.1 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> bins = compute_bins(X_train)

    Computing the tree-based bins (Section 3.2.2 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> y_train = torch.randn(len(X_train))
    >>> bins = compute_bins(
    ...     X_train,
    ...     y=y_train,
    ...     regression=True,
    ...     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    ... )

    Args:
        X: the training features.
        n_bins: the number of bins.
        tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor`
            (if ``regression`` is `True`) or `sklearn.tree.DecisionTreeClassifier`
            (if ``regression`` is `False`).
            NOTE: requires ``scikit-learn>=1.0,>2`` to be installed.
        y: the training labels (must be provided if ``tree`` is not None).
        regression: whether the labels are regression labels
            (must be provided if ``tree`` is not None).
        verbose: if True and ``tree_kwargs`` is not None, than ``tqdm``
            (must be installed) will report the progress while fitting trees.
    Returns:
        A list of bin edges for all features. For one feature:

        - the maximum possible number of bin edges is ``n_bins + 1``.
        - the minumum possible number of bin edges is ``1``.
    """
    if not isinstance(X, torch.Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')
    if not X.isfinite().all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if (X == X[0]).all(dim=0).any():
        raise ValueError(
            'All columns of X must have at least two distinct values.'
            ' However, X contains columns with just one distinct value.'
        )
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            'n_bins must be more than 1, but less than len(X), however:'
            f' {n_bins=}, {len(X)=}'
        )

    if tree_kwargs is None:
        if y is not None or regression is not None or verbose:
            raise ValueError(
                'If tree_kwargs is None, then y must be None, regression must be None'
                ' and verbose must be False'
            )

        # NOTE[DIFF]
        # The original implementation in the official paper repository has an
        # unintentional divergence from what is written in the paper.
        # This package implements the algorithm described in the paper,
        # and it is recommended for future work
        # (this may affect the optimal number of bins
        #  reported in the official repository).
        #
        # Additional notes:
        # - this is the line where the divergence happens:
        #   (the thing is that limiting the number of quantiles by the number of
        #   distinct values is NOT the same as removing identical quantiles
        #   after computing them)
        #   https://github.com/yandex-research/tabular-dl-num-embeddings/blob/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8/bin/train4.py#L612C30-L612C30
        # - for the tree-based bins, there is NO such divergence;
        bins = [
            q.unique()
            for q in torch.quantile(
                X, torch.linspace(0.0, 1.0, n_bins + 1).to(X), dim=0
            ).T
        ]
        _check_bins(bins)
        return bins
    else:
        try:
            import sklearn.tree as sklearn_tree
        except ImportError:
            sklearn_tree = None
        if sklearn_tree is None:
            raise RuntimeError(
                'The scikit-learn package is missing.'
                ' See README.md for installation instructions'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, however: {y.ndim=}')
        if len(y) != len(X):
            raise ValueError(
                f'len(y) must be equal to len(X), however: {len(y)=}, {len(X)=}'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if 'max_leaf_nodes' in tree_kwargs:
            raise ValueError(
                'tree_kwargs must not contain the key "max_leaf_nodes"'
                ' (it will be set to n_bins automatically).'
            )

        if verbose:
            if tqdm is None:
                raise ImportError('If verbose is True, tqdm must be installed')
            tqdm_ = tqdm
        else:
            tqdm_ = lambda x: x  # noqa: E731

        if X.device.type != 'cpu' or y.device.type != 'cpu':
            warnings.warn(
                'Computing tree-based bins involves the conversion of the input PyTorch'
                ' tensors to NumPy arrays. The provided PyTorch tensors are not'
                ' located on CPU, so the conversion has some overhead.',
                UserWarning,
            )
        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()
        bins = []
        for column in tqdm_(X_numpy.T):
            feature_bin_edges = [float(column.min()), float(column.max())]
            tree = (
                (
                    sklearn_tree.DecisionTreeRegressor
                    if regression
                    else sklearn_tree.DecisionTreeClassifier
                )(max_leaf_nodes=n_bins, **tree_kwargs)
                .fit(column.reshape(-1, 1), y_numpy)
                .tree_
            )
            for node_id in range(tree.node_count):
                # The following condition is True only for split nodes. Source:
                # https://scikit-learn.org/1.0/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    feature_bin_edges.append(float(tree.threshold[node_id]))
            bins.append(torch.as_tensor(feature_bin_edges).unique())
        _check_bins(bins)
        return [x.to(device=X.device, dtype=X.dtype) for x in bins]


class _PiecewiseLinearEncodingImpl(nn.Module):
    # NOTE
    # 1. DO NOT USE THIS CLASS DIRECTLY (ITS OUTPUT CONTAINS INFINITE VALUES).
    # 2. This implementation is not memory efficient for cases when there are many
    #    features with low number of bins and only few features
    #    with high number of bins. If this becomes a problem,
    #    just split features into groups and encode the groups separately.

    # The output of this module has the shape (*batch_dims, n_features, max_n_bins),
    # where max_n_bins = max(map(len, bins)) - 1.
    # If the i-th feature has the number of bins less than max_n_bins,
    # then its piecewise-linear representation is padded with inf as follows:
    # [x_1, x_2, ..., x_k, inf, ..., inf]
    # where:
    #            x_1 <= 1.0
    #     0.0 <= x_i <= 1.0 (for i in range(2, k))
    #     0.0 <= x_k
    #     k == len(bins[i]) - 1  (the number of bins for the i-th feature)

    # If all features have the same number of bins, then there are no infinite values.

    edges: torch.Tensor
    width: torch.Tensor
    mask: torch.Tensor

    def __init__(self, bins: List[torch.Tensor]) -> None:
        _check_bins(bins)

        super().__init__()
        # To stack bins to a tensor, all features must have the same number of bins.
        # To achieve that, for each feature with a less-than-max number of bins,
        # its bins are padded with additional phantom bins with infinite edges.
        max_n_edges = max(len(x) for x in bins)
        padding = torch.full(
            (max_n_edges,),
            math.inf,
            dtype=bins[0].dtype,
            device=bins[0].device,
        )
        edges = torch.row_stack([torch.cat([x, padding])[:max_n_edges] for x in bins])

        # The rightmost edge is needed only to compute the width of the rightmost bin.
        self.register_buffer('edges', edges[:, :-1])
        self.register_buffer('width', edges.diff())
        # mask is false for the padding values.
        self.register_buffer(
            'mask',
            torch.row_stack(
                [
                    torch.cat(
                        [
                            torch.ones(len(x) - 1, dtype=torch.bool, device=x.device),
                            torch.zeros(
                                max_n_edges - 1, dtype=torch.bool, device=x.device
                            ),
                        ]
                    )[: max_n_edges - 1]
                    for x in bins
                ]
            ),
        )
        self._bin_counts = tuple(len(x) - 1 for x in bins)
        self._same_bin_count = all(x == self._bin_counts[0] for x in self._bin_counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(
                f'The input must have at least two dimensions, however: {x.ndim=}'
            )

        # See Equation 1 in the paper.
        x = (x[..., None] - self.edges) / self.width

        # If the number of bins is greater than 1, then, the following rules must
        # be applied to a piecewise-linear encoding of a single feature:
        # - the leftmost value can be negative, but not greater than 1.0.
        # - the rightmost value can be greater than 1.0, but not negative.
        # - the intermediate values must stay within [0.0, 1.0].
        n_bins = x.shape[-1]
        if n_bins > 1:
            if self._same_bin_count:
                x = torch.cat(
                    [
                        x[..., :1].clamp_max(1.0),
                        *([] if n_bins == 2 else [x[..., 1:-1].clamp(0.0, 1.0)]),
                        x[..., -1:].clamp_min(0.0),
                    ],
                    dim=-1,
                )
            else:
                # In this case, the rightmost values for all features are located
                # in different columns.
                x = torch.stack(
                    [
                        x[..., i, :]
                        if count == 1
                        else torch.cat(
                            [
                                x[..., i, :1].clamp_max(1.0),
                                *(
                                    []
                                    if n_bins == 2
                                    else [x[..., i, 1: count - 1].clamp(0.0, 1.0)]
                                ),
                                x[..., i, count - 1: count].clamp_min(0.0),
                                x[..., i, count:],
                            ],
                            dim=-1,
                        )
                        for i, count in enumerate(self._bin_counts)
                    ],
                    dim=-2,
                )
        return x


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, n_features, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.
    """

    def __init__(self, bins: List[torch.Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        _check_bins(bins)

        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.impl(x)
        return x.flatten(-2) if self.impl._same_bin_count else x[:, self.impl.mask]


class PiecewiseLinearEmbeddings(nn.Module):
    """Piecewise-linear embeddings.

    **Shape**

    - Input: ``(n_features, bs, seq_lens)``
    - Output: ``(bs, seq_lens, d_embedding)``
    """

    def __init__(
            self, bins: List[torch.Tensor], d_embedding: int, *, activation: bool
    ) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
            d_embedding: the embedding size.
            activation: if False, the embedding becomes what is called "Q-L"/"T-L"
                in Table 2 in the paper (depending on how bins were computed).
                Otherwise, the embedding is "Q-LR"/"T-LR".
        """
        if d_embedding <= 0:
            raise ValueError(
                f'd_embedding must be a positive integer, however: {d_embedding=}'
            )
        _check_bins(bins)

        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.linear = _NLinear(len(bins), max(self.impl._bin_counts), d_embedding)
        self.activation = nn.ReLU() if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # as-is input size: [n_features, bs, seq_len]
        # needed input size: [*, bs, n_features]
        x = x.permute(2, 1, 0)

        x = self.impl(x)
        if not self.impl._same_bin_count:
            # Replace infinite values with zeros.
            x = torch.where(
                self.impl.mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device)
            )
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x.permute(1, 0, 2, 3).flatten(2, 3)


EMBEDDING_TYPES = {
    "linear": LinearEmbeddings,
    "linear_relu": LinearReLUEmbeddings,
    "categorical": CategoricalEmbeddings,
    "periodic": PeriodicEmbeddings,
    "piecewise": PiecewiseLinearEmbeddings
}
