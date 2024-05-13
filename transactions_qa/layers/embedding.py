import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any

from romashka.transactions_qa.layers import (EMBEDDING_TYPES,
                                             LinearEmbeddings,
                                             LinearReLUEmbeddings,
                                             CategoricalEmbeddings,
                                             PeriodicEmbeddings,
                                             PiecewiseLinearEmbeddings)
from romashka.transactions_qa.utils import get_mantissa_number, get_exponent_number


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 cat_embedding_projections: Dict[str, Tuple[int, int]],
                 cat_features: List[str],
                 cat_bias: Optional[bool] = True,
                 num_embedding_projections: Optional[Dict[str, Tuple[int, int]]] = None,
                 num_features: Optional[List[str]] = None,
                 use_real_num_features: Optional[bool] = False,
                 num_embeddings_type: Optional[str] = 'linear',
                 num_embeddings_kwargs: Optional[Dict[str, Any]] = {},
                 meta_embedding_projections: Optional[Dict[str, Tuple[int, int]]] = None,
                 meta_features: Optional[List[str]] = None,
                 time_embedding: Optional[bool] = None,
                 dropout: Optional[float] = 0.0,
                 ):

        super().__init__()
        self.cat_embedding = CatEmbeddingsCustom(cat_embedding_projections, cat_features, bias=cat_bias)
        self.dropout = nn.Dropout(dropout)

        self.num_embedding = None
        self.meta_embedding = None
        self.time_embedding = None

        self.use_real_num_features = use_real_num_features
        # if self.use_real_num_features and ((num_embeddings_type == 'linear')
        #                                    or (num_embeddings_type == 'linear_relu')):
        #     raise AttributeError(f"Using real-valued numeric features is impossible with 'linear'/'linear_relu' "
        #                          f"embeddings type!")

        self.num_embeddings_type = num_embeddings_type
        self.num_embeddings_kwargs = num_embeddings_kwargs
        if num_embedding_projections is not None and num_features is not None:
            self.num_embedding = NumericalEmbedding(num_embedding_projections,
                                                    num_features,
                                                    embeddings_type=num_embeddings_type,
                                                    embeddings_kwargs=num_embeddings_kwargs)

        if meta_embedding_projections is not None and meta_features is not None:
            self.meta_embedding = CatEmbedding(meta_embedding_projections, meta_features)

        if time_embedding is not None:
            self.time_embedding = DateEmbedding()

    def get_embedding_size(self):
        res = self.cat_embedding.get_embedding_size()
        if self.num_embedding is not None:
            res += self.num_embedding.get_embedding_size()

        if self.meta_embedding is not None:
            res += self.meta_embedding.get_embedding_size()

        if self.time_embedding is not None:
            res += self.time_embedding.get_embedding_size()
        return res

    def forward(self, batch, mask=None):
        batch_size = batch['mask'].shape[0]
        seq_len = batch['cat_features'][0].shape[1]

        cat_features, num_features = batch['cat_features'], batch['num_features']

        if "time" in batch:
            time_features = batch.get('time')
        else:
            time_features = None

        if "real_num_features" in batch:
            real_num_features = batch.get('real_num_features')
        else:
            real_num_features = None

        if "meta_features" in batch:
            meta_features = batch.get('meta_features')
        else:
            meta_features = None

        embeddings = self.cat_embedding(cat_features)

        if self.time_embedding is not None:
            time_embeddings = self.time_embedding(time_features)
            embeddings = torch.cat([embeddings, time_embeddings], dim=-1)

        if self.num_embedding is not None:
            num_embeddings = self.num_embedding(real_num_features) if self.use_real_num_features \
                else self.num_embedding(num_features)
            embeddings = torch.cat([embeddings, num_embeddings], dim=-1)

        if (self.meta_embedding is not None) and (meta_features is not None):
            meta_embeddings = self.meta_embedding(meta_features).unsqueeze(1)
            meta_embeddings = meta_embeddings.repeat(1, seq_len, 1)
            embeddings = torch.cat([embeddings, meta_embeddings], dim=-1)

        embeddings = self.dropout(embeddings)
        return embeddings


class NumericalEmbedding(nn.Module):
    def __init__(self, embedding_projections: Dict[str, Tuple[int, int]],
                 numeric_features: List[str],
                 embeddings_type: Optional[str] = 'linear',
                 embeddings_kwargs: Optional[Dict[str, Any]] = {}):
        super().__init__()
        self.embeddings_type = embeddings_type
        self.embeddings_additional_kwargs = embeddings_kwargs
        self.numeric_features = numeric_features
        self.embedding_projections = embedding_projections  # as tuples (input_size, output_size)
        self.num_embedding = self._create_embedding_projection()
        self.output_size = sum([embedding_projections[feature][1] for feature in self.numeric_features])

    def forward(self, num_features):
        return self.num_embedding(num_features)

    def get_embedding_size(self):
        return self.output_size

    def _create_embedding_projection(self):
        if not (self.embeddings_type in EMBEDDING_TYPES):
            raise AttributeError(f"Unknown embedding type: {self.embeddings_type}!"
                                 f"\nChoose one from the following: {list(EMBEDDING_TYPES.keys())}")
        # TODO: make automatic creation from dict instead of if/else
        embedding_dims = [self.embedding_projections[feature][1] for feature in self.numeric_features]

        if self.embeddings_type == 'linear':
            kwargs = {
                "n_features": len(self.numeric_features),
                "d_embeddings": embedding_dims
            }
            kwargs.update(self.embeddings_additional_kwargs)
            return LinearEmbeddings(**kwargs)
        elif self.embeddings_type == 'linear_relu':
            kwargs = {
                "n_features": len(self.numeric_features),
                "d_embeddings": embedding_dims
            }
            kwargs.update(self.embeddings_additional_kwargs)
            return LinearReLUEmbeddings(**kwargs)
        elif self.embeddings_type == 'periodic':
            kwargs = {
                "n_features": len(self.numeric_features),
                "d_embeddings": embedding_dims
            }
            kwargs.update(self.embeddings_additional_kwargs)
            return PeriodicEmbeddings(**kwargs)
        else:
            raise AttributeError(f"Currently this type of embeddings is not supported!")


class CatEmbeddingsCustom(nn.Module):
    """
    Embeddings for categorical features.
    """

    def __init__(self,
                 embedding_projections: Dict[str, Tuple[int, int]],
                 cat_features: List[str],
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
        self.bias = bias
        self.cat_features = cat_features
        self.embedding_projections = embedding_projections
        self.cat_embedding = self._create_embedding_projection()
        self.output_size = sum([embedding_projections[feature][1] for feature in cat_features])

    def get_embedding_size(self):
        return self.output_size

    def forward(self, cat_features):
        return self.cat_embedding(cat_features)

    def _create_embedding_projection(self):
        cat_cardinalities = []
        cat_embedding_dims = []
        for i, cat_feature_name in enumerate(self.cat_features):
            cat_cardinalities.append(self.embedding_projections[cat_feature_name][0])
            cat_embedding_dims.append(self.embedding_projections[cat_feature_name][1])

        return CategoricalEmbeddings(cat_cardinalities,
                                     d_embeddings=cat_embedding_dims,
                                     add_missing=self.add_missing,
                                     bias=self.bias)


class CatEmbedding(nn.Module):
    def __init__(self, embedding_projections, use_features):
        super().__init__()
        self.cat_embedding = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature])
                                            for feature in use_features])

        self.output_size = sum([embedding_projections[feature][1] for feature in use_features])

    def forward(self, cat_features):
        cat_embeddings = [embedding(cat_features[i]) for i, embedding in enumerate(self.cat_embedding)]
        return torch.cat(cat_embeddings, dim=-1)

    def get_embedding_size(self):
        return self.output_size

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx)


class DateEmbedding(nn.Module):
    def __init__(self, k=32, act="sin"):
        super().__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.fc1 = nn.Linear(6, k1)

        self.fc2 = nn.Linear(6, k2)

        self.output_size = k

        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], -1)
        return out

    def get_embedding_size(self):
        return self.output_size


class PiecewiseLinearEmbedding(nn.Module):
    def __init__(self, embedding_dim, buckets):
        super().__init__()
        self.buckets = buckets
        self.num_buckets = len(buckets) + 1

        self.layer = nn.Linear(self.num_buckets, embedding_dim)
        self.matrix = torch.tril(torch.ones(self.num_buckets, self.num_buckets))

        self.bucket_sizes = self.buckets.diff()

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device

        indices = torch.bucketize(x, self.buckets, right=False)
        new_indices = torch.clamp(indices - 1, 0, self.num_buckets - 2)
        original_matrix = self.matrix[indices]

        buck_indices = torch.clamp(new_indices, 0, self.num_buckets - 3)
        size_down = self.bucket_sizes[buck_indices]

        adding = ((x - self.buckets[new_indices]) / size_down)

        mask_borders = ((indices != 0) & (indices != self.num_buckets - 1))
        adding[~mask_borders] = 1.0
        mask_matrix = original_matrix.sum(1).long() - 1

        original_matrix[torch.arange(batch_size), mask_matrix] *= adding

        return self.layer(original_matrix)


class NumEmbedding(nn.Module):
    def __init__(self, embedding_dim, buckets) -> None:
        super().__init__()

        self.mantissa_embedding = PiecewiseLinearEmbedding(embedding_dim // 2, buckets)
        self.exponent_embedding = nn.Embedding(17, embedding_dim // 2)

    def forward(self, x):
        exponent = get_exponent_number(x) + 8
        mantissa = get_mantissa_number(x)

        embedding_exponent = self.exponent_embedding(exponent)
        embedding_mantissa = self.mantissa_embedding(mantissa)

        return torch.cat([embedding_mantissa, embedding_exponent], dim=-1)


def eq_fn(x):
    return x


def cos_fn(freq, x):
    return torch.cos(x * freq)


def sin_fn(freq, x):
    return torch.sin(x * freq)


class PositionalEncoder(nn.Module):

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False
    ):

        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [eq_fn]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            # self.embed_fns.append(partial(sin_fn(freq)))
            # self.embed_fns.append(partial(cos_fn(freq)))

            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)
