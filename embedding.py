import torch
import torch.nn as nn
from copy import deepcopy

class EmbeddingLayer(nn.Module):
    def __init__(self,
                transactions_cat_features,
                embedding_projections, 
                product_col_name='product',
                emb_mult=1,
                num_buckets=None,
                ):

        super().__init__()

        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], emb_mult=emb_mult) 
                                                        for feature in transactions_cat_features])
        self.buckets = None
        if num_buckets is not None:
            self.buckets = deepcopy(num_buckets)
            for key in self.buckets:
                self.buckets[key] = torch.Tensor(self.buckets[key]).cuda()
                
        self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None, emb_mult=emb_mult)
        self.output_size = sum([embedding_projections[x][1] * emb_mult for x in transactions_cat_features]) + embedding_projections[product_col_name][1]

    def forward(self, batch):
        transactions_cat_features, product_feature = deepcopy(batch['transactions_features']), batch['product']

        if self.buckets is not None:
            for j, key in enumerate(self.buckets):
                transactions_cat_features[-3+j] = torch.bucketize(transactions_cat_features[-3+j], self.buckets[key])

        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)

        batch, seq_len, dim = concated_embeddings.shape

        product_embed = self._product_embedding(product_feature).unsqueeze(1).repeat(1, seq_len, 1)

        embedding = torch.cat([concated_embeddings, product_embed], dim=-1)

        return embedding
    
    def get_embedding_size(self):
        return self.output_size

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0, emb_mult=1):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size*emb_mult, padding_idx=padding_idx)



class PiecewiseLinearEmbedding(nn.Module):
    def __init__(self, num_feature, vector_dim):

        super().__init__()
        self.linear = nn.Linear(num_feature+1, vector_dim)
        self.transform_matrix = torch.tril(torch.ones(num_feature+1, num_feature+1))

    def forward(self, x):
        embeddings = self.transform_matrix[x]
        return self.linear(embeddings)

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
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
    def forward(self, x):
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)