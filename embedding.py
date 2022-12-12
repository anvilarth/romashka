import torch
import torch.nn as nn
from copy import deepcopy

class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 cat_embedding_projections,
                 cat_features,
                 num_embedding_projections=None,
                 num_features=None,
                 meta_embedding_projections=None,
                 meta_features=None,
                 time_embedding=None,
                 dropout=0.0,
                ):
        
        super().__init__()
        self.cat_embedding = CatEmbedding(cat_embedding_projections, cat_features)
        self.dropout = nn.Dropout(dropout)
        
        self.num_embedding = None
        self.meta_embedding = None
        self.time_embedding = None
        
        if num_embedding_projections is not None and num_features is not None:
            self.num_embedding = NumericalEmbedding(num_embedding_projections, num_features)
        
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
        
    def forward(self, batch):
        cat_features, num_features = batch['cat_features'], batch['num_features']
        time_features, meta_features = batch.get('time'), batch['meta_features']
        
        
        seq_len = batch['cat_features'][0].shape[1]
        embeddings = self.cat_embedding(cat_features)
        
        if self.time_embedding is not None:
            time_embeddings = self.time_embedding(time_features)
            embeddings = torch.cat([embeddings, time_embeddings], dim=-1)
        
        if self.num_embedding is not None:
            num_embeddings = self.num_embedding(num_features)
            embeddings = torch.cat([embeddings, num_embeddings], dim=-1)
        
        if self.meta_embedding is not None:
            meta_embeddings = self.meta_embedding(meta_features).unsqueeze(1)
            meta_embeddings = meta_embeddings.repeat(1, seq_len, 1)
            embeddings = torch.cat([embeddings, meta_embeddings], dim=-1)
        
        embeddings = self.dropout(embeddings)
        return embeddings
        
class NumericalEmbedding(nn.Module):
    def __init__(self, embedding_projections, use_features):
        super().__init__()
        self.num_embedding = nn.ModuleList([self._create_embedding_projection(embedding_projections[feature][1]) 
                                                for feature in use_features])
        
        self.output_size = sum([embedding_projections[feature][1] for feature in use_features])
        
    def forward(self,  num_features):
        num_embeddings = [embedding(num_features[i][..., None]) for i, embedding in enumerate(self.num_embedding)]
        return torch.cat(num_embeddings, dim=-1)
    
    def get_embedding_size(self):
        return self.output_size
    
    @classmethod
    def _create_embedding_projection(cls, embed_size):
        return nn.Linear(1, embed_size)
    
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
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size, padding_idx=padding_idx)

    
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
    
    
# class MLP(nn.Module):
#     def __init__(self, size):
#         pass
    
#     def forward(self, x):
#         pass
    

# class EmbeddingLayer(nn.Module):
#     def __init__(self,
#                 transactions_cat_features,
#                 embedding_projections, 
#                 product_col_name='product',
#                 emb_mult=1,
#                 num_buckets=None,
#                 ):

#         super().__init__()

#         self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], emb_mult=emb_mult) 
#                                                         for feature in transactions_cat_features])
#         self.buckets = None
#         if num_buckets is not None:
#             self.buckets = deepcopy(num_buckets)
#             for key in self.buckets:
#                 self.buckets[key] = torch.Tensor(self.buckets[key]).cuda()
                
#         self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None, emb_mult=emb_mult)
#         self.output_size = sum([embedding_projections[x][1] * emb_mult for x in transactions_cat_features]) + embedding_projections[product_col_name][1]

#     def forward(self, batch):
#         transactions_cat_features, product_feature = deepcopy(batch['transactions_features']), batch['product']

#         if self.buckets is not None:
#             for j, key in enumerate(self.buckets):
#                 transactions_cat_features[-3+j] = torch.bucketize(transactions_cat_features[-3+j], self.buckets[key])

#         embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
#         concated_embeddings = torch.cat(embeddings, dim=-1)

#         batch, seq_len, dim = concated_embeddings.shape

#         product_embed = self._product_embedding(product_feature).unsqueeze(1).repeat(1, seq_len, 1)

#         embedding = torch.cat([concated_embeddings, product_embed], dim=-1)

#         return embedding
    
#     def get_embedding_size(self):
#         return self.output_size

#     @classmethod
#     def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0, emb_mult=1):
#         add_missing = 1 if add_missing else 0
#         return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size*emb_mult, padding_idx=padding_idx)
    
    
    
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