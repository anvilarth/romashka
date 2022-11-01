import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self,
                transactions_cat_features, 
                embedding_projections, 
                product_col_name='product',
                emb_mult=1,
                ):

        super().__init__()

        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature], emb_mult=emb_mult) 
                                                        for feature in transactions_cat_features])
                
        self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None, emb_mult=emb_mult)
        self.inp_size = sum([embedding_projections[x][1] * emb_mult for x in transactions_cat_features]) + embedding_projections[product_col_name][1]

    def forward(self, transactions_cat_features, product_feature):
        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)

        batch, seq_len, dim = concated_embeddings.shape

        product_embed = self._product_embedding(product_feature).unsqueeze(1).repeat(1, seq_len, 1)

        embedding = torch.cat([concated_embeddings, product_embed], dim=-1)

        return embedding
    
    def get_embedding_size(self):
        return self.inp_size

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0, emb_mult=1):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size*emb_mult, padding_idx=padding_idx)