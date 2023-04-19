import torch
import sys
import torch.nn as nn

import timm
import torch.nn.functional as F
from time import time

import copy

from .embedding import EmbeddingLayer

from .head import TransactionHead
from .encoder import TransactionEncoder
from .connectors import TransactionConnector


# sys.path.append('/home/jovyan/romashka/adapter_transformers')
# from adapter_transformers import AutoAdapterModel

# sys.path.append('/home/jovyan/romashka/perceiver-pytorch')
# from perceiver_pytorch import Perceiver

# from adapter_transformers import GPT2Model as GPT2ModelAdapter, BertModel as BertModelAdapter




class WrapVisionModel(nn.Module):
    def __init__(self, process, encoder):
        super().__init__()
        self.process = process
        self.encoder = encoder

    def forward(self, x, attention_mask=None):
        x = self.process(x, mask=attention_mask)
        return self.encoder(x)


class TransactionsModel(nn.Module):
    def __init__(self,
                 cat_embedding_projections,
                 cat_features,
                 num_embedding_projections=None,
                 num_features=None,
                 meta_embedding_projections=None,
                 meta_features=None,
                 time_embedding=None,
                 head_type='linear',
                 encoder_type='gpt2/base',
                 num_layers=1,
                 embedding_dropout=0.0,
                 cutmix=False,
                 mixup=False,
                 emb_mult=1,
                 rel_pos_embs=False,
                 pretrained=False,
                 adapters=False,
                 hidden_size=None,
                 alpha=1.0,
                 *args,
                 **kwargs):

        super().__init__()
        self.embedding = EmbeddingLayer(cat_embedding_projections,
                                        cat_features,
                                        num_embedding_projections,
                                        num_features,
                                        meta_embedding_projections,
                                        meta_features,
                                        time_embedding,
                                        dropout=embedding_dropout)
        inp_size = self.embedding.get_embedding_size()

        self.encoder_model = TransactionEncoder(encoder_type,
                                                inp_size,
                                                hidden_size,
                                                pretrained,
                                                num_layers,
        )

        output_size = self.encoder_model.get_output_size()
        connector_type = self.encoder_model.get_connector_type()

        self.cls_token = nn.Parameter(torch.randn(1, 1, output_size))
        self.connector = TransactionConnector(inp_size, output_size, connector_type)

        self.head_type = head_type
        self.head = TransactionHead(head_type,  output_size, cat_embedding_projections, num_embedding_projections)

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha

    def get_base_embeddings(self, batch):
        mask = batch['mask']
        
        embedding = self.embedding(batch)
        embedding = self.connector(embedding, attention_mask=mask)

        return embedding

    def get_embs(self, batch, embeds=None):
        mask = batch['mask']
        batch_size = mask.shape[0]

        if embeds is None:
            embedding = self.get_base_embeddings(batch)
        else:
            embedding = embeds

        if self.head_type != 'next' and self.head_type != 'next_time' and self.head_type!= 'last_output':
            cls_token = self.cls_token.repeat(batch_size, 1, 1)
            embedding = torch.cat([embedding, cls_token], dim=1)

            cls_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
            mask = torch.cat([mask, cls_token_mask], dim=1)

        x, mask = self.encoder_model(embedding, mask)

        return x, mask

    def forward(self, batch=None, embeds=None):
        x, mask = self.get_embs(batch, embeds)

        logit = self.head(x, mask)

        return logit

# class NextTransactionModel(nn.Module):
#     def __init__(self, 
#                  cat_embedding_projections,
#                  cat_features,
#                  num_embedding_projections,
#                  num_features,  
#                  num_layers=4,
#                  emb_mult=1,
#                  num_buckets=None,
#                  model_type='rnn',
#                 ):

#         super().__init__()
#         self.embedding = EmbeddingLayer(cat_embedding_projections, 
#                                         cat_features, 
#                                         num_embedding_projections, 
#                                         num_features)

#         inp_size = self.embedding.get_embedding_size()

#         self.model_type = model_type
#         if model_type == 'gpt':
#             configuration = GPT2Config(vocab_size=1, n_positions=750, n_embd=inp_size, n_layer=num_layers, n_head=2)
#             self.model = GPT2Model(configuration)
#         else:
#             self.model = nn.GRU(inp_size, inp_size)

#         self.head = NSPHead(inp_size, embedding_projections)

#     def forward(self, batch):

#         mask = batch['mask']
#         batch_size = mask.shape[0]

#         embedding = self.embedding(batch)

#         cls_token = self.cls_token.repeat(batch_size, 1, 1)
#         embedding = torch.cat([cls_token, embedding], dim=1)

#         first_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
#         mask = torch.cat([mask, first_token_mask], dim=1)
#         mask = mask.unsqueeze(1).unsqueeze(2)

#         if self.model_type == 'gpt':
#             out = self.model(inputs_embeds=embedding, attention_mask=mask).last_hidden_state
#         else:
#             out, _ = self.model(embedding)

#         logits = self.head(out)

#         return logits

#     def modify_numerical_head(self):
#         for i in range(3):
#             self.head.heads[-3 + i] = nn.Linear(self.head.heads[-3 + i].in_features, 1, device=self.head.heads[-3 + i].weight.device)


# class TransactionsRnn(nn.Module):
#     def __init__(self, transactions_cat_features, embedding_projections, product_col_name='product', rnn_units=128, top_classifier_units=32, mixup=False):
#         super().__init__()
#         self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
#                                                           for feature in transactions_cat_features])

#         self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None)

#         inp_size = sum([embedding_projections[x][1] for x in transactions_cat_features])
#         self._gru = nn.GRU(input_size=inp_size, hidden_size=rnn_units, batch_first=True, bidirectional=False)

#         self._hidden_size = rnn_units

#         self._top_classifier = nn.Linear(in_features=rnn_units+embedding_projections[product_col_name][1], 
#                                          out_features=top_classifier_units)
#         self._intermediate_activation = nn.ReLU()

#         self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
#         self.mixup = mixup

#     def forward(self, batch):
#         transactions_cat_features, product_feature = batch['transactions_features'], batch['product']
#         batch_size = product_feature.shape[0]

#         if self.mixup:
#             transactions_cat_features = add_noise(transactions_cat_features)

#         embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
#         concated_embeddings = torch.cat(embeddings, dim=-1)

#         _, last_hidden = self._gru(concated_embeddings)

#         last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, self._hidden_size))
#         product_embed = self._product_embedding(product_feature)

#         intermediate_concat = torch.cat([last_hidden, product_embed], dim=-1)

#         classification_hidden = self._top_classifier(intermediate_concat)
#         activation = self._intermediate_activation(classification_hidden)

#         logit = self._head(activation)

#         return logit

#     @classmethod
#     def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0, emb_mult=1):
#         add_missing = 1 if add_missing else 0
#         return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size*emb_mult, padding_idx=padding_idx)


# 


# class MaskedModel(nn.Module):
#     def __init__(self, 
#                  transactions_cat_features, 
#                  embedding_projections, 
#                  product_col_name='product',
#                  num_layers=4,
#                  emb_mult=1,
#                 ):

#         super().__init__()
#         self.embedding = EmbeddingLayer(transactions_cat_features, embedding_projections, product_col_name, emb_mult)
#         inp_size = self.embedding.get_embedding_size()

#         configuration = BertConfig(vocab_size=1, n_positions=760, n_embd=inp_size, n_layer=num_layers, n_head=2)
#         self.model = BertModel(configuration)

#         self.head = NSPHead(inp_size, embedding_projections)

#     def forward(self, transactions_cat_features, product_feature):
#         batch_size = product_feature.shape[0]
#         mask1 = transactions_cat_features[-6] != 0

#         mask = mask1.unsqueeze(1).unsqueeze(2)
#         embedding = self.embedding(transactions_cat_features, product_feature)

#         out = self.model(inputs_embeds=embedding, attention_mask=mask)
#         logits = self.head(out.last_hidden_state)

#         return logits
