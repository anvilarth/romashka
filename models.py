import torch
import sys
import torch.nn as nn

import torch.nn.functional as F
from time import time
from transformers import GPT2Model, GPT2Config, BertConfig, BertModel

from embedding import EmbeddingLayer
from augmentations import mixup_data

from encoder import BERT, Informer, InformerConfigs
from head import LinearHead, RNNClassificationHead, NSPHead, MLPHead, TransformerHead, IdentityHead
from tools import LambdaLayer

sys.path.append('./perceiver-pytorch')

from perceiver_pytorch import Perceiver


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
                 encoder_type='bert',
                 add_token='before',
                 num_layers=6, 
                 dropout=0.1,
                 cutmix=False,
                 mixup=False,
                 emb_mult=1,
                 rel_pos_embs=False,
                 alpha=1.0):

        super().__init__()
        self.embedding = EmbeddingLayer(cat_embedding_projections, 
                                        cat_features, 
                                        num_embedding_projections, 
                                        num_features,
                                        meta_embedding_projections,
                                        meta_features,
                                        time_embedding,
                                        dropout=dropout)
        
        inp_size = self.embedding.get_embedding_size()
        
        self.add_token = add_token
        self.head_type = head_type
        self.encoder_type = encoder_type
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_size) / inp_size)

        if encoder_type == 'bert':
            self.encoder = BERT(inp_size, heads=2*emb_mult, num_layers=num_layers, dropout=dropout, layer_norm_eps=1e-7, rel_pos_embs=rel_pos_embs)
        
        elif encoder_type == 'informer':
            configs = InformerConfigs(d_model=inp_size * emb_mult,
                        dropout=dropout,
                        n_heads=2*emb_mult,
                        num_layers=num_layers)
            self.encoder = Informer(configs)
        elif encoder_type == 'rnn':
            self.encoder = nn.GRU(inp_size, inp_size, batch_first=True)
        elif encoder_type == 'rnn2':
            self.encoder = nn.GRU(inp_size, inp_size, num_layers=2, batch_first=True)
            
        elif encoder_type == 'gpt':
            configuration = GPT2Config(vocab_size=1, n_positions=2000, 
                                       n_embd=inp_size, n_layer=num_layers, 
                                       n_head=2, resid_pdrop=dropout,
                                       embd_pdrop=dropout, attn_pdrop=dropout)
            self.encoder = GPT2Model(configuration)
            # removing positional encoding
            self.encoder.wpe = LambdaLayer(lambda x: 0)
        
        elif encoder_type == 'perceiver':
            self.encoder = Perceiver(input_channels=inp_size,
                                     depth = 6,                   
                                     num_latents = 16,           # number of latents, or induced set points, or centroids. different papers giving it different names
                                     latent_dim = 100,            # latent dimension
                                     cross_heads = 1,             # number of heads for cross attention. paper said 1
                                     latent_heads = 4,            # number of heads for latent self attention, 8
                                     cross_dim_head = 32,         # number of dimensions per cross attention head
                                     latent_dim_head = 32,        # number of dimensions per latent self attention head
                                     num_classes = 1,          # output number of classes
                                     attn_dropout = dropout,
                                     ff_dropout = dropout,
                                     weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
                                     self_per_cross_attn = 2      # number of self attention blocks per cross attention
                                )
        
        else:
            raise NotImplementedError

        if head_type == 'linear':
            self.head = LinearHead(inp_size)
        elif head_type == 'rnn':
            self.head = RNNClassificationHead(inp_size)
        elif head_type == 'mlp':
            self.head = MLPHead(inp_size)
        elif head_type == 'transformer':
            self.head = TransformerHead(inp_size)
        elif head_type == 'id':
            self.head = IdentityHead()
        elif head_type == 'next':
            self.head = NSPHead(inp_size, cat_embedding_projections, num_embedding_projections)
        else:
            raise NotImplementedError

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha
    
    def forward(self, batch=None, embeds=None):
        mask = batch['mask']
        batch_size = mask.shape[0]
        
        if embeds is None:
            embedding = self.embedding(batch)
        else:
            embedding = embeds
        
        if self.head_type != 'next':
            if self.add_token == 'before':
                cls_token = self.cls_token.repeat(batch_size, 1, 1)
                embedding = torch.cat([embedding, cls_token], dim=1)

                cls_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
                mask = torch.cat([mask, cls_token_mask], dim=1)

#         if self.cutmix and self.training:
#             transactions_cat_features = add_noise(transactions_cat_features, mask1)

        # if self.mixup and self.training:
        #     embedding = mixup_data(embedding, self.alpha, mask=mask.squeeze(2))
        
        if self.encoder_type == 'gpt':
            x = self.encoder(inputs_embeds=embedding, attention_mask=mask).last_hidden_state
        elif 'rnn' in self.encoder_type:
            x, _ = self.encoder(embedding)
        elif self.encoder_type == 'bert':
            mask = mask.unsqueeze(1).unsqueeze(2)
            x = self.encoder(embedding, mask)
        else:
            x = self.encoder(embedding, mask)
            
        if self.add_token == 'after':
            cls_token = self.cls_token.repeat(batch_size, 1, 1)
            x = torch.cat([x, cls_token], dim=1)
            
            cls_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
            mask = torch.cat([mask, cls_token_mask], dim=1)
                
        logit = self.head(x, mask)
        
        return logit
    
class LastPrediction(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, batch):
        pass
        
    
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
