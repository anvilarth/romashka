import torch
import sys
import torch.nn as nn

import torch.nn.functional as F
from time import time
from transformers import GPT2Model, GPT2Config, BertConfig, BertModel

from embedding import EmbeddingLayer
from augmentations import add_noise, mixup_data

from encoder import BERT, Informer, InformerConfigs
from head import ClassificationHead, RNNClassificationHead, NSPHead


class TransactionsModel(nn.Module):
    def __init__(self, 
                 transactions_cat_features, 
                 embedding_projections, 
                 product_col_name='product', 
                 head_type='linear',
                 encoder_type='bert',
                 num_layers=6, 
                 top_classifier_units=32,
                 dropout=0.0,
                 cutmix=False,
                 mixup=False,
                 emb_mult=1,
                 rel_pos_embs=False,
                 alpha=1.0):

        super().__init__()
        self.embedding = EmbeddingLayer(transactions_cat_features, embedding_projections, product_col_name, emb_mult)
        inp_size = self.embedding.get_embedding_size()

        self.head_type = head_type
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_size) / inp_size)

        if encoder_type == 'bert':
            self.encoder = BERT(inp_size, heads=2*emb_mult, num_layers=num_layers, dropout=dropout, layer_norm_eps=1e-7, rel_pos_embs=rel_pos_embs)
        
        elif encoder_type == 'informer':
            configs = InformerConfigs(d_model=inp_size * emb_mult,
                        dropout=dropout,
                        n_heads=2*emb_mult,
                        num_layers=num_layers)
            self.encoder = Informer(configs)
        
        else:
            raise NotImplementedError

        if head_type == 'linear':
            self.head = ClassificationHead(inp_size, top_classifier_units)
        elif head_type == 'rnn':
            self.head = RNNClassificationHead(inp_size, top_classifier_units)
        elif head_type == 'id':
            self.head = nn.Identity()
        else:
            raise NotImplementedError

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha
    
    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        mask1 = transactions_cat_features[-6] != 0
            
        first_token_mask = torch.ones(batch_size, 1, dtype=bool, device=transactions_cat_features[0].device)
        mask = torch.cat([first_token_mask, mask1], dim=1)
        mask = mask.unsqueeze(1).unsqueeze(2)
        
        if self.cutmix and self.training:
            transactions_cat_features = add_noise(transactions_cat_features, mask1)
        
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        embedding = self.embedding(transactions_cat_features, product_feature)
        embedding = torch.cat([cls_token, embedding], dim=1)

        
        if self.mixup and self.training:
            embedding = mixup_data(embedding, self.alpha, mask=mask.squeeze(2))
            
        x = self.encoder(embedding, mask)
        if self.head_type == 'linear':
        
            x = x[:, 0]
        logit = self.head(x)
        return logit
    
    
class TransactionsRnn(nn.Module):
    def __init__(self, transactions_cat_features, embedding_projections, product_col_name='product', rnn_units=128, top_classifier_units=32, mixup=False):
        super().__init__()
        self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
                                                          for feature in transactions_cat_features])
                
        self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None)
        
        inp_size = sum([embedding_projections[x][1] for x in transactions_cat_features])
        self._gru = nn.GRU(input_size=inp_size, hidden_size=rnn_units, batch_first=True, bidirectional=False)
        
        self._hidden_size = rnn_units
                
        self._top_classifier = nn.Linear(in_features=rnn_units+embedding_projections[product_col_name][1], 
                                         out_features=top_classifier_units)
        self._intermediate_activation = nn.ReLU()
        
        self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
        self.mixup = mixup
    
    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        
        if self.mixup:
            transactions_cat_features = add_noise(transactions_cat_features)
        
        embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
        concated_embeddings = torch.cat(embeddings, dim=-1)
        
        _, last_hidden = self._gru(concated_embeddings)

        last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, self._hidden_size))
        product_embed = self._product_embedding(product_feature)
        
        intermediate_concat = torch.cat([last_hidden, product_embed], dim=-1)
                
        classification_hidden = self._top_classifier(intermediate_concat)
        activation = self._intermediate_activation(classification_hidden)
        
        logit = self._head(activation)
        
        return logit

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0, emb_mult=1):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size*emb_mult, padding_idx=padding_idx)


class NextTransactionModel(nn.Module):
    def __init__(self, 
                 transactions_cat_features, 
                 embedding_projections, 
                 product_col_name='product',
                 num_layers=4,
                 emb_mult=1,
                ):

        super().__init__()
        self.embedding = EmbeddingLayer(transactions_cat_features, embedding_projections, product_col_name, emb_mult)
        inp_size = self.embedding.get_embedding_size()

        configuration = GPT2Config(vocab_size=1, n_positions=750, n_embd=inp_size, n_layer=num_layers, n_head=2)
        self.model = GPT2Model(configuration)

        self.head = NSPHead(inp_size, embedding_projections)

    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        mask1 = transactions_cat_features[-6] != 0
            
        mask = mask1.unsqueeze(1).unsqueeze(2)
        embedding = self.embedding(transactions_cat_features, product_feature)
        
        out = self.model(inputs_embeds=embedding, attention_mask=mask)
        logits = self.head(out.last_hidden_state)

        return logits


class MaskedModel(nn.Module):
    def __init__(self, 
                 transactions_cat_features, 
                 embedding_projections, 
                 product_col_name='product',
                 num_layers=4,
                 emb_mult=1,
                ):

        super().__init__()
        self.embedding = EmbeddingLayer(transactions_cat_features, embedding_projections, product_col_name, emb_mult)
        inp_size = self.embedding.get_embedding_size()

        configuration = BertConfig(vocab_size=1, n_positions=750, n_embd=inp_size, n_layer=num_layers, n_head=2)
        self.model = BertModel(configuration)

        self.head = NSPHead(inp_size, embedding_projections)

    def forward(self, transactions_cat_features, product_feature):
        batch_size = product_feature.shape[0]
        mask1 = transactions_cat_features[-6] != 0
            
        mask = mask1.unsqueeze(1).unsqueeze(2)
        embedding = self.embedding(transactions_cat_features, product_feature)
        
        out = self.model(inputs_embeds=embedding, attention_mask=mask)
        logits = self.head(out.last_hidden_state)

        return logits




    

