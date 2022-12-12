import torch
import sys
import torch.nn as nn

import torch.nn.functional as F
from time import time
from transformers import GPT2Model, GPT2Config

class TransactionsModel(nn.Module):
    def __init__(self, 
                 cat_embedding_projections,
                 cat_features,
                 num_embedding_projections=None,
                 num_features=None,  
                 head_type='linear',
                 encoder_type='bert',
                 num_layers=6, 
                 dropout=0.0,
                 cutmix=False,
                 mixup=False,
                 emb_mult=1,
                 rel_pos_embs=False,
                 alpha=1.0):

        super().__init__()
        self.embedding = EmbeddingLayer(cat_embedding_projections, 
                                        cat_features, 
                                        num_embedding_projections, 
                                        num_features)
        
        inp_size = self.embedding.get_embedding_size()

        self.head_type = head_type
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_size) / inp_size)

        if encoder_type == 'rnn':
            self.encoder = nn.GRU(inp_size, inp_size, batch_first=True)
            
        elif encoder_type == 'gpt':
            configuration = GPT2Config(vocab_size=1, n_positions=760, n_embd=inp_size, n_layer=num_layers, n_head=2)
            self.model = GPT2Model(configuration)
        
        else:
            raise NotImplementedError

        if head_type == 'linear':
            self.head = LinearHead(inp_size)
        elif head_type == 'rnn':
            self.head = RNNClassificationHead(inp_size, inp_size)
        elif head_type == 'id':
            self.head = nn.Identity()
            
        elif head_type == 'next':
            self.head = NSPHead(inp_size, cat_embedding_projections)
        else:
            raise NotImplementedError

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha
    
    def forward(self, batch):
        mask = batch['mask']
        batch_size = mask.shape[0]
        
        embedding = self.embedding(batch)
        
        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        embedding = torch.cat([embedding, cls_token], dim=1)
        
        cls_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
        mask = torch.cat([mask, cls_token_mask], dim=1)
        mask = mask.unsqueeze(1).unsqueeze(2)

            
        x = self.encoder(embedding, mask)
        logit = self.head(x)
        
        return logit
    
    
    
import torch
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 1)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = x[:, -1]
        return self.linear1(x)


class RNNClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self._gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True, bidirectional=True)
        self.linear = ClassificationHead(input_size * 2, hidden_size)
    
    def forward(self, x):
        batch_size, _ , d = x.shape
        _, last_hidden = self._gru(x)

        last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, d * 2))

        return self.linear(last_hidden)

class NSPHead(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 cat_embedding_projections,
                 num_embedding_projections=None):
        
        super().__init__()

        cat_heads = []
        for elem in cat_embedding_projections:
            head = nn.Linear(embedding_dim, cat_embedding_projections[elem][0] + 1)
            cat_heads.append(head)

        self.cat_heads = nn.ModuleList(cat_heads)
        
        num_heads = []
        self.num_heads = None
        
        if num_embedding_projections is not None:
            for elem in num_embedding_projections:
                head = nn.Linear(embedding_dim, 1)
                num_heads.append(head)
                
            self.num_heads = nn.ModuleList(num_heads)

    def forward(self, x):
        cat_res = []
        for m in self.cat_heads:
            tmp = m(x)
            cat_res.append(tmp[:, :-1])

        num_res = []
        if self.num_heads is not None:
            for m in self.num_heads:
                tmp = m(x)
                num_res.append(tmp[:, :-1])
                
        return {'cat_features': cat_res, 'num_features': num_res} 
    
    
import torch
import torch.nn as nn
from copy import deepcopy

class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 cat_embedding_projections,
                 cat_features,
                 num_embedding_projections=None,
                 num_features=None):
        super().__init__()
        self.cat_embedding = CatEmbedding(cat_embedding_projections, cat_features)
        self.num_embedding = None
        if num_embedding_projections is not None and num_features is not None:
            self.num_embedding = NumericalEmbedding(num_embedding_projections, num_features)
        
    def get_embedding_size(self):
        res = self.cat_embedding.get_embedding_size()
        if self.num_embedding is not None:
            res += self.num_embedding.get_embedding_size()
        return res
        
    def forward(self, batch):
        cat_features, num_features = batch['cat_features'], batch['num_features']
        
        cat_embeddings = self.cat_embedding(cat_features)
        if self.num_embedding is not None:
            num_embeddings = self.num_embedding(num_features)
            return torch.cat([cat_embeddings, num_embeddings], dim=-1)
     
        else:
            return cat_embeddings
        
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
