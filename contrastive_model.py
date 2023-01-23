import torch
import sys
import torch.nn as nn

import torch.nn.functional as F
from time import time
from transformers import GPT2Model, GPT2Config, BertConfig, BertModel

from embedding import EmbeddingLayer
from augmentations import mixup_data

from encoder import BERT, Informer, InformerConfigs
from head import LinearHead, RNNClassificationHead, NSPHead
from tools import LambdaLayer

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=32):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class PretrainModel(nn.Module):
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
                 num_layers=6, 
                 dropout=0.0,
                 cutmix=False,
                 mixup=False,
                 emb_mult=1,
                 rel_pos_embs=False,
                 add_token='before',
                 alpha=1.0):

        super().__init__()
        self.embedding = EmbeddingLayer(cat_embedding_projections, 
                                        cat_features, 
                                        num_embedding_projections, 
                                        num_features,
                                        meta_embedding_projections,
                                        meta_features,
                                        time_embedding)
        
        inp_size = self.embedding.get_embedding_size()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_size) / inp_size)

        self.head_type = head_type
        self.encoder_type = encoder_type
        self.cls_token = nn.Parameter(torch.randn(1, 1, inp_size) / inp_size)

        self.encoder = BERT(inp_size, heads=2*emb_mult, num_layers=num_layers, dropout=dropout, layer_norm_eps=1e-7, rel_pos_embs=rel_pos_embs)
        self.mlp = MLP(inp_size,  90)
        self.head = NSPHead(inp_size, cat_embedding_projections, num_embedding_projections)
        
        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha
        self.add_token = add_token
    
    def get_embs(self, batch=None, embeds=None):
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
            
        return x, mask
    
    def forward(self, batch):
        mask = batch['mask']
        batch_size = mask.shape[0]

        embedding = self.embedding(batch)
        mask = mask.unsqueeze(1).unsqueeze(2)
        x = self.encoder(embedding, mask)
                
        # logit = self.head(x, mask)
        return x