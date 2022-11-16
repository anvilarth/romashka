import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2Config, BertConfig, BertModel

from embedding import EmbeddingLayer
from augmentations import add_noise, mixup_data

from encoder import BERT, Informer, InformerConfigs
from head import ClassificationHead, RNNClassificationHead, NSPHead


class TokenGPT(nn.Module):
    def __init__(self, 
                 transactions_cat_features, 
                 embedding_projections, 
                 product_col_name='product',
                 num_layers=4,
                 emb_mult=1,
                 num_buckets=None,
                ):

        super().__init__()
        self.embedding = EmbeddingLayer(transactions_cat_features, embedding_projections, product_col_name, emb_mult, num_buckets=num_buckets)
        inp_size = self.embedding.get_embedding_size()

        configuration = GPT2Config(vocab_size=1, n_positions=760, n_embd=inp_size, n_layer=num_layers, n_head=2)
        self.model = GPT2Model(configuration)

        self.head = ClassificationHead(inp_size, inp_size)
        self.default_token = nn.Parameter(torch.randn(1, 1, inp_size) / inp_size)

    def forward(self, batch):
        transactions_cat_features, product_feature = batch['transactions_features'], batch['product']
        batch_size = product_feature.shape[0]
        mask1 = transactions_cat_features[-6] != 0
        
        first_token_mask = torch.ones(batch_size, 1, dtype=bool, device=transactions_cat_features[0].device)
        mask1 = torch.cat([mask1, first_token_mask], dim=1)
            
        mask = mask1.unsqueeze(1).unsqueeze(2)
        embedding = self.embedding(batch)
        embedding = torch.cat([embedding, self.default_token.repeat(batch_size, 1, 1)], dim=1)
        
        out = self.model(inputs_embeds=embedding, attention_mask=mask).last_hidden_state[:, -1]
        logits = self.head(out)
        
        return logits
