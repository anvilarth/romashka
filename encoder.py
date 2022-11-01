import sys
import torch
import torch.nn as nn

sys.path.append('./FEDformer')

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers import Autoformer_EncDec

from utils.masking import TriangularCausalMask, ProbMask

from layers import Transformer_EncDec
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
import numpy as np

class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super().__init__()
        # Encoder
        self.encoder = Transformer_EncDec.Encoder(
            [
                Transformer_EncDec.EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
#             [
#                 ConvLayer(
#                     configs.d_model
#                 ) for l in range(configs.e_layers - 1)
#             ] 
#             if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
       

    def forward(self, x_enc, enc_self_mask=None):

        enc_out, attns = self.encoder(x_enc, attn_mask=enc_self_mask)
        return enc_out

class InformerConfigs:
    def __init__(self, d_model, dropout, n_heads, num_layers):
        self.moving_avg = [12, 24]
        self.seq_len = 750
        self.d_model = d_model
        self.dropout = dropout
        self.factor = 1
        self.n_heads = n_heads
        self.d_ff = d_model
        self.e_layers = num_layers
        self.activation = 'gelu'


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        rel_pos_embs=False,
    ):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims/heads)
        self.pos_emb = rel_pos_embs
        
        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        self.fc = nn.Linear(self.head_dims*self.heads, self.embedding_dims)
        self.max_relative_position = 512
        
        if self.pos_emb:
            self.relative_position_k = RelativePosition(self.head_dims, self.max_relative_position)
            self.relative_position_v = RelativePosition(self.head_dims, self.max_relative_position)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]

        query = query.reshape(batch_size, query_len, self.heads, self.head_dims)
        key = key.reshape(batch_size, key_len, self.heads, self.head_dims)
        value = value.reshape(batch_size, value_len, self.heads, self.head_dims)

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        
        attention_score = torch.einsum('bqhd,bkhd->bhqk', [query, key])

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, float('-1e20'))
        
        if self.pos_emb:
            a_key = self.relative_position_k(query_len, key_len)
            a_value = self.relative_position_v(query_len, value_len)

            relative_qk_attn = torch.einsum("blhd,lrd->bhlr", query, a_key)
            attention_score += relative_qk_attn

        attention_score = attention_score/((self.head_dims)**(1/2))
        attention_score = torch.softmax(attention_score, dim=-1)

        qkv_attn = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
            batch_size, query_len, self.heads*self.head_dims
        )
        out = qkv_attn
        
        # CHECK is  output index is correct?
        if self.pos_emb:
            relative_qkv_attn = torch.einsum("brhl, hld->bhrd", attention_score, a_value).reshape(
            batch_size, query_len, self.heads*self.head_dims
        )
            out += relative_qkv_attn
#         relative_qkv_attn = torch.einsum("brhd,lrd->bhlr", query, a_value)
        
        out = self.fc(out)

        return out
    
class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class BertBlock(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        dropout, 
        forward_expansion,
        layer_norm_eps,
        rel_pos_embs=False,
    ):
        super(BertBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.attention = SelfAttention(embedding_dims, heads, rel_pos_embs)
        self.feed_forward = nn.Sequential(
                nn.Linear(embedding_dims, embedding_dims*forward_expansion),
                nn.GELU(),
                nn.Linear(embedding_dims*forward_expansion, embedding_dims)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attention_block = self.attention(x, x, x, mask)
        add = self.dropout(self.layer_norm1(x + attention_block))
        feed_forward = self.feed_forward(add)
        out = self.dropout(self.layer_norm2(feed_forward + add))
        return out



class BERT(nn.Module):
    def __init__(
        self,
        embedding_dims,
        heads,
        num_layers,
        dropout,
        layer_norm_eps,
        rel_pos_embs,
        forward_expansion=1,
        mask_idx=0,
    ):
        super().__init__()
        print(f"USING rel_pos_embs={rel_pos_embs}")
        self.bert_blocks = nn.Sequential(
            *[
                BertBlock(
                    embedding_dims,
                    heads,
                    dropout,
                    forward_expansion,
                    layer_norm_eps,
                    rel_pos_embs,
                )
                for _ in range(num_layers)
            ]
            
        )

        self.layer_norm = nn.LayerNorm(embedding_dims, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.mask_idx = mask_idx

        self.apply(self._init_weight)
    
    # @hugging_face
    def _init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
        
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def create_mask(self, mask):
        mask  = (mask != self.mask_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def forward(self, x, mask):
        for block in self.bert_blocks:
            x = block(x, mask)
        return x