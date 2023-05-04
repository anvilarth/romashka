import torch
import torch.nn as nn

from typing import Optional

from romashka.transactions_qa.layers.initialization import init_xavier_uniform_layers


class LambdaLayer(nn.Module):
    """
    A simple layer for function application in autograd graph.
    """
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x, *args, **kwargs):
        return self.function(x)


class MuiltiHeadSelfAttention(nn.Module):

    def __init__(
            self,
            embedding_dims: int,
            heads: int,
            has_rel_pos_embeddings: Optional[bool] = False,
            max_relative_position: Optional[int] = 512
    ):
        super(MuiltiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.embedding_dims = embedding_dims
        self.head_dims = int(embedding_dims / heads)
        self.has_rel_pos_embeddings = has_rel_pos_embeddings
        self.max_relative_position = max_relative_position
        self._create_layers()

    def _create_layers(self):
        # Create K, Q, V projections
        self.key = nn.Linear(self.head_dims, self.head_dims)
        self.query = nn.Linear(self.head_dims, self.head_dims)
        self.value = nn.Linear(self.head_dims, self.head_dims)

        # Initialize
        for l in [self.key, self.query, self.value]:
            init_xavier_uniform_layers(l)

        self.fc = nn.Linear(self.head_dims * self.heads, self.embedding_dims)
        init_xavier_uniform_layers(self.fc)

        if self.has_rel_pos_embeddings:
            self.relative_position_k = RelativePositionEmbeddings(self.head_dims, self.max_relative_position)
            self.relative_position_v = RelativePositionEmbeddings(self.head_dims, self.max_relative_position)

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
            attention_score = attention_score.masked_fill(mask == 0, float('-1e20'))

        if self.has_rel_pos_embeddings:
            a_key = self.relative_position_k(query_len, key_len)
            a_value = self.relative_position_v(query_len, value_len)

            relative_qk_attn = torch.einsum("blhd,lrd->bhlr", query, a_key)
            attention_score += relative_qk_attn

        attention_score = attention_score / ((self.head_dims) ** (1 / 2))
        attention_score = torch.softmax(attention_score, dim=-1)

        qkv_attn = torch.einsum('bhqv,bvhd->bqhd', [attention_score, value]).reshape(
            batch_size, query_len, self.heads * self.head_dims
        )
        out = qkv_attn

        if self.has_rel_pos_embeddings:
            relative_qkv_attn = torch.einsum("brhl, hld->bhrd", attention_score, a_value).reshape(
                batch_size, query_len, self.heads * self.head_dims
            )
            out += relative_qkv_attn
        #   relative_qkv_attn = torch.einsum("brhd,lrd->bhlr", query, a_value)

        out = self.fc(out)

        return out


class RelativePositionEmbeddings(nn.Module):

    def __init__(self, num_units: int, max_relative_position: Optional[int] = 512):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        init_xavier_uniform_layers(self.embeddings_table)

    def forward(self, length_q: int, length_k: int):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


class TransformerEncoderLayer(nn.Module):
    """
    A simple Transformer Encoder layer (full-attention) with Multi-head self-attention (with pre-normalization).
    """
    def __init__(
            self,
            embedding_dim: int,
            heads: int,
            dropout: Optional[float] = 0.1,
            ff_output_dim: Optional[int] = None,
            forward_expansion: Optional[int] = 2,
            layer_norm_eps: Optional[float] = 1e-5,
            has_rel_pos_embeddings: Optional[bool] = False,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.attention = MuiltiHeadSelfAttention(embedding_dim, heads, has_rel_pos_embeddings)
        ff_output_dim = ff_output_dim if ff_output_dim is not None else embedding_dim * forward_expansion

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_output_dim),
            nn.GELU(),
            nn.Linear(ff_output_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_block = self.attention(x, x, x, mask)
        add = self.dropout(self.layer_norm1(x + attention_block))
        feed_forward = self.feed_forward(add)
        out = self.dropout(self.layer_norm2(feed_forward + add))
        return out
