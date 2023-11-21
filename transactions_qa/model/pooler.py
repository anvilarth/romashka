from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import TensorType

import transformers
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions

from romashka.logging_handler import get_logger

logger = get_logger(
    name="Pooler"
)


class MeanPooler(nn.Module):
    """
    Mean pooling.
    """

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


class MaxPooler(nn.Module):
    """
    Max pooling.
    """

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


class ClsPooler(nn.Module):
    """
    CLS token pooling.
    """
    def __init__(self, cls_token_position: Optional[int] = 0,
                 use_pooler_output: Optional[bool] = True):
        super().__init__()
        self.cls_token_position = cls_token_position
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


class ClsLastHiddenStatePooler(nn.Module):
    """
    CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """
    def __init__(self, cls_token_position: Optional[int] = 0):
        super().__init__()
        self.cls_token_position = cls_token_position

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


class EosLastHiddenStatePooler(nn.Module):
    """
    EOS token pooling.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        # attention_mask of size: (batch_size, sequence_length) -> (batch_size,)
        last_tokens_ids = attention_mask.sum(-1)
        # last hidden state of size: (batch_size, sequence_length, hidden_size)
        return x.hidden_states[-1][:, last_tokens_ids, :]


POOLER_TYPES = [
    ("CLS_POOLER", ClsPooler),
    ("CLS_LHS_POOLER", ClsLastHiddenStatePooler),
    ("EOS_POOLER", EosLastHiddenStatePooler),
    ("MEAN_POOLER", MeanPooler),
    ("MAX_POOLER", MaxPooler)
]
POOLER_TYPES = OrderedDict(POOLER_TYPES)


class PoolerType:
    """
    Selector class for specific pooling types.
    """
    @classmethod
    def get(cls, pooler_type_name: str, **kwargs):
        try:
            return POOLER_TYPES[pooler_type_name](**kwargs)
        except Exception as e:
            logger.error(f"Error during PoolerType creation with `pooler_type_name`-`{pooler_type_name}`\n:{e}")
            raise ValueError(f"Error during PoolerType creation with `pooler_type_name`-`{pooler_type_name}`\n:{e}")

    @classmethod
    def get_available_names(cls):
        """
        Returns a list of available enumeration name.
        """
        return [member for member in POOLER_TYPES.keys()]

    @classmethod
    def to_str(cls):
        s = " / ".join([member for member in POOLER_TYPES.keys()])
        return s