from typing import List, Optional

import torch
from torch import nn as nn
from torch.nn import Linear, BatchNorm1d, Sigmoid, Sequential, ReLU, LogSoftmax, Flatten, Softplus, Dropout



class L2NormEncoder(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x / (x.pow(2).sum(dim=-1, keepdim=True) + self.eps).pow(0.5)


class Head(torch.nn.Module):
    r"""
    Head for the sequence encoder

    Parameters
    ----------
         input_size: int
            input size
         use_norm_encoder: bool. Default: False
            whether to use normalization layers before the head
         use_batch_norm: bool. Default: False.
            whether to use BatchNorm.
         hidden_layers_sizes: List[int]. Default: None.
            sizes of linear layers. If None without additional linear layers.
         objective: str. Options:
            None (default) - corresponds to linear output with relu
            classification - linear output with sigmoid or logsoftmax (num_classes > 1)
            regression - pure linear output
            softplus - linear output with softplus
         num_classes: int. Default: 1.
            The number of classed in classification problem. Default correspond to binary classification.

     """

    def __init__(self,
                 input_size: int = None,
                 use_norm_encoder: bool = False,
                 use_batch_norm: bool = False,
                 hidden_layers_sizes: List[int] = None,
                 drop_probs: List[float] = None,
                 objective: str = None,
                 num_classes: int = 1):
        super().__init__()
        # TODO: check possibility to create empty head with do nothing

        layers = []

        if use_batch_norm:
            layers.append(BatchNorm1d(input_size))

        if drop_probs: assert len(drop_probs) == len(hidden_layers_sizes), \
            'dimensions of `drop_probs` and `hidden_layers_sizes` should be equal'

        if hidden_layers_sizes is not None:
            layers_size = [input_size] + list(hidden_layers_sizes)
            for ix, (size_in, size_out) in enumerate(zip(layers_size[:-1], layers_size[1:])):
                layers.append(Linear(size_in, size_out))
                if use_batch_norm:
                    layers.append(BatchNorm1d(size_out))
                layers.append(ReLU())
                if drop_probs:
                    layers.append(Dropout(drop_probs[ix]))
                input_size = size_out

        if objective == 'classification':
            if num_classes == 1:
                h = Sequential(Linear(input_size, num_classes), Sigmoid(), Flatten(0))
            else:
                h = Sequential(Linear(input_size, num_classes), LogSoftmax(dim=1))
            layers.append(h)

        elif objective == 'regression':
            if num_classes == 1:
                layers.append(Sequential(Linear(input_size, 1), Flatten(0)))
            else:
                layers.append(Linear(input_size, num_classes))

        elif objective == 'softplus':
            if num_classes == 1:
                layers.append(Sequential(Linear(input_size, num_classes), Softplus(), Flatten(0)))
            else:
                layers.append(Sequential(Linear(input_size, num_classes), Softplus()))

        elif objective is not None:
            raise AttributeError(f"Unknown objective {objective}. Supported: classification, regression and softplus.")

        if use_norm_encoder:
            layers.append(L2NormEncoder())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class LLMClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks.
    """

    def __init__(self, d_model: int, num_labels: int,
                 classifier_dropout: Optional[float] = 0.0):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(d_model, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states