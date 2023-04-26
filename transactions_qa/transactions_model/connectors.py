import sys
import torch
import torch.nn as nn


class TransactionConnector(nn.Module):
    def __init__(self, input_size, output_size, connector_type) -> None:
        super().__init__()
        if connector_type == 'id':
            self.connector = IdentityMapping()
        elif connector_type == 'linear':
            self.connector = LinearMapping(input_size, output_size)
        else:
            raise NotImplementedError

        self.output_size = output_size

    def forward(self, x, attention_mask=None):
        return self.connector(x, attention_mask)


class IdentityMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()

    def forward(self, x, attention_mask=None):
        return self.id(x)


class LinearMapping(nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()

        self.linear = nn.Linear(inp_size, out_size)

    def forward(self, x, attention_mask=None):
        return self.linear(x)
