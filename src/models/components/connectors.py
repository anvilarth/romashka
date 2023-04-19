import torch
import torch.nn as nn
import sys

from .perceiver_pytorch import Perceiver


class TransactionConnector(nn.Module):
    def __init__(self, input_size, output_size, connector_type) -> None:
        super().__init__()
        if connector_type == 'id':
            self.connector = IdentityMapping()
        elif connector_type == 'linear':
            self.connector = LinearMapping(input_size, output_size)
        elif connector_type == 'perceiver':
            self.connector = PerceiverMapping(input_size, output_size)
        else:
            raise NotImplementedError

    def forward(self, x, attention_mask=None):
        return self.connector(x, attention_mask)

class IdentityMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()
    
    def forward(self, x, attention_mask=None):
        return  self.id(x)


class LinearMapping(nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()
        
        self.linear = nn.Linear(inp_size, out_size)
        
    def forward(self, x, attention_mask=None):
        return self.linear(x)
    
    
class PerceiverMapping(nn.Module):
    def __init__(self, inp_size, out_size, num_latents=196):
        super().__init__()
        self.perceiver = Perceiver(
                input_channels=inp_size,
                depth = 1,                   # depth of net. The shape of the final attention mechanism will be:                                
                num_latents = num_latents,           # number of latents, or induced set points, or centroids. different papers giving it different names
                latent_dim = out_size,            # latent dimension
                cross_heads = 1,             # number of heads for cross attention. paper said 1   
                cross_dim_head = inp_size,         # number of dimensions per cross attention head       
            )
    
    def forward(self, x, attention_mask=None):
        return self.perceiver(x, mask=attention_mask)