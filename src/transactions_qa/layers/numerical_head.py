import torch 
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, output_size)
    
    def forward(self, x, mask=None):
        return self.linear1(x)

class MLPHead(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, output_size)
        
        self.act = nn.GELU()
    
    def forward(self, x, mask=None):
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        
        return x