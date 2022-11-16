import torch
import numpy as np 
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
