import torch
import numpy as np

def add_noise(x_categ, mask = None, lam = 0.1):
    device = x_categ[0].device
    batch_size = x_categ[0].shape[0]    
    full_tensor = torch.stack(x_categ)

    index = torch.randperm(batch_size)
    cat_corr = torch.from_numpy(np.random.choice(2,(full_tensor.shape),p=[lam,1-lam])).to(device)
    x1  =  full_tensor[:, index]
    x_categ_corr = full_tensor.clone().detach()
    x_categ_corr[(cat_corr==0) & (mask == 1)] = x1[(cat_corr==0) & (mask == 1)]
    
    return list(x_categ_corr)

def mask_tokens(x_categ, mask = None, lam = 0.1):
    device = x_categ[0].device
    batch_size = x_categ[0].shape[0]    
    full_tensor = torch.stack(x_categ)

    cat_corr = torch.from_numpy(np.random.choice(2,(full_tensor.shape),p=[lam,1-lam])).to(device)
    full_tensor[(cat_corr==0) & (mask == 1)] = 0
    new_mask = (cat_corr==0) & (mask == 1)

    return list(full_tensor), new_mask

def mixup_data(x, mask = None, alpha=0.9):
    '''Returns mixed inputs, pairs of targets'''
    
    device = x.device
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)
    mixed_x = alpha * x * mask + (1 - alpha) * x[index, :] * mask

    return mixed_x