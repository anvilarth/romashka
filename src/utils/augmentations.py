import torch
import numpy as np

def cutmix_data(x_categ, x_num, mask = None, lam = 0.1):
    device = x_categ[0].device
    batch_size = x_categ[0].shape[0]    
    categ_tensor = torch.stack(x_categ)
    num_tensor = torch.stack(x_num)
    
    index = torch.randperm(batch_size)
    cat_corr = torch.from_numpy(np.random.choice(2,(categ_tensor.shape),p=[lam,1-lam])).to(device)
    num_corr = torch.from_numpy(np.random.choice(2,(num_tensor.shape),p=[lam,1-lam])).to(device)
    
    x_categ_shuffled  =  categ_tensor[:, index]
    x_num_shuffled = num_tensor[:, index]
        
    categ_tensor[(cat_corr==0) & (mask == 1)] = x_categ_shuffled[(cat_corr==0) & (mask == 1)]
    num_tensor[(num_corr==0) & (mask == 1)] = x_num_shuffled[(num_corr==0) & (mask == 1)]
    
    return list(categ_tensor), list(num_tensor)

def mask_tokens(x_categ, x_num, lam = 0.1):
    device = x_categ[0].device
    batch_size = x_categ[0].shape[0]   
    
    categ_tensor = torch.stack(x_categ)
    num_tensor = torch.stack(x_num)
    
    cat_corr = torch.from_numpy(np.random.choice(2,(categ_tensor.shape[:2]),p=[lam,1-lam])).to(device)
    num_corr = torch.from_numpy(np.random.choice(2,(num_tensor.shape[:2]),p=[lam,1-lam])).to(device)
    
    categ_tensor[(cat_corr==0)] = 0
    num_tensor[(num_corr==0)] = 0
    return list(categ_tensor), list(num_tensor), cat_corr, num_corr

def mixup_data(x, mask = None, alpha=0.9):
    '''Returns mixed inputs, pairs of targets'''
    
    device = x.device
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(device)
    mixed_x = alpha * x * mask.unsqueeze(-1) + (1 - alpha) * x[index, :] * mask.unsqueeze(-1)

    return mixed_x