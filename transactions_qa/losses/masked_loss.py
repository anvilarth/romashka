import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, true, mask):
        mse_loss = nn.MSELoss(reduction='none')

        loss = mse_loss(pred, true)
        loss = (loss * mask.float()).sum()  # gives \sigma_euclidean over unmasked elements

        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements

        return mse_loss_val

