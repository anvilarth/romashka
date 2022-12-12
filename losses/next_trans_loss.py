import torch 
import torch.nn as nn

from losses.masked_loss import MaskedMSELoss

class NextTransactionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.num_criterion = MaskedMSELoss()

    def forward(self, output, batch, mask=None, cat_weights=None, num_weights=None):
        cat_pred, num_pred = output['cat_features'], output['num_features']
        cat_trues, num_trues = batch['cat_features'], batch['num_features']
        mask = batch['mask'][:, 1:]
        
        res = []
        
        for i, (pred, true) in enumerate(zip(cat_pred, cat_trues)):
            if cat_weights is not None:
                coef = cat_weights[i]
            else:
                coef = 1.0
            res.append(coef * self.cat_criterion(pred.permute(0, 2, 1), true[:, 1:]))
        
        for i, (pred, true) in enumerate(zip(num_pred, num_trues)):
            if num_weights is not None:
                coef = num_weights[i]
            else:
                coef = 1.0
                
            res.append(coef * self.num_criterion(pred.squeeze(), true[:, 1:].squeeze(), mask))

        return sum(res)