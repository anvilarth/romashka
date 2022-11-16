import torch 
import torch.nn as nn

from losses.masked_loss import MaskedMSELoss

class NextTransactionLoss(nn.Module):
    def __init__(self, process_numerical):
        super().__init__()
        self.process_numerical = process_numerical
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.num_criterion = MaskedMSELoss()

    def forward(self, logits, labels, mask=None):
        res = []
        
        if self.process_numerical:
            for (pred, true) in zip(logits[:-3], labels[:-3]):
                res.append(self.criterion(pred.permute(0, 2, 1), true[:, 1:]))

            for i in range(3):
                pred, true = logits[-3+i], labels[-3+i]
                res.append(self.num_criterion(pred.squeeze(), true[:, 1:].squeeze(), mask))
        
        else: 
            for (pred, true) in zip(logits, labels):
                res.append(self.criterion(pred.permute(0, 2, 1), true[:, 1:]))

        return sum(res)