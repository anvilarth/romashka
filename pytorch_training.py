import torch
import wandb
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from augmentations import mask_tokens

from data_generators import batches_generator
from losses import NextTransactionLoss, MaskedMSELoss, NextNumericalFeatureLoss
from torch.utils.data import DataLoader


def train_epoch(model, optimizer, dataloader, task='default',
                print_loss_every_n_batches=500, device=None, scheduler=None,
                cat_weights=None, num_weights=None, number=None):
    
    if task == 'default':
        loss_function = nn.BCEWithLogitsLoss()
    elif task == 'next':
        loss_function = NextTransactionLoss()
    elif task == 'next_num_feature':
        loss_function = NextNumericalFeatureLoss(number)
    else:
        raise NotImplementedError
        
        
    num_batches = 1
    running_loss = 0.0

    model.train()

    for batch in tqdm(dataloader, desc='Training'):
        
        if task == 'default':
            output = torch.flatten(model(batch))
            batch_loss = loss_function(output, batch['label'].float())
            
        elif task == 'next':
            output = model(batch)
            batch_loss = loss_function(output, batch, num_weights=num_weights, cat_weights=cat_weights)
            
        elif task == 'next_num_feature':
            output = model(batch)
            batch_loss = loss_function(output, batch)
        
        batch_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        optimizer.zero_grad()

        running_loss += batch_loss

        if num_batches % print_loss_every_n_batches == 0:
            print(f'Training loss after {num_batches} batches: {running_loss / num_batches}', end='\r')
            wandb.log({'train_loss': batch_loss.item()})
        
        num_batches += 1
    
    print(f'Training loss after epoch: {running_loss / num_batches}', end='\r')
    

def eval_model(model, dataloader, task='default', data='vtb', batch_size=32, device=None, process_numerical=False) -> float:
    """
    функция для оценки качества модели на отложенной выборке, возвращает roc-auc на валидационной
    выборке
    :param model: nn.Module модель
    :param dataset_val: путь до директории с последовательностями
    :param batch_size: размер батча
    :param device: device, на который будут положены данные внутри батча
    :return: val roc-auc score
    """
    model.eval()

    num_batches = 0

    if task == 'default':
        batch_number = 0
        preds = []
        targets = []
        
    elif task == 'next':
        acc = 0.0
        pred_err = 0.0
        batch_number = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating model'):
            batch_number += 1
            if task == 'default':
                num_batches += 1
                targets.extend(batch['label'].cpu().numpy().flatten())
                output = model(batch)
                preds.extend(output.cpu().numpy().flatten())

            elif task == 'next':
                targets = torch.stack(batch['cat_features'])
                output = model(batch)
                cat_pred = list(map(lambda x: x.argmax(-1), output['cat_features']))
                t_pred = torch.stack(cat_pred)
                mask = batch['mask'][:, 1:]

                not_masked_acc = (t_pred == targets[..., 1:])
                
                if data == 'vtb':
                    mask = torch.ones_like(not_masked_acc)
                    mask_cat = targets[0] != 0
                    mask_trans = targets[-1] != 0

                    mask[0] = mask_cat[:, 1:]
                    mask[-1] = mask_trans[:, 1:]
                    mask[-2] = mask_trans[:, 1:]

                    acc += (not_masked_acc * mask).sum(axis=(1, 2))
                    num_batches += mask.sum(axis=(1,2))

                elif data == 'alfa':
                    acc += (not_masked_acc * mask).sum(axis=(1, 2))
                    num_batches += mask.sum()
                    
                
                pred = torch.tensor([(abs(output['num_features'][i].squeeze() - batch['num_features'][i][:, 1:]) / abs(output['num_features'][i].squeeze())).mean(1).mean(0) \
                        for i in range(len(batch['num_features']))])

                pred_err += pred  
                    
#             elif task == 'click':
#                 batch = list(map(lambda x: x.to(device), batch))
#                 mask = batch[0] != 0
#                 output = model(batch[0], mask=mask).squeeze()[:, :-1]

#                 not_masked_num_metric = (output - batch[1][:, 1:]) ** 2
#                 num_metric = (output * mask[:, 1:]).sum() / mask.sum()

    if task == 'default':
        return roc_auc_score(targets, preds), None
    else:  
        # output2 = num_metric 
        return acc / num_batches, pred_err / batch_number