import torch
import wandb
import pandas as pd
import torch.nn as nn
import numpy as np

from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score
from augmentations import mask_tokens

from data_generators import batches_generator, transaction_features, num_features_names, cat_features_names, meta_features_names
from losses import NextTransactionLoss, MaskedMSELoss, NextTimeLoss, NextNumericalFeatureLoss
from torch.utils.data import DataLoader

from tools import make_time_batch, masked_mean


def train_epoch(model, optimizer, train_dataloader, val_dataloader, task='default',
                print_loss_every_n_batches=500, device=None, scheduler=None,
                cat_weights=None, num_weights=None, num_feature_ids=None, cat_feature_ids=None, val_steps=None):
    
    cat_feature_ids = [] if cat_feature_ids is None else cat_feature_ids
    num_feature_ids = [] if num_feature_ids is None else num_feature_ids
    
    if task == 'default':
        loss_function = nn.BCEWithLogitsLoss()
    elif task == 'next':
        loss_function = NextTransactionLoss()
    elif task == 'next_time':
        loss_function = NextTimeLoss()
    # elif task == 'product':
    #     loss_function = nn.CrossEntropyLoss()
    # elif task == 'next_num_feature':
    #     loss_function = NextNumericalFeatureLoss(num_number)
    # elif task == 'next_cat_feature':
    #     loss_function = NextCatFeatureLoss(cat_number)
    else:
        raise NotImplementedError
        
        
    num_batches = 1
    running_loss = 0.0

    model.train()

    for batch_idx, batch in tqdm(enumerate(train_dataloader), desc='Training'):
        
        if task == 'default':
            output = torch.flatten(model(batch))
            batch_loss = loss_function(output, batch['label'].float())
            
        elif task == 'next':
            output = model(batch)
            batch_loss = loss_function(output, batch, num_weights=num_weights, cat_weights=cat_weights, cat_feature_ids=cat_feature_ids, num_feature_ids=num_feature_ids)
        
        elif task == 'next_time':
            trues = make_time_batch(batch)
            padding_mask = trues[-1]
            
            if any(padding_mask.sum(1) == 0):
                continue
                
            output = model(batch)
            batch_loss = loss_function(output, trues, mask=padding_mask) 
        
        # elif task == 'product':
        #     output = model(batch)
        #     trues = batch['meta_features'][0]
        #     batch_loss = loss_function(output, trues)
            
        batch_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2.0)
        
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
    # if val_steps is not None:
    #     if batch_idx % val_steps == 0 and batch_idx != 0:
    #         eval_model(model, val_dataloader)


def eval_model(model, dataloader, epoch, task='default', data='vtb', batch_size=32, device=None, process_numerical=False, train=False, num_feature_ids=None, cat_feature_ids=None) -> float:
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
    
    cat_feature_ids = [] if cat_feature_ids is None else cat_feature_ids
    num_feature_ids = [] if num_feature_ids is None else num_feature_ids
    
    start = 'train_' if train else 'val_'
    num_objects = 0

    if task == 'default':
        log_dict = {start + 'roc_auc': 0.0, }
        preds = []
        targets = []
        
    # elif task == 'product':
    #     log_dict = {start + 'acc': 0.0}
    #     preds = []
    #     targets = []
        
    elif task == 'next':
        log_dict = {}
        for j in num_feature_ids:
            log_dict[start + num_features_names[j]] = 0.0
        
        for j in cat_feature_ids:
            log_dict[start + cat_features_names[j]] = 0.0
            
        for i in [1, 3, 6, 12, 24, 72, 168]:
            log_dict[start + f'interval_{i}_hours'] = 0.0
            
        
    elif task == 'next_time':
        log_dict = {start + 'amnt': 0.0, start + 'num': 0.0, start + 'num_accuracy': 0.0,
                    start + 'code_f1': 0.0, start + 'code_recall': 0.0, start + 'code_precision': 0.0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating model'):
            num_objects += batch['mask'].shape[0]
            if task == 'default':
                targets.extend(batch['label'].cpu().numpy().flatten())
                output = model(batch)
                # if task == 'product':
                #     output = output.argmax(-1) 
                    
                preds.extend(output.cpu().numpy().flatten())

            elif task == 'next':
                output = model(batch)
                mask = batch['mask'][:, 1:]
                
                cat_pred = list(map(lambda x: x.argmax(-1), output['cat_features']))
                t_pred = torch.stack(cat_pred)
                t_targets = torch.stack(batch['cat_features'])[..., 1:]
                
                not_masked_acc = (t_pred == t_targets) # cat_feat x bs x seq_len
                masked_acc = (not_masked_acc * mask).sum(axis=2) # cat_feat x bs
                num_transactions = mask.sum(1)
                cat_tmp = (masked_acc / num_transactions).sum(1)

                for i in cat_feature_ids:
                    log_dict[start + cat_features_names[i]] += cat_tmp[i]

#                 not_masked_acc = (t_pred == targets[..., 1:])
                
#                 if data == 'vtb':
#                     mask = torch.ones_like(not_masked_acc)
#                     mask_cat = targets[0] != 0
#                     mask_trans = targets[-1] != 0

#                     mask[0] = mask_cat[:, 1:]
#                     mask[-1] = mask_trans[:, 1:]
#                     mask[-2] = mask_trans[:, 1:]

#                     acc += (not_masked_acc * mask).sum(axis=(1, 2))
#                     num_batches += mask.sum(axis=(1,2))

#                 elif data == 'alfa':
#                     acc += (not_masked_acc * mask).sum(axis=(1, 2))
#                     num_batches += mask.sum()
              
                
                num_tmp = [masked_mean(abs(output['num_features'][i].squeeze() - batch['num_features'][i][:, 1:]), mask).sum().item() for i in range(len(num_features_names))]

                for i in num_feature_ids:
                    log_dict[start + num_features_names[i]] += num_tmp[i]
                    
                    
                for i in [1, 3, 6, 12, 24, 72, 168]:
                    tmp_interval = torch.isclose(output['num_features'][-1].squeeze(2)[:, 1:], batch['num_features'][-1][:, 1:-1], atol=i/95, rtol=0.0).float()
                    tmp_interval_masked = masked_mean(tmp_interval, mask[:, 1:]).sum().item()
                    log_dict[start + f'interval_{i}_hours'] += tmp_interval_masked
            
            elif task == 'next_time':

                trues = make_time_batch(batch)
                all_amnt_transactions, all_num_transactions, all_code_transactions, padding_mask = trues
                
                if any(padding_mask.sum(1) == 0):
                    continue
                
                output = model(batch)
                code_preds = (torch.sigmoid(output[-1]) > 0.5).int()
                
                f1s = []
                prs = []
                recalls = []

                for i in range(batch['mask'].shape[0]):
                    indices = torch.where(padding_mask[i] == 1)[0]
                    
                    f1s.append(f1_score(code_preds[i][indices].cpu().numpy(), all_code_transactions[i][indices].cpu().numpy(), average=None, zero_division=1))
                    prs.append(precision_score(code_preds[i][indices].cpu().numpy(), all_code_transactions[i][indices].cpu().numpy(), average=None, zero_division=1))
                    recalls.append(recall_score(code_preds[i][indices].cpu().numpy(), all_code_transactions[i][indices].cpu().numpy(), average=None, zero_division=1))
                
                log_dict[start + 'code_f1'] +=  np.sum(f1s, axis=0)
                log_dict[start + 'code_precision'] += np.sum(prs, axis=0)
                log_dict[start + 'code_recall'] += np.sum(recalls, axis=0)
                
                masked_amnt = masked_mean(abs(output[0].squeeze(2) - all_amnt_transactions), padding_mask)
                masked_num = masked_mean(abs(output[1].squeeze(2) - all_num_transactions), padding_mask)
                masked_num_acc = masked_mean((torch.clamp(torch.round(output[1]).squeeze(2),  0) == all_num_transactions).float(), padding_mask)
                
                log_dict[start + 'amnt'] += masked_amnt.sum().item()
                log_dict[start + 'num'] += masked_num.sum().item()
                log_dict[start + 'num_accuracy'] += masked_num_acc.sum().item()
                
#                 pred = torch.tensor([(abs(output['num_features'][i].squeeze() - batch['num_features'][i][:, 1:]) / abs(output['num_features'][i].squeeze())).mean(1).mean(0) \
#                         for i in range(len(batch['num_features']))])

#                 pred_err += pred  
                    
#             elif task == 'click':
#                 batch = list(map(lambda x: x.to(device), batch))
#                 mask = batch[0] != 0
#                 output = model(batch[0], mask=mask).squeeze()[:, :-1]

#                 not_masked_num_metric = (output - batch[1][:, 1:]) ** 2
#                 num_metric = (output * mask[:, 1:]).sum() / mask.sum()

    if task == 'default':
        log_dict[start + 'roc_auc'] = roc_auc_score(targets, preds)
        
    # elif task == 'product':
    #     log_dict[start + 'acc'] = accuracy_score(targets, preds)
        
    elif task == 'next_time':
        it_list = list(log_dict.keys())
        
        for elem in it_list:
            log_dict[elem] /= num_objects
            
    elif task == 'next':
        for i in num_feature_ids:
            log_dict[start + num_features_names[i]] /= num_objects
        
        for i in cat_feature_ids:
            log_dict[start + cat_features_names[i]] /= num_objects
            
        for i in [1, 3, 6, 12, 24, 72, 168]:
            log_dict[start + f'interval_{i}_hours'] /= num_objects
                       
    else:
        raise NotImplementedError
    
    log_dict['epoch'] = epoch
    wandb.log(log_dict)
    
    del log_dict['epoch']
    
    return log_dict