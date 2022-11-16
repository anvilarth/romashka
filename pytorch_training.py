import torch
import wandb
import pandas as pd
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from augmentations import mask_tokens

from data_generators import batches_generator
from losses import NextTransactionLoss, MaskedMSELoss
from clickstream import ClickstreamDataset
from torch.utils.data import DataLoader


def train_epoch(model, optimizer, dataset_train, task='default',  batch_size=64, shuffle=True,
                print_loss_every_n_batches=500, device=None, scheduler=None, process_numerical=False):

    train_generator = batches_generator(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                        device=device, is_train=True, output_format='torch',  process_numerical=process_numerical)
    if task=='default':
        loss_function = nn.BCEWithLogitsLoss()
    
    elif task == 'next':
        loss_function = NextTransactionLoss(process_numerical)

    elif task  ==  'mask':
        loss_function = NextTransactionLoss(process_numerical)
    
    elif task == 'click':
        loss_function = MaskedMSELoss()
        d = ClickstreamDataset('processed.pt', 'click.pt')
        train_generator = DataLoader(d, batch_size=128, shuffle=True)

    else:
        raise NotImplementedError

    num_batches = 1
    running_loss = 0.0

    model.train()

    for batch in tqdm(train_generator, desc='Training'):
        
        if task == 'default':
            output = torch.flatten(model(batch))
            batch_loss = loss_function(output, batch['label'].float())
    
        elif task == 'next':
            mask = batch['transactions_features'][-6] != 0
            output = model(batch)
            if process_numerical:
                batch['transactions_features'][-2] /= 365.
                batch['transactions_features'][-1] = torch.clamp(batch['transactions_features'][-1], max=95)
                batch['transactions_features'][-1] /= 95.
            batch_loss = loss_function(output, batch['transactions_features'], mask[:, :-1])
        
        elif task == 'mask':
            mask1 = batch['transactions_features'][-6] != 0
            corrupted_features, replace_mask = mask_tokens(batch['transactions_features'],  mask1)
            output = model(batch['transactions_features'], batch['product'])
            batch_loss = loss_function(output, batch['transactions_features'])
        
        elif task == 'click':
            batch = list(map(lambda x: x.to(device), batch))
            mask = batch[0] != 0
            output = model(batch[0], mask=mask).squeeze()
            batch_loss = loss_function(output[:, :-1], batch[1][:, 1:], mask[:, :-1])

        else:
            raise NotImplementedError
        
        wandb.log({'train_loss': batch_loss.item()})
        
        batch_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        optimizer.zero_grad()

        running_loss += batch_loss

        if num_batches % print_loss_every_n_batches == 0:
            print(f'Training loss after {num_batches} batches: {running_loss / num_batches}', end='\r')
        
        num_batches += 1
    
    print(f'Training loss after epoch: {running_loss / num_batches}', end='\r')
    

def eval_model(model, dataset_val, task='default', batch_size=32, device=None, process_numerical=False) -> float:
    """
    функция для оценки качества модели на отложенной выборке, возвращает roc-auc на валидационной
    выборке
    :param model: nn.Module модель
    :param dataset_val: путь до директории с последовательностями
    :param batch_size: размер батча
    :param device: device, на который будут положены данные внутри батча
    :return: val roc-auc score
    """

    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      device=device, is_train=True, output_format='torch', process_numerical=process_numerical)
    model.eval()

    num_batches = 0

    if task == 'default':
        preds = []
        targets = []

    elif task == 'click':
        d = ClickstreamDataset('processed.pt', 'click.pt')
        val_generator = DataLoader(d, batch_size=128)
        acc = 0.0
        num_batches = 0
        num_metric = 0
    else:
        acc = 0.0
        num_metric = 0

    with torch.no_grad():
        for batch in tqdm(val_generator, desc='Evaluating model'):
            num_batches += 1
            if task == 'default':
                targets.extend(batch['label'].cpu().numpy().flatten())
                output = model(batch)
                preds.extend(output.cpu().numpy().flatten())

            elif task == 'next':
                targets = torch.stack(batch['transactions_features'])

                output = model(batch)

                if process_numerical:
                    num_output = torch.stack(output[-3:]).squeeze()
                    output = output[:-3]

                    num_targets = targets[-3:]
                    targets = targets[:-3]
                    # tmp_num = torch.stack(output).squeeze()
                    
                    num_targets[-2] /= 365.
                    num_targets[-1] /= 95.
                    
                    not_masked_num_metric = (num_output - num_targets[..., 1:]) ** 2
                    mask = (targets[-6, :, 1:] != 0)
                    num_metric += (not_masked_num_metric * mask).sum(axis=(1,2)) / mask.sum()

                pred = list(map(lambda x: x.argmax(-1), output))
                t_pred = torch.stack(pred)
                mask = (targets[-6, :, 1:] != 0)
                not_masked_acc = (t_pred == targets[..., 1:])
                acc += (not_masked_acc * mask).sum(axis=(1,2)) / mask.sum()

            elif task == 'click':
                batch = list(map(lambda x: x.to(device), batch))
                mask = batch[0] != 0
                output = model(batch[0], mask=mask).squeeze()[:, :-1]

                not_masked_num_metric = (output - batch[1][:, 1:]) ** 2
                num_metric = (output * mask[:, 1:]).sum() / mask.sum()

    if task == 'default':
        return roc_auc_score(targets, preds)
    elif task == 'click':
        return acc / num_batches, [num_metric / num_batches]
    else:  
        output1 = acc / num_batches
        output2 = num_metric / num_batches if process_numerical else [num_metric / num_batches] 
        return output1, output2


def inference(model, dataset_test, batch_size=32, device=None) -> pd.DataFrame:
    """
    функция, которая делает предикты на новых данных, возвращает pd.DataFrame из двух колонок:
    (app_id, score)
    :param model: nn.Module модель
    :param dataset_test: путь до директории с последовательностями
    :param batch_size: размер батча
    :param device: device, на который будут положены данные внутри батча
    :return: pd.DataFrame из двух колонок: (app_id, score)
    """
    model.eval()
    preds = []
    app_ids = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False,
                                       verbose=False, device=device, is_train=False,
                                       output_format='torch')

    for batch in tqdm(test_generator, desc='Test time predictions'):
        app_ids.extend(batch['app_id'])
        output = model(batch['transactions_features'], batch['product'])
        preds.extend(output.detach().cpu().numpy().flatten())
        
    return pd.DataFrame({
        'app_id': app_ids,
        'score': preds
    })
