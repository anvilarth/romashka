import os
import pandas as pd
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import wandb
import argparse
import os
import random
import string

from copy import deepcopy
from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)
sys.path.append('./FEDformer')

from data_generators import batches_generator, transaction_features, num_features_names, cat_features_names, meta_features_names
from pytorch_training import train_epoch, eval_model
from torch.utils.data import DataLoader

from data_utils import read_parquet_dataset_from_local
from models import TransactionsModel
from tools import set_seeds, count_parameters
from data import  TransactionClickStreamDataset, TransactionClickStreamDatasetClickstream, TransactionClickStreamDatasetTransactions

from adapter_transformers import UniPELTConfig


os.environ['WANDB_API_KEY'] = 'e1847d5866973dab40f29db28eefb77987d4b66a'

def loading_ptls_model(ckpt_dict):
    new_dict = {}

    encoder_prefix = '_seq_encoder.seq_encoder.encoder.'
    embedding_prefix = '_seq_encoder.trx_encoder.'

    for key in ckpt_dict:
        if encoder_prefix in key:
            new_dict[key[len(encoder_prefix):]] = ckpt[key]

        if embedding_prefix in key:
            new_dict['embedding.' + key[len(embedding_prefix):]] = ckpt[key]
    
    return new_dict


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='transformer')
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--rel_pos_embs', action='store_true')
parser.add_argument('--emb_mult', type=int, default=1)
parser.add_argument('--loss_freq', type=int, default=100)
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--cutmix', action='store_true')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--head_type', type=str, default='linear')
parser.add_argument('--encoder_type', type=str, default='bert')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--group', type=str, default='models')
parser.add_argument('--numerical', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--train_limitation', type=int, default=10)
parser.add_argument('--freeze_model', action='store_true')
parser.add_argument('--datapath', type=str, default='/home/jovyan/data/alfa/')
parser.add_argument('--data', type=str, default='alfa')
parser.add_argument('--task', type=str, default='next')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--focus_feature', type=str, default=None)
parser.add_argument('--add_token', type=str, default='before')
parser.add_argument('--embedding_dim_size', type=int, default=None)
parser.add_argument('--pretrained', action='store_true', default=False)
parser.add_argument('--run_name', type=str, default='')
parser.add_argument('--model_source', type=str, default='scratch')
parser.add_argument('--adapters', action='store_true')

args = parser.parse_args()
rnd_prt = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(12))

if args.run_name == '':
    run_name = f'{args.model}-{args.num_layers}-emb_mult={args.emb_mult}-mixup={args.mixup}-{args.optimizer}-lr={args.lr}-{rnd_prt}-rel_pos_embs={args.rel_pos_embs}'
else:
    run_name = args.run_name

wandb.init(project="romashka", entity="serofade", group=args.group, name=run_name)
wandb.config.update(args)

set_seeds(args.seed)

checkpoint_dir = wandb.run.dir + '/checkpoints'

os.mkdir(checkpoint_dir)

num_epochs = args.num_epochs
train_batch_size = args.batch_size
val_batch_size = args.batch_size

with open('./assets/num_embedding_projections.pkl', 'rb') as f:
    num_embedding_projections = pickle.load(f)
    
with open('./assets/cat_embedding_projections.pkl', 'rb') as f:
    cat_embedding_projections = pickle.load(f)
    
with open('./assets/meta_embedding_projections.pkl', 'rb') as f:
    meta_embedding_projections = pickle.load(f)

if args.data == 'alfa':
    print("USING ALFA DATA")
    path_to_dataset = args.datapath + 'val_buckets'

    dir_with_datasets = os.listdir(path_to_dataset)
    dataset_val = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

    path_to_dataset = args.datapath + 'train_buckets'

    dir_with_datasets = os.listdir(path_to_dataset)
    dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

    dataset_train = dataset_train[:args.train_limitation]
    
    cat_weights=torch.ones(len(cat_features_names))
    num_weights=torch.ones(len(num_features_names))
    
    if args.focus_feature is not None:
        try:
            index = num_features_names.index(args.focus_feature)
            num_weights[index] *= 20
        except:
            pass

        try:
            index = cat_features_names.index(args.focus_feature)
            cat_weights[index] *= 20
        except:
            pass
        
    print("CAT WEIGHTS", cat_weights)
    print("NUM WEIGHTS", num_weights)
        
elif args.data == 'synth':
    print("USING SYNTH DATA")
    dataset_train = TransactionDataset(train=True)
    dataset_val = TransactionDataset(train=False)

elif args.data == 'vtb':
    print("USING VTB DATA")
    dataset_train = TransactionClickStreamDataset('/home/jovyan/afilatov/data/vtb/train_small.pkl')
    dataset_val = TransactionClickStreamDataset('/home/jovyan/afilatov/data/vtb/val_small.pkl')
    
    cat_embedding_projections = dataset_train.cat_embedding_projections
    cat_features_names = dataset_train.features2use

    num_embedding_projections = dataset_train.num_embedding_projections
    num_features_names = dataset_train.num_features2use
    
    meta_embedding_projections = None
    meta_features_names = None

elif args.data == 'vtb_trans':
    dataset_train = TransactionClickStreamDatasetTransactions('/home/jovyan/afilatov/data/vtb/train_small.pkl')
    dataset_val = TransactionClickStreamDatasetTransactions('/home/jovyan/afilatov/data/vtb/val_small.pkl')
    
    cat_embedding_projections = dataset_train.cat_embedding_projections
    cat_features_names = dataset_train.features2use

    num_embedding_projections = dataset_train.num_embedding_projections
    num_features_names = dataset_train.num_features2use
    
    meta_embedding_projections = None
    meta_features_names = None

elif args.data == 'vtb_click':
    dataset_train = TransactionClickStreamDatasetClickstream('/home/jovyan/afilatov/data/vtb/train_small.pkl')
    dataset_val = TransactionClickStreamDatasetClickstream('/home/jovyan/afilatov/data/vtb/val_small.pkl')
    
    cat_embedding_projections = dataset_train.cat_embedding_projections
    cat_features_names = dataset_train.features2use

    num_embedding_projections = dataset_train.num_embedding_projections
    num_features_names = dataset_train.num_features2use
    
    meta_embedding_projections = None
    meta_features_names = None
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
buckets = None 

if args.model == 'transformer':
    print("USING TRANSFORMER")
    model = TransactionsModel(cat_embedding_projections,
                              cat_features_names,
                              num_embedding_projections,
                              num_features_names,
                              meta_embedding_projections,
                              meta_features_names,
                              num_layers=args.num_layers,
                              head_type=args.head_type,
                              encoder_type=args.encoder_type,
                              dropout=args.dropout,
                              mixup=args.mixup,
                              cutmix=args.cutmix,
                              emb_mult=args.emb_mult,
                              alpha=args.alpha,
                              rel_pos_embs=args.rel_pos_embs,
                              add_token=args.add_token,
                              embedding_dim_size=args.embedding_dim_size,
                              pretrained=args.pretrained,
                              adapters=args.adapters,
                             )
    
    wandb.config.update({'parameters': count_parameters(model)})
    
    if args.checkpoint_path != '':
        # ckpt = torch.load('/home/jovyan/romashka/wandb/run-20221205_220219-1nfqlfsm/files/checkpoints/epoch_15.ckpt')
        ckpt = torch.load(args.checkpoint_path)
        
        if args.model_source == 'ptls':
            ckpt = loading_ptls_model(ckpt)
        
        
        model.load_state_dict(ckpt, strict=False)
        
    if args.finetune is not None:
        for param in model.parameters():
            param.requires_grad = False
            
        if args.adapters:
            print("USING ADAPTERS")
            config = UniPELTConfig()
            model.encoder.add_adapter("alfa_battle", config=config)
            model.encoder.train_adapter("alfa_battle")
            
            for p in model.modules():
                if type(p) == nn.LayerNorm:
                    p.requires_grad_(True)

        
        model.cls_token.requires_grad_(True)
        model.head.requires_grad_(True)
        
        if args.finetune == 'all':
            model.requires_grad_(True)
            
        elif args.finetune ==  'embedding':
            model.embedding.requires_grad_(True)
            model.mapping_embedding.requires_grad_(True)
            
        elif args.finetune == 'encoder':
            model.encoder.requires_grad_(True)
            
        elif args.finetune == 'none':
            pass
               
    wandb.config.update({'train_parameters': count_parameters(model)})
    model.to(device)

else:
    raise NotImplementedError

# if args.checkpoint_path != '':
#     ckpt = torch.load(args.checkpoint_path)
#     new_head = deepcopy(model.head)
#     model.head = nn.Identity()
#     model.load_state_dict(ckpt, strict=False)
#     model.head = new_head

print("CREATING OPTIMIZER")

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
else:
    raise NotImplementedError

print("OPTIMIZER DONE")
    
if 'vtb' not in args.data:
    # train_dataloader = batches_generator(dataset_train, batch_size=train_batch_size, shuffle=True,
    #                                 device='cpu', is_train=True, output_format='torch')
    
    epoch_len = 7200 * 128 / args.batch_size 
    num_training_steps = num_epochs * epoch_len
    warmup_steps = int(1e3 * 128 / args.batch_size)
else:
    epoch_len= 138
    num_training_steps = num_epochs * epoch_len
    warmup_steps = int(1e2)
    
if args.scheduler:
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    print("USING scheduler warmup_steps", warmup_steps)
else:
    scheduler = None
    
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    if args.data == 'alfa':
        train_dataloader = batches_generator(dataset_train, batch_size=train_batch_size, shuffle=True,
                                            device=device, is_train=True, output_format='torch')
    else:
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, collate_fn=dataset_train.collate_fn, shuffle=True)

    train_epoch(model, optimizer, train_dataloader, task=args.task, print_loss_every_n_batches=args.loss_freq, device=device, 
                scheduler=scheduler, cat_weights=cat_weights, num_weights=num_weights)
    
    if args.data == 'alfa':
        val_dataloader = batches_generator(dataset_val, batch_size=train_batch_size, device=device, is_train=True, output_format='torch')
    else:
        val_dataloader = DataLoader(dataset_val, batch_size=train_batch_size, collate_fn=dataset_val.collate_fn, shuffle=False)
    
    if args.task == 'default':
        val_roc_auc, _ = eval_model(model, val_dataloader, task=args.task, data=args.data, device=device)
    
    else:
        val_acc, val_num = eval_model(model, val_dataloader, task=args.task, data=args.data, device=device)
    
    if args.data == 'alfa':
        train_dataloader = batches_generator(dataset_train, batch_size=train_batch_size, device=device, is_train=True, output_format='torch')
    
    else:
        train_dataloader = DataLoader(dataset_train, batch_size=train_batch_size, collate_fn=dataset_train.collate_fn, shuffle=False)
    
    if args.task == 'default':
        train_roc_auc, _ = eval_model(model, train_dataloader, task=args.task, data=args.data, device=device)
    else:
        train_acc, train_num = eval_model(model, train_dataloader, task=args.task, data=args.data, device=device)
    
    
    if args.task == 'default':
        wandb.log({'train_roc_auc': train_roc_auc, 'val_roc_auc': val_roc_auc})
    else:
        for i, elem in enumerate(cat_features_names):
            wandb.log({f'train_{elem}': train_acc[i], f'val_{elem}': val_acc[i]})
        
        for i, elem in enumerate(num_features_names):
            wandb.log({f'train_{elem}': train_num[i], f'val_{elem}': val_num[i]})
    
    torch.save(model.state_dict(), checkpoint_dir + f'/epoch_{epoch}.ckpt')
    
    if args.task == 'default':
        print(f'Epoch {epoch+1} completed. Train roc-auc: {train_roc_auc}, Val roc-auc: {val_roc_auc}')