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
from augmentations import mixup_data

# torch.autograd.set_detect_anomaly(True)
sys.path.append('./FEDformer')

from data_generators import batches_generator, transaction_features, num_features_names, cat_features_names, meta_features_names
from pytorch_training import train_epoch, eval_model
from torch.utils.data import DataLoader

from data_utils import read_parquet_dataset_from_local
from contrastive_model import PretrainModel
from tools import set_seeds
from losses import NextTransactionLoss

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
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--cutmix', action='store_true')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--head_type', type=str, default='linear')
parser.add_argument('--encoder_type', type=str, default='bert')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--group', type=str, default='pretrain')
parser.add_argument('--numerical', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--train_limitation', type=int, default=10)
parser.add_argument('--freeze_model', action='store_true')
parser.add_argument('--datapath', type=str, default='/home/jovyan/data/alfa/')
parser.add_argument('--data', type=str, default='original')
parser.add_argument('--task', type=str, default='next')
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
rnd_prt = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(12))

run_name = f'{args.model}-{args.num_layers}-emb_mult={args.emb_mult}-mixup={args.mixup}-{args.optimizer}-lr={args.lr}-{rnd_prt}-rel_pos_embs={args.rel_pos_embs}'

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


print("USING ORIGINAL DATA")
path_to_dataset = args.datapath + 'val_buckets'

dir_with_datasets = os.listdir(path_to_dataset)
dataset_val = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

path_to_dataset = args.datapath + 'train_buckets'

dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

dataset_train = dataset_train[:args.train_limitation]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("USING TRANSFORMER")
model = PretrainModel(cat_embedding_projections,
                          cat_features_names,
                          num_embedding_projections,
                          num_features_names,  
                          num_layers=args.num_layers,
                          head_type=args.head_type,
                          encoder_type=args.encoder_type,  
                          mixup=args.mixup,
                          cutmix=args.cutmix,
                          emb_mult=args.emb_mult,
                          alpha=args.alpha,
                          rel_pos_embs=args.rel_pos_embs).to(device)


if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
else:
    raise NotImplementedError

criterion1 = nn.CrossEntropyLoss()
loss_function = NextTransactionLoss()

for epoch in range(num_epochs):
    train_dataloader = batches_generator(dataset_train, batch_size=train_batch_size, shuffle=True,
                                    device=device, is_train=True, output_format='torch')
    
    for i, batch in enumerate(train_dataloader):
        mask = batch['mask']
        original = model.embedding(batch)
        
        batch_size, seq_len = mask.shape[0], mask.shape[1]
        auto_regr_mask = torch.tril(torch.ones(size=(seq_len, seq_len))).unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1).cuda()
        
        original_features = model.encoder(original, mask=auto_regr_mask * mask.unsqueeze(1).unsqueeze(2))
        logit = model.head(original_features)
        loss = loss_function(logit, batch)
        
#         elif i % 2 == 1:
#             original_features = model.encoder(original, mask= mask.unsqueeze(1).unsqueeze(2))
#             corrupted = mixup_data(original, mask=mask, alpha=0.5)
#             corrupted_features = model.encoder(corrupted, mask=mask.unsqueeze(1).unsqueeze(2))
     
#             aug_features_1 = model.mlp(original_features)
#             aug_features_2 = model.mlp(corrupted_features)

#             aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
#             aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)

#             logits_per_aug1 = aug_features_1 @ aug_features_2.t()
#             logits_per_aug2 = aug_features_2 @ aug_features_1.t()

#             targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)

#             loss_1 = criterion1(logits_per_aug1, targets)
#             loss_2 = criterion1(logits_per_aug2, targets)

#             criterion1 = nn.CrossEntropyLoss()

#             loss = (loss_1 + loss_2) / 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 1:
            print(f'Training loss after {i} batches: {loss.item()}', end='\r')
        
        wandb.log({'train_loss_nsp': loss.item()})
#         elif i % 2 == 1:
#             wandb.log({'train_loss_contrastive': loss.item()})
    
    val_dataloader = batches_generator(dataset_val, batch_size=train_batch_size, device=device, is_train=True, output_format='torch')
    val_acc, val_num = eval_model(model, val_dataloader, task=args.task, device=device)
    
    for i, elem in enumerate(cat_features_names):
        wandb.log({f'val_{elem}': val_acc[i]})
        
    for i, elem in enumerate(num_features_names):
        wandb.log({f'val_{elem}': val_num[i]})
    
    print(f"Epoch {epoch} ended")