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

from data_generators import batches_generator, transaction_features
from pytorch_training import train_epoch, eval_model

from data_utils import read_parquet_dataset_from_local
from models import NextTransactionModel

from dataset_preprocessing_utils import transform_transactions_to_sequences, create_padded_buckets

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt')
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
parser.add_argument('--group', type=str, default='next-transaction')
parser.add_argument('--task', type=str, default='next')

args = parser.parse_args()
rnd_prt = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(12))

run_name = f'{args.model}-{args.num_layers}-emb_mult={args.emb_mult}-mixup={args.mixup}-{args.optimizer}-lr={args.lr}-{rnd_prt}-rel_pos_embs={args.rel_pos_embs}'

wandb.init(project="romashka", entity="serofade", group=args.group, name=run_name)
wandb.config.update(args)

checkpoint_dir = wandb.run.dir + '/checkpoints'

os.mkdir(checkpoint_dir)

path_to_dataset = '../val_buckets'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_val = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

path_to_dataset = '../train_buckets'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('./assets/embedding_projections.pkl', 'rb') as f:
    embedding_projections = pickle.load(f)

num_epochs = args.num_epochs
train_batch_size = 128
val_batch_size = 128

if args.model == 'gpt':
    print("USING GPT-2")
    model = NextTransactionModel(transaction_features, embedding_projections).to(device)

else:
    raise NotImplementedError

if args.checkpoint_path != '':
    ckpt = torch.load(args.checkpoint_path)
    new_head = deepcopy(model.head)
    model.head = nn.Identity()
    new_dict = {}
    for key in ckpt:
        new_dict[key[6:]] = ckpt[key]

    model.load_state_dict(new_dict)
    model.head = new_head


if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
else:
    raise NotImplementedError

    
num_training_steps = num_epochs * 7200
if args.scheduler:
    scheduler = get_linear_schedule_with_warmup(optimizer, int(1e3), num_training_steps)
    print("USING scheduler")
else:
    scheduler = None
    
for epoch in range(num_epochs):
    print(f'Starting epoch {epoch+1}')
    train_epoch(model, optimizer, dataset_train, task=args.task, batch_size=train_batch_size, 
                shuffle=True, print_loss_every_n_batches=args.loss_freq, device=device, scheduler=scheduler)
    
    # val_roc_auc = eval_model(model, dataset_val, batch_size=val_batch_size, device=device)
    
    # train_roc_auc = eval_model(model, dataset_train, batch_size=val_batch_size, device=device)
    # wandb.log({'train_roc_auc': train_roc_auc, 'val_roc_auc': val_roc_auc})
    torch.save(model.state_dict(), checkpoint_dir + f'/epoch_{epoch}.ckpt')
    print(f'Epoch {epoch+1} completed')