import os
import tqdm
import pickle
import wandb
import torch
import torch.nn as nn 

from augmentations import add_noise
from losses import InfoNCELoss
from models import TransactionsRnn, TransactionsModel

from data_generators import batches_generator, transaction_features
from pytorch_training import train_epoch, eval_model

class ContrastiveModel(nn.Module):
    def __init__(self, model, corruption):
        super().__init__()
        self.corruption = corruption 
        self.model = model 

    def forward(self, transaction_features, product):
        corr_transaction_features = self.corruption(transaction_features)
        
        y = self.model(transaction_features, product)
        y_hat = self.model(corr_transaction_features, product)

        return y, y_hat


wandb.init(project="romashka", entity="serofade", group='pretraining')

checkpoint_dir = wandb.run.dir + '/checkpoints'

os.mkdir(checkpoint_dir)

path_to_dataset = '../train_buckets'
dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_generator = batches_generator(dataset_train, batch_size=128, shuffle=True,
                                    device=device, is_train=True, output_format='torch')

with open('./assets/embedding_projections.pkl', 'rb') as f:
    embedding_projections = pickle.load(f)

model = TransactionsModel(transaction_features, embedding_projections, head_type='id').to(device)
contr_model = ContrastiveModel(model, corruption=add_noise)

optimizer = torch.optim.Adam(lr=1e-4, params=contr_model.parameters())
running_loss = 0.0
num_batches = 1
print_loss_every_n_batches = 100
num_epochs = 10

for epoch in range(num_epochs):
    for batch in tqdm.tqdm(train_generator, desc='Training'):
        original_view, corrupted_view = contr_model(batch['transactions_features'], batch['product'])
        original_view, corrupted_view = original_view[:, 0], corrupted_view[:, 0]
        batch_loss = InfoNCELoss(original_view, corrupted_view)

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += batch_loss
        wandb.log({'train_loss': batch_loss.item()})

        if num_batches % print_loss_every_n_batches == 0:
            print(f'Training loss after {num_batches} batches: {running_loss / num_batches}', end='\r')

        torch.save(contr_model.state_dict(), checkpoint_dir + f'/epoch_{epoch}.ckpt')

        num_batches += 1
