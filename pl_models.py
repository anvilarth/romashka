import os
import numpy as np
import torch
import torch.nn as nn

import pytorch_lightning as pl

from tools import make_time_batch
from transformers import get_linear_schedule_with_warmup

def transform_labels(label):
    if label.isdigit():
        return int(label)
    return -10000

class TransactionQAModel(pl.LightningModule):
    def __init__(self, language_model, transaction_model, connector, tokenizer, num_days=7):
        super().__init__()
        self.tok = tokenizer
        self.lm_model = language_model
        self.trx_model = transaction_model
        self.connector = connector
        self.num_days = num_days
        self.starting = self.tok.encode("this is the client's transaction history.", return_tensors='pt').cuda()
        self.question = self.tok.encode(f"how many transactions the client will make in the next {num_days} days", return_tensors='pt').cuda()
        self.answer_ending = self.tok.encode('the client will make', return_tensors='pt')[:, :-1].cuda()

    def get_predictions(self, batch, batch_idx=None):
        
        number_of_days = self.num_days
        embedding_prefix = self.lm_model.encoder.embed_tokens(self.starting)
        embedding_question = self.lm_model.encoder.embed_tokens(self.question)
        
        batch_size = batch['mask'].shape[0]

        batch_embedding_prefix = embedding_prefix.repeat(batch_size, 1, 1)
        batch_embedding_question = embedding_question.repeat(batch_size, 1, 1)
        batch_answer_ending = self.answer_ending.repeat(batch_size, 1)

        answer_mask = torch.ones(batch_size, self.answer_ending.shape[1]).cuda()

        _, labels, _, padding_mask = make_time_batch(batch, number_days=number_of_days)
        padding_min = max(1, padding_mask.sum(1).min().item())

        if padding_min == 1:
            return 0.0

        out = self.connector(self.trx_model.get_embs(batch)[0])
        encoder_input = out[:, :padding_min]

        rly_encoder_input = torch.cat([batch_embedding_prefix, encoder_input, batch_embedding_question], dim=1)

        input_labels = labels[:, padding_min - 1]
        target = self.tok.batch_encode_plus(list(map(lambda x: str(x.item()) + ' transactions', input_labels.int())), padding=True, return_tensors='pt')

        torch_labels = target.input_ids.cuda()
        attention_mask = target.attention_mask.cuda()

        answer = torch.cat([batch_answer_ending, torch_labels], dim=1)
        decoder_mask = torch.cat([answer_mask, attention_mask], dim=1)

        outputs = self.lm_model(inputs_embeds=rly_encoder_input, labels=answer, decoder_attention_mask=decoder_mask)
        return outputs, input_labels
        
    def training_step(self, batch, batch_idx=None):
        
        outputs, _ = self.get_predictions(batch, batch_idx)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        batch_size = batch['mask'].shape[0]
        outputs, input_labels = self.get_predictions(batch, batch_idx)
        
        tmp = self.tok.batch_decode(outputs.logits.argmax(2)[:, -4:])
        new_tmp = [i.split(' ')[0] for i in tmp] 
        
        accuracy5 = (abs(torch.tensor(list(map(lambda x: transform_labels(x), new_tmp)), device='cuda') - input_labels) < 5).float().mean()
        accuracy3 = (abs(torch.tensor(list(map(lambda x: transform_labels(x), new_tmp)), device='cuda') - input_labels) < 3).float().mean()
        accuracy1 = (abs(torch.tensor(list(map(lambda x: transform_labels(x), new_tmp)), device='cuda') - input_labels) < 1).float().mean()
        
        self.log('accuracy5', accuracy5, batch_size=batch_size)
        self.log('accuracy3', accuracy3, batch_size=batch_size)
        self.log('accuracy1', accuracy3, batch_size=batch_size)
        self.log('val_loss', outputs.loss, batch_size=batch_size)
        return accuracy1, accuracy5, accuracy3, outputs.loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, 10000, 10000*20)
        return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
        }