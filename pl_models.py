import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import random

import pytorch_lightning as pl

from torchmetrics.text.rouge import ROUGEScore
from romashka.tools import make_time_batch
from transformers import get_linear_schedule_with_warmup
from romashka.data_generators import cat_features_names

def transform_labels(label):
    if label.isdigit():
        return int(label)
    return -10000

class TransactionQAModel(pl.LightningModule):
    def __init__(self, language_model, transaction_model, connector, tokenizer, num_days=7, warmup_steps=100):
        super().__init__()

        self.num_days = num_days
        self.warmup_steps = warmup_steps
        
        self.qa_pool = [('next_mcc', '</trx> Will the next transactions have merchang category code 1? Yes or No?', '')]
        
        # self.qa_pool = [('next7_num', 'how many transactions the client will make in the next 7 days', 'the client will make')]
        self.tok = tokenizer
        self.starting = self.tok.encode("This is the client's transaction history <trx>", return_tensors='pt').cuda()
        
        self.save_hyperparameters()
            
        self.rouge = ROUGEScore()
        self.lm_model = language_model
        self.trx_model = transaction_model
        self.connector = connector
        
    def prepare_tokens(self, batch):
        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        question_notok = batch['question']
        answer_notok = batch['answer']
        task = batch['task']

        question = self.tok.encode(question_notok, return_tensors='pt').to(device)
        answer = self.tok.encode(answer_notok, return_tensors='pt')[:, :-1].to(device)

        embedding_prefix = self.lm_model.encoder.embed_tokens(self.starting)
        embedding_question = self.lm_model.encoder.embed_tokens(question)

        batch_embedding_prefix = embedding_prefix.repeat(batch_size, 1, 1)
        batch_embedding_question = embedding_question.repeat(batch_size, 1, 1)
        batch_answer_ending = answer.repeat(batch_size, 1)

        answer_mask = torch.ones(batch_size, answer.shape[1]).to(device)

        return batch_embedding_prefix, batch_embedding_question, batch_answer_ending
    
    def add_qa2transactions(self, batch):
        task_name, question, answer = random.choice(self.qa_pool)
        batch['question'] = question
        batch['answer'] = answer
        batch['task'] = task_name
        
        return batch

    def get_predictions(self, batch, batch_idx=None):
        ### Preparing input
        task = batch['task']
        batch_embedding_prefix, batch_embedding_question, batch_answer_ending = self.prepare_tokens(batch)
        answer_mask = torch.ones(batch_answer_ending.shape[1]).cuda()
        
    
        if task == 'next_mcc':
            trx_index = max(1, batch['mask'].sum(1).min().item()) - 1

            if trx_index == 0:
                return None, None

            input_labels = batch['cat_features'][cat_features_names.index('mcc_category')][:, trx_index]
            text_answer = list(map(lambda x: 'Yes' if x else 'No', (input_labels == 1).cpu()))

            target = self.tok.batch_encode_plus(text_answer, padding=True, return_tensors='pt')
            
        else:
            _, labels, _, padding_mask = make_time_batch(batch, number_days=number_of_days)
            trx_index = max(1, padding_mask.sum(1).min().item())

            input_labels = labels[:, trx_index - 1]
            target = self.tok.batch_encode_plus(list(map(lambda x: str(x.item()) + ' transactions', input_labels.int())), padding=True, return_tensors='pt')
           
            if trx_index == 1:
                return None, None

            ### Passing input to text model
        
        out = self.connector(self.trx_model.get_embs(batch)[0])
        encoder_input = out[:, :trx_index]

        rly_encoder_input = torch.cat([batch_embedding_prefix, encoder_input, batch_embedding_question], dim=1)
        torch_labels = target.input_ids.cuda()
        attention_mask = target.attention_mask.cuda()

        answer = torch.cat([batch_answer_ending, torch_labels], dim=1)
        decoder_mask = torch.cat([answer_mask, attention_mask], dim=1)

        outputs = self.lm_model(inputs_embeds=rly_encoder_input, labels=answer, decoder_attention_mask=decoder_mask)

        return outputs, answer
        
    def training_step(self, batch, batch_idx=None):
        qa_batch = self.add_qa2transactions(batch)
        outputs, _ = self.get_predictions(qa_batch, batch_idx)
        
        if outputs is None:
            return None
        
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss    
    
    def validation_step(self, batch, batch_idx=None):
        batch_size = batch['mask'].shape[0]
        
        qa_batch = self.add_qa2transactions(batch)
        outputs, answer = self.get_predictions(qa_batch, batch_idx)
        
        if outputs is None:
            return None
        
        text_output = self.tok.batch_decode(outputs.logits.argmax(2))
        answer_output = self.tok.batch_decode(answer)
        
#         new_tmp = [i.split(' ')[0] for i in text_output] 
#         tensorized_output = torch.tensor(list(map(lambda x: transform_labels(x), new_tmp)), device='cuda')
        
        
#         accuracy5 = (abs(tensorized_output - input_labels) < 5).float().mean()
#         accuracy3 = (abs(tensorized_output - input_labels) < 3).float().mean()
#         accuracy1 = (abs(tensorized_output - input_labels) < 1).float().mean()
        
#         baseline0 = (abs(0 - input_labels) < 3).float().mean()
#         baseline1 = (abs(1 - input_labels) < 3).float().mean()
#         baseline2 = (abs(2 - input_labels) < 3).float().mean()
#         baseline3 = (abs(3 - input_labels) < 3).float().mean()
        
#         self.log('accuracy5', accuracy5, batch_size=batch_size)
#         self.log('accuracy3', accuracy3, batch_size=batch_size)
#         self.log('accuracy1', accuracy3, batch_size=batch_size)

        wandb.log(self.rouge(text_output, answer_output))
    
        self.log('val_loss', outputs.loss, batch_size=batch_size)
        
        if batch_idx == 0:
            self.logger.log_table("Comparison", columns=['Predicted', 'True'],  data=np.stack([np.array(text_output), np.array(answer_output)], axis=1))

        return outputs.loss #, accuracy1, accuracy5, accuracy3, 
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.warmup_steps, 10000*20)
        
        return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
        }