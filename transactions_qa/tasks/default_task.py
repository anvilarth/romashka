import torch
import torch.nn as nn
import random

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional)

import transformers
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy

from .task_abstract import AbstractTask

@dataclass
class DefaultTask(AbstractTask):
    tokenizer: transformers.PreTrainedTokenizerBase = None

    def __post_init__(self):
        
        self.task_name = 'default'
        self.target_feature_name = 'amnt'
        self.is_open_ended_task = False  # for a default for this task
        
        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })


        self.question_templates = [
            ("This is the client's transaction history ",
             "Will the client have a credit default? Yes or No?"),
        ]

        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"

        self.answers_options = ["Yes", "No"]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()


        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            new_tokens = [self.transactions_embeddings_start_token,
                          self.transactions_embeddings_end_token]
            if self.task_special_token is not None:
                new_tokens += [self.task_special_token]
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=new_tokens,
                                   special=False)

        self.positive_token = self.tokenizer(self.positive_answer_word).input_ids[0]
        self.negative_token = self.tokenizer(self.negative_answer_word).input_ids[0]

    def generate_target(self, batch: Any, **kwargs) -> Any:
        target_feature_batch = batch['label']  # Tensor [batch_size]

        # Construct target values 
        # Target's questions numeric/categorical answers as str

        target_batch = list(map(lambda x: self.positive_answer_word if x else self.negative_answer_word, (target_feature_batch == 1)))

        return target_batch

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start

        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        transactions_embedding_mask = batch['mask']  # bool Tensor [batch_size, seq_len]
        # Construct target values 
        # Target's questions numeric/categorical answers as str

        target_batch = self.generate_target(batch)

        # Construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        # Encode
        # question_start  -> 'start str <trx>'
        # question_target_bin/question_target_batch  -> '</trx> + end str + OPTIONS:... / Yes or No'
        # target_batch -> feature values as str ('15')

        question_start_tokens = self.tokenizer.encode(question_start,
                                                      return_tensors='pt')[:, :-1].to(device)


        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.tokenizer(question_target_batch,
                                                       padding=True,
                                                       truncation=True,
                                                       return_tensors='pt').to(device)
        # Attention masks
        # already for full batch
        question_start_tokens_mask = torch.ones(question_start_tokens.size()).repeat(batch_size, 1).to(device)
        question_end_tokens_mask = question_target_encoded_batch['attention_mask']

        encoder_input_mask = torch.cat(
                                        [question_start_tokens_mask, 
                                        transactions_embedding_mask, 
                                        question_end_tokens_mask], dim=1
        )

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
                                                                padding=True,
                                                                return_tensors='pt').to(device)


        # target_encoded_batch = [self.tokenizer.encode(target,
        #                                               return_tensors='pt')[:, :-1].to(device)
        #                         for target in target_batch]
        # list of 2d tensors [num_tokens, 1], each token_ids (no eos token!)

        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.tokenizer.encode(self.answer_template,
                                                        return_tensors='pt')[:, :-1].to(device)

        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        batch_answer_encoded = torch.cat([batch_answer_template_encoded,
                                          target_encoded_batch['input_ids']], dim=1).to(device)
        
        batch_answer_template_mask = torch.ones(batch_size, answer_template_encoded.shape[1]).to(device)
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       target_encoded_batch['attention_mask']], dim=1)

        return dict(
            question_start_tokens=question_start_tokens,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            target_tokens=target_encoded_batch['input_ids'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask,
        )

    def process_outputs(self, outputs, answers: torch.Tensor):
        targets = (answers[:, -2] == self.positive_token).long()
        preds = torch.sigmoid(outputs.logits[:, 0, self.positive_token] - outputs.logits[:, 0, self.negative_token])

        return targets, preds

    def calculate_metrics(self, outputs, answers, task_metrics):
        metrics = {}

        targets, preds = self.process_outputs(outputs, answers)

        if 'auc' in task_metrics:
            task_metrics['auc'](preds, targets)
            metrics['auc'] = task_metrics['auc']
        
        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics['accuracy'] = task_metrics['accuracy']

        return metrics 

