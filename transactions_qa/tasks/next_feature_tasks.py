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
from torchmetrics.classification import BinaryAccuracy, Accuracy

from .task_abstract import AbstractTask

from romashka.tools import make_time_batch 

@dataclass
class NextFeatureTask(AbstractTask):
    tokenizer: transformers.PreTrainedTokenizerBase = None

    def __post_init__(self):
        self.task_name = "next_cat_feature_binary"
        self.target_feature_name = 'mcc_category'

        self.threshold = 2
        self.is_open_ended_task = False  # for a default for this task
        
        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })

        self.question_templates = [
            ("This is the client's transaction history ",
             "Will the next transactions have merchant category code 2? Yes or No?"),
        ]

        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"
        self.answers_options = ["Yes", "No"]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]

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

    def generate_target(self, sample: Any, **kwargs) -> Any:
        raise NotImplementedError
    
    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        raise NotImplementedError

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

        target_batch, trx_index = self.generate_target(batch)

        # Masking elements which we want to predict
        transactions_embedding_mask[:, trx_index.flatten()] = 0

        # Construct target sequences
        question_target_batch = self.generate_target_question(question_end, target_batch) # as strings
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
        raise NotImplementedError

    def calculate_metrics(self, outputs, answers, task_metrics):
        raise NotImplementedError

@dataclass
class NextCatFeatureTaskBinary(NextFeatureTask):
    def __post_init__(self):
        super().__post_init__()

    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        return [question_end for _ in range(len(target_batch))] 

    def generate_target(self, batch: Any, **kwargs) -> Any:
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        input_labels = torch.gather(target_feature_batch, 1, trx_index)
        target_batch = list(map(lambda x: 'Yes' if x else 'No', (input_labels == self.threshold)))

        return target_batch, trx_index
    
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

@dataclass
class NextMCCFeatureTaskBinary(NextCatFeatureTaskBinary):
    def __post_init__(self):
        self.task_name = "next_mcc_binary"
        self.target_feature_name = 'mcc_category'
        self.threshold = 2

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "Will the next transactions have merchant category code 2? Yes or No?"),
        ]

@dataclass
class NextNumFeatureTaskBinary(NextFeatureTask):

    def __post_init__(self):
        super().__post_init__()
    
    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        return [question_end for _ in range(len(target_batch))] 

    def generate_target(self, batch: Any, **kwargs) -> Any:
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        input_labels = torch.gather(target_feature_batch, 1, trx_index)
        target_batch = list(map(lambda x: 'Yes' if x else 'No', (input_labels >= self.threshold)))

        return target_batch, trx_index
    
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
  
@dataclass
class NextAmntFeatureTaskBinary(NextNumFeatureTaskBinary):
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "next_num_feature_binary"
        self.target_feature_name = 'amnt'
        self.is_open_ended_task = False  # for a default for this task
        
        self.update_feature_index()

        self.threshold = 0.41
        self.question_templates = [
            ("This is the client's transaction history ",
             f"Will the next transactions have amount more than {self.threshold}? Yes or No?"),
        ]

@dataclass
class NextHourFeatureTaskBinary(NextNumFeatureTaskBinary):
    def  __post_init__(self):
        super().__post_init__()

        self.task_name = "next_hour_binary"
        self.target_feature_name = 'hour_diff'

        self.update_feature_index()
        
        self.question_templates = [
            ("This is the client's transaction history ",
            "Will the next transaction be made in the next 36 hours? Yes or No?"),
        ]
        self.threshold = 36 / 95

@dataclass
class NextTransactions30DaysTaskBinary(AbstractTask):
    def __post_init__(self):
        self.task_name = "next_transactions_30_days_binary"
        self.target_feature_name = 'mcc_category'
        self.N = 30
        self.is_open_ended_task = False  # for a default for this task
        
        self.threshold = 9

        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })

        self.question_templates = [
            ("This is the client's transaction history ",
             "Will there be more than 9 transactions in the next 30 days? Yes or No?"),
        ]

        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"
        self.answers_options = ["Yes", "No"]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]

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
        labels, _, _, padding_mask = make_time_batch(batch, number_days=self.N)
        trx_index = padding_mask.sum(1, keepdim=True) - 1
        
        ### This is the case when we don't have target so we skip this step
        if any(trx_index == -1):
            return None, None

        input_labels = torch.gather(labels, 1, trx_index)
        text_answer = list(map(lambda x: self.positive_answer_word if x else self.negative_answer_word, (input_labels >= self.threshold)))

        return text_answer, trx_index

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

        target_batch, trx_index = self.generate_target(batch)

        # We don't have target so we return empty dict
        if target_batch is None:
            return dict()

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

@dataclass
class NextAmnt30DaysTaskBinary(AbstractTask):
    def __post_init__(self):
        self.task_name = "next_amount_30_days_binary"
        self.target_feature_name = 'mcc_category'
        self.N = 30
        self.is_open_ended_task = False  # for a default for this task
        
        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })

        self.threshold = 2.66
        self.question_templates = [
            ("This is the client's transaction history ",
             "Will there be more transactions of more than 2.66 in the next 30 days? Yes or No?"),
        ]

        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"
        self.answers_options = ["Yes", "No"]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]

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
        _, labels, _, padding_mask = make_time_batch(batch, number_days=self.N)
        trx_index = padding_mask.sum(1, keepdim=True) - 1
        
        ### This is the case when we don't have target so we skip this step
        if any(trx_index == -1):
            return None, None

        input_labels = torch.gather(labels, 1, trx_index)
        text_answer = list(map(lambda x: self.positive_answer_word if x else self.negative_answer_word , (input_labels >= self.threshold)))

        return text_answer, trx_index

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

        target_batch, trx_index = self.generate_target(batch)

        # We don't have target so we return empty dict
        if target_batch is None:
            return dict()

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

@dataclass
class NextCatFeatureTaskMulti(NextFeatureTask):
    def __post_init__(self):

        super().__post_init__()
        self.task_name = "next_cat_feature_multi"
        self.target_feature_name = 'mcc_category'  # 108 unique values
        self.is_open_ended_task = False  # for a default for this task

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "Which merchant category will the next transactions have merchant?"),
        ]

        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.answers_options = ["1"]

        self.answer_template = ""  # left empty for a first time
        self.num_options = 6  # ground truth + 5 additional options

        self.metrics = nn.ModuleDict({
            "accuracy": Accuracy(task='multiclass', num_classes=self.num_options),
        })

    def generate_target(self, batch: Any, **kwargs) -> Any:
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        input_labels = torch.gather(target_feature_batch, 1, trx_index)
        target_batch = list(map(lambda x: str(x.item()), input_labels))

        return target_batch, trx_index

    def generate_target_question(self, question_end, target_batch, **kwargs) -> Any:
        target_batch_options = []  # a list of str target options
        for gt_target in target_batch:
            target_options = {gt_target.item()}
            while len(target_options) < self.num_options:
                target_options.add(random.sample(self.answers_options, k=1)[0])

            # shuffle
            target_options = random.sample(list(target_options), k=len(target_options))
            # connect with separator
            target_options = "".join([self.multichoice_separator % target for target in target_options])
            target_batch_options.append(target_options)

        question_target_batch = [question_end + " OPTIONS:" + target_options for target_options in
                                 target_batch_options]  # as strings

        return question_target_batch

@dataclass
class NextMCCFeatureTaskMulti(NextCatFeatureTaskMulti):
    def __post_init__(self):
        self.task_name = "next_mcc_multi"
        self.target_feature_name = 'mcc_category'

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "Will the next transactions have merchant category code 2? Yes or No?"),
        ]

        self.answers_options = [str(i) for i in range(28)]

@dataclass
class NextNumFeatureTaskMulti(NextFeatureTask):

    tokenizer: transformers.PreTrainedTokenizerBase = None

    def __post_init__(self):

        super().__post_init__()
        self.task_name = "next_num_feature_multi"
        self.target_feature_name = 'amnt'  # 108 unique values
        self.is_open_ended_task = False  # for a default for this task

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "Which merchant category will the next transactions have merchant?"),
        ]

        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.buckets = torch.linspace(0, 1, 100)
        self.answers_options = [str(i) for i in range(28)]

        self.answer_template = ""  # left empty for a first time
        self.num_options = 6  # ground truth + 5 additional options
        self.metrics = nn.ModuleDict({
            "accuracy": Accuracy(task='multiclass', num_classes=self.num_options),
        })

    def generate_target(self, batch: Any, **kwargs) -> Any:
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        input_labels = torch.gather(target_feature_batch, 1, trx_index)
        target_batch = list(map(lambda x: str(x.item()), input_labels))

        return target_batch, trx_index

    def generate_target_question(self, question_end, target_batch, **kwargs) -> Any:
        target_batch_options = []  # a list of str target options
        for gt_target in target_batch:
            target_options = {gt_target.item()}
            while len(target_options) < self.num_options:
                target_options.add(random.sample(self.answers_options, k=1)[0])

            # shuffle
            target_options = random.sample(list(target_options), k=len(target_options))
            # connect with separator
            target_options = "".join([self.multichoice_separator % target for target in target_options])
            target_batch_options.append(target_options)

        question_target_batch = [question_end + " OPTIONS:" + target_options for target_options in
                                 target_batch_options]  # as strings

        return question_target_batch

    def process_outputs(self, outputs, answers: torch.Tensor):
        targets = (answers[:, -2] == self.positive_token).long()
        preds = torch.sigmoid(outputs.logits[:, 0, self.positive_token] - outputs.logits[:, 0, self.negative_token])

        return targets, preds

    def calculate_metrics(self, outputs, answers, task_metrics):
        metrics = {}

        targets, preds = self.process_outputs(outputs, answers)
        
        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics['accuracy'] = task_metrics['accuracy']

        return metrics 