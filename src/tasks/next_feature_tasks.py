from tokenize import Exponent
import torch
import torch.nn as nn
import random
import re

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional)

import transformers
import inflect

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import AUROC, MeanAbsoluteError, MeanAbsolutePercentageError
from torchmetrics.classification import BinaryAccuracy, Accuracy
from src.transactions_qa.evaluation.eval_processings_utils import convert_to_numeric 
from src.transactions_qa.utils import transform_for_text_metrics

from .task_abstract import AbstractTask

from src.utils.tools import make_time_batch
from src.transactions_qa.utils import text2float

@dataclass
class NextFeatureTask(AbstractTask):
    tokenizer: transformers.PreTrainedTokenizerBase = None

    def __post_init__(self):
        self.task_name = "next_feature"
        # TODO: remove this plug
        self.target_feature_name = 'mcc_category'

        self.threshold = 2
        self.is_open_ended_task = False  # for a default for this task
        
        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:",
            "The following is the client's transaction history:",
            "The client's transaction history is listed below:"
        ]
        self.ending_prompts = ["Test prompt"]
        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"
        self.answers_options = ["Yes", "No"]
        self.numerical_token = "<NUM>"
        self.val_token = " <VAL>" if self.use_numerical_output else ""
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]

        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

        if self.task_type == 'text':
            if self.tokenizer is None:
                raise AttributeError("This task requires tokenizer to be set!")
            if self.add_tokens_to_tokenizer:
                new_tokens = [self.transactions_embeddings_start_token,
                            self.transactions_embeddings_end_token]
                if self.use_numerical_output:
                   new_tokens.append(self.numerical_token)
                   new_tokens.append(self.val_token)

                if self.task_special_token is not None:
                    new_tokens += [self.task_special_token]
                self.extend_vocabulary(tokenizer=self.tokenizer,
                                    new_tokens=new_tokens,
                                    special=False)
                                    
            self.positive_token = self.tokenizer(self.positive_answer_word).input_ids[0]
            self.negative_token = self.tokenizer(self.negative_answer_word).input_ids[0]
    
    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start

        question_start = question_start + self.transactions_embeddings_start_token

        question_end = self.transactions_embeddings_end_token + question_end

        batch_size = batch['mask'].shape[0]

        ### Constructing batch of questions

        batch_question_start = [question_start] * batch_size

        # Construct target values 
        # Target's questions numeric/categorical answers as str

        task_batch = self.generate_text_target(batch)

        if not task_batch:
            return dict()

        target_batch = task_batch['text_label']
        if self.use_numerical_output:
            target_batch = [self.numerical_token] * len(target_batch)

        transactions_embedding_mask = task_batch['mask']  # bool Tensor [batch_size, seq_len]
        # Masking elements which we want to predict

        # Construct target sequences
        question_target_batch = self.generate_target_question(question_end, task_batch) # as strings

        # Constructing batch of answer templates
        answer_template = [self.answer_template] * batch_size

    
        return dict(
            mask=transactions_embedding_mask,
            question_start_string=batch_question_start,
            answer_target_string=target_batch,
            answer_start_string=answer_template,
            question_end_string=question_target_batch,
            raw_labels=task_batch['label'],
        )

    def process_outputs(self, outputs, answers: torch.Tensor):
        raise NotImplementedError

    def calculate_metrics(self, outputs, answers, task_metrics):
        raise NotImplementedError

@dataclass
class NextCatFeatureTaskBinary(NextFeatureTask):
    def __post_init__(self):
        super().__post_init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        batch['label'] = (torch.gather(target_feature_batch, 1, trx_index) == self.threshold).float().squeeze(1)
        batch['mask'][:, trx_index.flatten()] = 0
        return batch

    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        target_batch = target_batch['text_label']
        return [question_end for _ in range(len(target_batch))] 

    def generate_text_target(self, batch: Any, **kwargs) -> Any:

        batch = self.prepare_task_batch(batch, **kwargs)
        if not batch:
            return {}
        
        input_labels = batch['label']
        batch['text_label'] = list(map(lambda x: self.positive_answer_word if x else self.negative_answer_word, input_labels))
        return batch
    
    def process_outputs(self, outputs, answers: torch.Tensor):
        targets = (answers[:, 0] == self.positive_token).long()
        preds = torch.sigmoid(outputs.logits[:, 0, self.positive_token] - outputs.logits[:, 0, self.negative_token])

        return preds, targets 

    def calculate_metrics(self, outputs, answers, task_metrics, stage):
        metrics = {}

        if self.task_type == 'text':
            preds, targets = self.process_outputs(outputs, answers)
        else:
            preds, targets = torch.sigmoid(outputs), answers

        if 'auc' in task_metrics:
            task_metrics['auc'](preds, targets)
            metrics[stage + self.task_name + '_auc'] = task_metrics['auc']
        
        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics[stage + self.task_name + '_accuracy'] = task_metrics['accuracy']

        return metrics 

@dataclass
class DefaultTaskBinary(NextCatFeatureTaskBinary):
    def __post_init__(self):
        super().__post_init__()
        
        self.task_name = "default"
        self.target_feature_name = 'mcc_category'
        self.threshold = 2

        self.update_feature_index()

        self.ending_prompts = [
            ". Will the client have a default?",
            ". The client will have a default, right?",
            ". The client is going to have a default?",
        ]
        self.question_templates = self.generate_question_templates(self.starting_prompts, self.ending_prompts)
    
    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        batch['label'] = batch['label'].float()
        return batch

@dataclass
class NextMCCFeatureTaskBinary(NextCatFeatureTaskBinary):
    def __post_init__(self):
        super().__post_init__()
        
        self.task_name = "next_mcc_binary"
        self.target_feature_name = 'mcc_category'
        self.threshold = 2

        self.update_feature_index()

        self.ending_prompts = [
            ". Determine if the next transaction will have an MCC code 2?",
            ". Will the next transactions have merchant category code 2? Yes or No?",
            ". Will the next transactions have merchant category code 2? Choose one: Yes or No",
            ". Is the upcoming transaction going to have merchant category code 2?",
            ". Find out whether or not the following statement is true: the next transactions MCC code is 2. Answer only: Yes or No?",
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts, self.ending_prompts)

@dataclass
class NextNumFeatureTaskBinary(NextFeatureTask):

    def __post_init__(self):
        super().__post_init__()
        self.range = (0.0, 1.0)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        batch_size = len(batch['mask'])
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1

        threshold = torch.tensor([self.threshold] * batch_size,  device=batch['mask'].device).unsqueeze(1)
        
        if self.floating_threshold:
            threshold = torch.rand(batch['mask'].shape[0], 1, device=batch['mask'].device) * (self.range[1] - self.range[0]) + self.range[0]
            threshold = torch.round(threshold, decimals=2)

        batch['label'] = (torch.gather(target_feature_batch, 1, trx_index) >= threshold).float().squeeze(1)
        batch['mask'][:, trx_index.flatten()] = 0
        batch['threshold'] = list(map(lambda x : str(round(x.item(), 2)), threshold))

        return batch

    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        batch_size = target_batch['mask'].shape[0]

        if 'threshold' in target_batch:
            question_end = [re.sub("<extra_id_0>", target_batch['threshold'][i], question_end) for i in range(batch_size)]
        else:
            raise NotImplementedError
        return question_end

    def generate_text_target(self, batch: Any, **kwargs) -> Any:

        batch = self.prepare_task_batch(batch, **kwargs)
        input_labels = batch['label']
        batch['text_label'] = list(map(lambda x: self.positive_answer_word if x else self.negative_answer_word, input_labels))
        return batch
    
    def process_outputs(self, outputs, answers: torch.Tensor):
        targets = (answers[:, 0] == self.positive_token).long()
        preds = torch.sigmoid(outputs.logits[:, 0, self.positive_token] - outputs.logits[:, 0, self.negative_token])

        return preds, targets 

    def calculate_metrics(self, outputs, answers, task_metrics, stage):
        metrics = {}

        preds, targets = self.process_outputs(outputs, answers)

        if 'auc' in task_metrics:
            task_metrics['auc'](preds, targets)
            metrics[stage + self.task_name + '_auc'] = task_metrics['auc']
        
        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics[stage + self.task_name + '_accuracy'] = task_metrics['accuracy']

        return metrics 
  
@dataclass
class NextAmntFeatureTaskBinary(NextNumFeatureTaskBinary):
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "next_amnt_binary"
        self.target_feature_name = 'amnt'
        self.is_open_ended_task = False  # for a default for this task
        
        self.update_feature_index()

        self.threshold = 0.41

        self.ending_prompts = [
            f". Determine if the next transaction will have amount more than <extra_id_0>?",
            f". Will the next transactions have amount more than <extra_id_0>? Yes or No?",
            f". Will the next transactions have amount more than <extra_id_0>? Choose one: Yes or No",
            f". Is the upcoming transaction going to have amount more than <extra_id_0>?",
            f". Find out whether or not the following statement is true: the next transactions amount is more than <extra_id_0>. Answer only: Yes or No?",
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts, self.ending_prompts)

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
class NextTransactions30DaysTaskBinary(NextCatFeatureTaskBinary):
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "next_transactions_30_days_binary"
        self.target_feature_name = 'mcc_category'
        self.N = 30
        self.is_open_ended_task = False  # for a default for this task
        
        self.threshold = 6

        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })

        self.question_templates = [
            ("This is the client's transaction history ",
             f"Will there be more than {self.threshold} transactions in the next 30 days? Yes or No?"),
        ]

        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"
        self.answers_options = ["Yes", "No"]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]

        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        if self.task_type == 'text':
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

    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        _, labels, _, padding_mask = make_time_batch(batch, number_days=self.N)
        trx_index = padding_mask.sum(1, keepdim=True) - 1

        if any(trx_index == -1):
            return {}

        batch['label'] = (torch.gather(labels, 1, trx_index) >= self.threshold).float().squeeze(1)
        batch['mask'] = padding_mask
        return batch

@dataclass
class NextAmnt30DaysTaskBinary(NextNumFeatureTaskBinary):
    def __post_init__(self):
        super().__post_init__()
        self.task_name = "next_amount_30_days_binary"
        self.target_feature_name = 'mcc_category'
        self.N = 30
        self.is_open_ended_task = False  # for a default for this task
        
        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })

        self.threshold = 2.27
        self.ending_prompts = [
            f". Determine if in 30 days the next transaction will have amount more than {self.threshold}?",
            f". Will the next transactions have amount more than {self.threshold}? Yes or No?",
            f". Will the next transactions have amount more than {self.threshold}? Choose one: Yes or No",
            f". Is the upcoming transaction going to have amount more than {self.threshold}?",
            f". Find out whether or not the following statement is true: the next transactions amount is more than {self.threshold}. Answer only: Yes or No?",
        ]
        
        self.question_templates = self.generate_question_templates(self.starting_prompts, self.ending_prompts)

        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"
        self.answers_options = ["Yes", "No"]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]

        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True



        if self.task_type == 'text':
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
    
    def prepare_task_batch(self, batch):
        labels, _, _, padding_mask = make_time_batch(batch, number_days=self.N)
        trx_index = padding_mask.sum(1, keepdim=True) - 1
        
        ### This is the case when we don't have target so we skip this step
        if any(trx_index == -1):
            return {}

        input_labels = (torch.gather(labels, 1, trx_index) >= self.threshold).float().squeeze(1)
        batch['label'] = input_labels
        batch['mask'] = padding_mask
        return batch

    def generate_text_target(self, batch: Any, **kwargs) -> Any:
        task_batch = self.prepare_task_batch(batch, **kwargs)
        if not task_batch:
            return None

        input_labels = batch['label']
        batch['text_label'] = list(map(lambda x: 'Yes' if x else 'No', input_labels))
        return batch

@dataclass
class NextCatFeatureTaskMulti(NextFeatureTask):
    def __post_init__(self):

        super().__post_init__()
        self.criterion = nn.CrossEntropyLoss()
        self.task_name = "next_cat_feature_multi"
        self.target_feature_name = 'mcc_category'  # 28 unique values
        self.is_open_ended_task = False  # for a default for this task
        self.num_classes = 2

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "Which merchant category will the next transactions have merchant?"),
        ]

        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.answers_options = [str(i) for i in range(self.num_classes)]

        self.answer_template = ""  # left empty for a first time
        self.num_options = 6  # ground truth + 5 additional options

        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task='multiclass',
                                 average='weighted',
                                 num_classes=self.num_classes)
        })

    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        batch['label'] = torch.gather(target_feature_batch, 1, trx_index).squeeze(1)
        batch['mask'][:, trx_index.flatten()] = 0
        return batch

    def generate_text_target(self, batch: Any, **kwargs) -> Any:
        batch = self.prepare_task_batch(batch, **kwargs)
        if not batch:
            return {}

        input_labels = batch['label']
        batch['text_label'] = list(map(lambda x: str(x.item()), input_labels))
        return batch

    def filter_range(self, value):
        return value if 0 <= value <= self.num_classes else -1 

    def generate_target_question(self, question_end, target_batch) -> Any:
        target_batch = target_batch['text_label']
        target_batch_options = []  # a list of str target options
        for gt_target in target_batch:
            target_options = {gt_target}
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
        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
                                                          skip_special_tokens=True)
        
        answers_decoded = self.tokenizer.batch_decode(answers, skip_special_tokens=True)
        processed_answers =  torch.tensor(list(map(lambda x: convert_to_numeric(x, -1, verbose=False), answers_decoded)), device=answers.device)

        processed = torch.tensor(list(map(lambda x: self.filter_range(convert_to_numeric(x, -1, verbose=False)), predictions_decoded)), device=answers.device)
        
        return processed, processed_answers

    def calculate_metrics(self, outputs, answers, task_metrics, stage):
        batch_size = answers.shape[0]
        device = answers.device
        metrics = {}
        if self.task_type == 'text':
            preds, targets = self.process_outputs(outputs, answers)
        else:
            indices = torch.cat([torch.randint(0, self.num_classes, size=(batch_size, self.num_options - 1), device=device), answers.unsqueeze(1)], dim=1)
            preds = outputs.gather(1, indices).argmax(1, keepdim=True)
            preds_processed = torch.take_along_dim(indices, preds, dim=1)
            preds, targets = preds_processed.squeeze(1), answers
        
        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics[stage + self.task_name + '_accuracy'] = task_metrics['accuracy']

        return metrics 

@dataclass
class NextMCCFeatureTaskMulti(NextCatFeatureTaskMulti):
    def __post_init__(self):
        
        super().__post_init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.task_name = "next_mcc_multi"
        self.target_feature_name = 'mcc_category'
        self.num_classes = 28

        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task='multiclass',
                                 average='weighted',
                                 num_classes=self.num_classes,
                                 ignore_index=0)
        })

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "What merchant category will the next transactions have?"),
        ]

        self.answers_options = [str(i) for i in range(self.num_classes)]

@dataclass
class NextNumTransactionTaskMulti(NextCatFeatureTaskMulti):
    def __post_init__(self):
        
        super().__post_init__()
        self.task_name = "next_num_30days_multi"
        self.target_feature_name = 'mcc_category'
        self.num_classes = 200 
        self.N = 30

        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task='multiclass',
                                 average='weighted',
                                 num_classes=self.num_classes, # Adding fake class to calculate
                                 ignore_index=0)
        })

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "How many transactions will be in the next 30 days?"),
        ]

        self.answers_options = [str(i) for i in range(self.num_classes)]

    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        _, labels, _, padding_mask = make_time_batch(batch, number_days=self.N)
        trx_index = padding_mask.sum(1, keepdim=True) - 1

        if any(trx_index == -1):
            return {}
        # TODO: fix adding 1 (bincount error with negative values)
        # clipping in range [1, self.num_classes]
        batch['label'] = torch.clamp(torch.gather(labels, 1, trx_index), 0, self.num_classes - 2).squeeze(1).long() + 1
        batch['mask'] = padding_mask
        return batch

@dataclass
class NextFeatureOpenEnded(NextFeatureTask):
    def __post_init__(self):
        super().__post_init__()
        self.num_classes = -1
        self.criterion = nn.CrossEntropyLoss()
        self.default_value = -1
        self.num2word_engine = inflect.engine()
    
    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        target_feature_batch = batch[self.target_feature_type][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values 
        # Target's questions numeric/categorical answers as str
        trx_index = batch['mask'].sum(1, keepdim=True) - 1
        batch['label'] = self.process_labels(torch.gather(target_feature_batch, 1, trx_index).squeeze(1))
        batch['mask'][:, trx_index.flatten()] = 0
        return batch
    
    def process_labels(self, labels):
        return labels

    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        target_batch = target_batch['text_label']
        return [question_end for _ in range(len(target_batch))] 

    def generate_text_target(self, batch: Any, **kwargs) -> Any:
        batch = self.prepare_task_batch(batch, **kwargs)
        if not batch:
            return {}

        input_labels = batch['label']
        batch['text_label'] = list(map(lambda x: str(round(x.item(), 2)), input_labels))
        if self.answer2text:
            batch['text_label'] = list(map(lambda x:self.num2word_engine.number_to_words(x, decimal='dot') , batch['text_label']))
        return batch

    def filter_range(self, value):
        return value if 0 <= value <= self.num_classes else self.default_value 
    
    def process_num_outputs(self, outputs, answers: torch.Tensor):
        processed_answers = outputs['label']
        processed = outputs['preds']
        
        return processed, processed_answers
    
    def process_outputs(self, outputs, answers: torch.Tensor):
        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
                                                          skip_special_tokens=True)

        answers_decoded = self.tokenizer.batch_decode(answers, skip_special_tokens=True)
        if self.answer2text:
            processed_answers =  torch.tensor(list(map(lambda x: self.filter_range(text2float(x, default_value=self.default_value)), answers_decoded)), device=answers.device)
            processed = torch.tensor(list(map(lambda x: self.filter_range(text2float(x, default_value=self.default_value)), predictions_decoded)), device=answers.device)
        else:
            processed_answers =  torch.tensor(list(map(lambda x: self.filter_range(convert_to_numeric(x, self.default_value)), answers_decoded)), device=answers.device)
            processed = torch.tensor(list(map(lambda x: self.filter_range(convert_to_numeric(x, self.default_value)), predictions_decoded)), device=answers.device)
    
        return processed, processed_answers

    def calculate_metrics(self, outputs, answers, task_metrics, stage):
        metrics = {}

        if self.task_type == 'text':
            if self.use_numerical_output:
                preds, targets = self.process_num_outputs(outputs, answers)
            else:
                preds, targets = self.process_outputs(outputs, answers)
        else:
            preds, targets = torch.sigmoid(outputs), answers

        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics[stage + self.task_name + '_accuracy'] = task_metrics['accuracy']

        if 'mae' in task_metrics:
            task_metrics['mae'](preds, targets)
            metrics[stage + self.task_name + '_mae'] = task_metrics['mae']

        if 'mape' in task_metrics:
            task_metrics['mape'](preds, targets)
            metrics[stage + self.task_name + '_mape'] = task_metrics['mape']

        # if 'rouge' in task_metrics:
        #     text_preds, text_targets = transform_for_text_metrics(preds, targets)
        #     task_metrics['rouge'](text_preds, text_targets)
        #     metrics[stage + self.task_name + '_rouge'] = task_metrics['rouge']

        return metrics 

@dataclass
class NextMCCFeatureOpenEnded(NextFeatureOpenEnded):
    def __post_init__(self):
        super().__post_init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        self.task_name = "next_mcc_open_ended"
        self.target_feature_name = 'mcc_category'
        self.num_classes = 28

        self.metrics = nn.ModuleDict({
            "accuracy": Accuracy(task='multiclass',
            num_classes=self.num_classes,
            average='weighted',
            ignore_index=0)
        })

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "What merchant category code will the next transactions have?"),
        ]

@dataclass
class NextNumTransactionTaskOpenEnded(NextFeatureOpenEnded):
    def __post_init__(self):
        
        super().__post_init__()
        self.task_name = "next_num_30days_open_ended"
        self.target_feature_name = 'mcc_category'
        self.num_classes = 200 # Adding fake class to calculate
        self.N = 30
        self.default_value = 0

        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task='multiclass',
                                 average='weighted',
                                 num_classes=self.num_classes,
                                 ignore_index=0)
        })

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "How many transactions will be in the next 30 days?"),
        ]

        self.answers_options = [str(i) for i in range(self.num_classes)]
    
    def process_labels(self, labels):
        return torch.clamp(labels, 0, self.num_classes - 2).squeeze(1).long() + 1

    def prepare_task_batch(self, batch: Dict[str, Any], **kwargs):
        _, labels, _, padding_mask = make_time_batch(batch, number_days=self.N)
        trx_index = padding_mask.sum(1, keepdim=True) - 1

        if any(trx_index == -1):
            return {}
        # TODO: fix adding 1 (bincount error with negative values)
        batch['label'] = self.process_labels(torch.gather(labels, 1, trx_index))
        batch['mask'] = padding_mask
        return batch

@dataclass
class NextAmntOpenEnded(NextFeatureOpenEnded):
    def __post_init__(self):
        super().__post_init__()

        self.default_value = 0

        self.metrics = nn.ModuleDict({
            "mae": MeanAbsoluteError(),
            "mape": MeanAbsolutePercentageError(),            
            "rouge": ROUGEScore(),
        })
        self.criterion = nn.L1Loss()
        
        self.task_name = "next_amnt_open_ended"
        self.target_feature_name = 'amnt'

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "The task is to predict on what amount will the client make the next transactions? The answer must be in range from 0 to 1" + self.val_token),
        ]

    def filter_range(self, value):
        return round(value, 2)

    def process_labels(self, labels):
        return torch.round(labels, decimals=2)
    
    def process_num_outputs(self, outputs, answers: torch.Tensor):
        processed_answers = outputs['label']
        processed = outputs['preds']
        
        return processed, processed_answers

@dataclass
class NextHourOpenEnded(NextFeatureOpenEnded):
    def __post_init__(self):
        super().__post_init__()

        self.default_value = 0

        self.metrics = nn.ModuleDict({
            "mae": MeanAbsoluteError(),
            "mape": MeanAbsolutePercentageError(),
            "rouge": ROUGEScore(),
        })
        self.criterion = nn.L1Loss()
        
        self.task_name = "next_hour_open_ended"
        self.target_feature_name = 'hour_diff'

        self.update_feature_index()

        self.question_templates = [
            ("This is the client's transaction history ",
             "The task is to predict how many hours until the client will make the next transaction? The answer is should in range from 0 to 40" + self.val_token),
        ]
    
    def filter_range(self, value):
        return round(value, 2)

    def process_labels(self, labels):
        return torch.round(labels, decimals=2)
    
    def process_num_outputs(self, outputs, answers: torch.Tensor):
        processed_answers = outputs['label']
        processed = outputs['preds']
        
        return processed, processed_answers