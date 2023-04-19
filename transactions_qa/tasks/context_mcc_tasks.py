import torch
import torch.nn as nn
import random

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

import transformers
from torchmetrics import Accuracy
from torchmetrics.text.rouge import ROUGEScore

from .task_abstract import AbstractTask
from .categorical_task_abstract import CategoricalTaskAbstract


@dataclass
class MostFrequentMCCCodeTaskMulti(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "most_frequent_mcc_code_multi"
        self.target_feature_name = 'mcc'  # 108 unique values
        self.task_special_token = "[most_freq_MCC_code_multichoice]"
        self.num_classes = 108
        self.is_open_ended_task = False  # for a default for this task
        self.metrics = nn.ModuleDict({
            "rouge": ROUGEScore(),
            'accuracy': Accuracy(task='multiclass',
                                 threshold=self.decision_threshold,
                                 average='weighted',
                                 ignore_index=self.ignore_class_index,
                                 num_classes=self.num_classes)
        })

        self.question_templates = [
                ("This is the client's transaction history ",
                 ". Which MCC code is the most frequent?"),
                ("This is the client's transaction history ",
                 ". Select the most frequent MCC code."),
                ("You are given the client's transaction history ",
                 ". Choose the most frequent MCC code."),
            ]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.answers_options = [str(i) for i in range(self.num_classes)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True
        self.num_options = 6  # ground truth + 5 additional options
        # self.task_special_tokens = []

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        target_feature_batch = batch['cat_features'][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"), mask=mask_.to("cpu")).long()  # get feature without padding
            codes, cnt = torch.unique(feature_masked, return_counts=True)
            most_freq_feature = codes[torch.argmax(cnt)].long()  # get a single Tensor value of a feature
            target_feature_value_batch.append(most_freq_feature.to(device))

        # Target's questions numeric/categorical answers as str
        target_batch = list(map(lambda x: str(x.item()), target_feature_value_batch))

        target_batch_options = []  # a list of str target options
        for gt_target in target_feature_value_batch:
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

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str + OPTIONS:...'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.tokenizer.encode(question_start,
                                                      return_tensors='pt')
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.tokenizer(question_target_batch,
                                                       padding=True,
                                                       truncation=True,
                                                       return_tensors='pt').to(device)
        # Attention masks
        # already for full batch
        question_start_tokens_mask = torch.ones(question_start_tokens.size()).repeat(batch_size, 1).to(device)
        question_end_tokens_mask = question_target_encoded_batch['attention_mask']
        transactions_embedding_mask = batch['mask']

        encoder_input_mask = torch.cat(
            [question_start_tokens_mask, transactions_embedding_mask, question_end_tokens_mask],
            dim=1)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        # add [:, :-1] for no EOS tokens - ?
        target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
                                                                padding=True,
                                                                return_tensors='pt').to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.tokenizer.encode(self.answer_template,
                                                        return_tensors='pt')[:, :-1].to(device)
        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        # Answer template encoding + target tokens + EOS token
        batch_answer_encoded = torch.cat([batch_answer_template_encoded,
                                          target_encoded_batch['input_ids']], dim=1).to(device)
        # Answer masks
        batch_answer_template_mask = torch.ones(batch_size, answer_template_encoded.shape[1]).to(device)
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       target_encoded_batch['attention_mask']], dim=1)

        return dict(
            question_start_tokens=question_start_tokens,
            question_start_tokens_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask
        )

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
        """
        Calculate task metrics for a task.
        Args:
            outputs: an output from model, can be a tuple of Tensors,
                    a dict with key-value pairs of a single Tensor;
            answers: a Tensor with target values;
            task_metrics: a dictionary (or a torch.nn.ModuleDict) with:
                key - metric name,
                value - a class/function for metric score calculation;

        Returns:
            a dict with:
                key - metric name,
                value - metric score.
        """
        return {}


@dataclass
class MostFrequentMCCCodeTaskBinary(AbstractTask):

    def __post_init__(self):
        self.task_name = "most_frequent_mcc_code_binary"
        self.target_feature_name = 'mcc'  # 108 unique values
        self.task_special_token = "[most_freq_MCC_code_binary]"
        self.num_classes = 108
        self.is_open_ended_task = False  # for a default for this task
        self.metrics = nn.ModuleDict({
            "rouge": ROUGEScore()
        })
        self.question_templates = [
                ("This is the client's transaction history ",
                 ". Is %s MCC code is the most frequent? Yes or No?"),
                ("You are given the client's transaction history ",
                 ". Is %s MCC code is the most frequent? Choose one: Yes or No?"),
                ("This is the client's transaction history ",
                 ". Is the most frequent MCC code of all client's transactions is %s? Yes or No?"),
                ("You are given the client's transaction history ",
                 ". Is it true or false: the most frequent MCC code of all client's transactions is %s? Yes or No?"),
                ("This is the client's transaction history ",
                 ". Define whether the following statement is correct: the most frequent MCC code "
                 "of all client's transactions is %s. Choose: Yes or No?"),
                ("You are given the client's transaction history ",
                 " Identify if the statement that the most frequent MCC code is %s is correct? Yes or No?"),
                ("This is the client's transaction history ",
                 ". Determine whether the following statement is true: %s is the most frequent MCC code "
                 "within all client's transactions. Choose: Yes or No?"),
                ("You are given the client's transaction history ",
                 ". Is the statement correct: the most frequent MCC code of all client's transactions is %s. " 
                 "Choose one: Yes or No?"),
                ("This is the client's transaction history ",
                 ". Answer the question whether or not the following statement is true: the most frequent "
                 "MCC code is %s. Yes or No?"),
                ("You are given the client's transaction history ",
                 ". Find out whether or not the following statement is true: the most frequent MCC code is %s. "
                 "Answer only: Yes or No?"),
            ]

        # all options for a target feature
        self.answers_options: List[str] = [str(i) for i in range(self.num_classes)]
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = "Answer: "  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        target_feature_batch = batch['cat_features'][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"), mask=mask_.to("cpu")).long()  # get feature without padding
            codes, cnt = torch.unique(feature_masked, return_counts=True)
            most_freq_feature = codes[torch.argmax(cnt)].long()  # get a single Tensor value of a feature
            target_feature_value_batch.append(most_freq_feature.to(device))

        # Map to strings
        target_feature_value_batch = list(map(lambda x: str(x.item()), target_feature_value_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = []  # as strings

        # Mask [0/1]
        pos_neg_target_mask = torch.randint(0, 2, (len(target_feature_value_batch),), dtype=torch.int).bool()

        # Target's questions binary [No/Yes]
        target_batch = list(map(lambda x:
                                self.binary_answer_options['positive'] if x
                                else self.binary_answer_options['negative'],
                                pos_neg_target_mask))

        # ground truth target (int/str), mask (bool)
        for target, pos_neg_mask in zip(target_feature_value_batch, pos_neg_target_mask):
            if pos_neg_mask:
                # positive
                question_target_batch.append(question_end % target)
            else:
                # negative
                rand_target = None
                while rand_target is None:
                    opt = random.sample(self.answers_options, k=1)[0]
                    if opt != target:
                        rand_target = opt
                question_target_batch.append(question_end % rand_target)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.tokenizer.encode(question_start,
                                                      return_tensors='pt')
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.tokenizer(question_target_batch,
                                                       padding=True,
                                                       truncation=True,
                                                       return_tensors='pt').to(device)
        # Attention masks
        # already for full batch
        question_start_tokens_mask = torch.ones(question_start_tokens.size()).repeat(batch_size, 1).to(device)
        question_end_tokens_mask = question_target_encoded_batch['attention_mask']
        transactions_embedding_mask = batch['mask']

        encoder_input_mask = torch.cat(
            [question_start_tokens_mask, transactions_embedding_mask, question_end_tokens_mask],
            dim=1)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        # add [:, :-1] for no EOS tokens - ?
        target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
                                                                padding=True,
                                                                return_tensors='pt').to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.tokenizer.encode(self.answer_template,
                                                        return_tensors='pt')[:, :-1].to(device)
        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        # Answer template encoding + target tokens + EOS token
        batch_answer_encoded = torch.cat([batch_answer_template_encoded,
                                          target_encoded_batch['input_ids']], dim=1).long().to(device)
        # Answer masks
        batch_answer_template_mask = torch.ones(batch_size, answer_template_encoded.shape[1]).to(device)
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       target_encoded_batch['attention_mask']], dim=1)

        return dict(
            question_start_tokens=question_start_tokens,
            question_start_tokens_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask
        )

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
        """
        Calculate task metrics for a task.
        Args:
            outputs: an output from model, can be a tuple of Tensors,
                    a dict with key-value pairs of a single Tensor;
            answers: a Tensor with target values;
            task_metrics: a dictionary (or a torch.nn.ModuleDict) with:
                key - metric name,
                value - a class/function for metric score calculation;

        Returns:
            a dict with:
                key - metric name,
                value - metric score.
        """
        return {}

@dataclass
class MostFrequentMCCCodeTaskOpenEnded(AbstractTask):

    def __post_init__(self):
        self.task_name = "most_frequent_mcc_code_open-ended"
        self.target_feature_name = 'mcc'  # 108 unique values
        self.task_special_token = "[most_freq_MCC_code_openended]"
        self.num_classes = 108
        self.is_open_ended_task = True  # for a default for this task
        self.metrics = nn.ModuleDict({
            "rouge": ROUGEScore()
        })
        self.question_templates = [
                ("This is the client's transaction history ",
                 ". Which MCC code is the most frequent?"),
                ("You are given the client's transaction history ",
                 ". Choose the most frequent MCC code."),
                ("This is the client's transaction history ",
                 ". Which MCC code is the most frequent in all clint's transactions?"),
                ("You are given the client's transaction history ",
                 ". Answer the question: which MCC code is the most frequent?"),
                ("This is the client's transaction history ",
                 ". Select the most frequent MCC code."),
                ("You are given the client's transaction history ",
                 ". Answer the question: which MCC code is the most frequent within "
                 "clients' transactions throughout the history?"),
                ("This is the client's transaction history ",
                 ". Find out which MCC code is the most frequent in history?"),
                ("You are given the client's transaction history ",
                 ". Which MCC code is the most frequent within clients' transactions throughout the history?"),
                ("This is the client's transaction history ",
                 ". Identify which MCC code is the most frequent within clients' transactions?"),
                ("You are given the client's transaction history ",
                 ". Can you find out which MCC code is the most frequent?"),
            ]

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(self.num_classes)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        target_feature_batch = batch['cat_features'][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"), mask=mask_.to("cpu")).long()  # get feature without padding
            codes, cnt = torch.unique(feature_masked, return_counts=True)
            most_freq_feature = codes[torch.argmax(cnt)].long()  # get a single Tensor value of a feature
            target_feature_value_batch.append(most_freq_feature.to(device))

        # Target's questions numeric/categorical answers as str
        target_batch = list(map(lambda x: str(x.item()), target_feature_value_batch))

        # Construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str.'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.tokenizer.encode(question_start,
                                                      return_tensors='pt')
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.tokenizer(question_target_batch,
                                                       padding=True,
                                                       truncation=True,
                                                       return_tensors='pt').to(device)
        # Attention masks
        # already for full batch
        question_start_tokens_mask = torch.ones(question_start_tokens.size()).repeat(batch_size, 1).to(device)
        question_end_tokens_mask = question_target_encoded_batch['attention_mask']
        transactions_embedding_mask = batch['mask']

        encoder_input_mask = torch.cat(
                                        [question_start_tokens_mask, 
                                        transactions_embedding_mask, 
                                        question_end_tokens_mask], dim=1
        )

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        # add [:, :-1] for no EOS tokens - ?
        target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
                                                                padding=True,
                                                                return_tensors='pt').to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.tokenizer.encode(self.answer_template,
                                                        return_tensors='pt')[:, :-1].to(device)
        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        # Answer template encoding + target tokens + EOS token
        batch_answer_encoded = torch.cat([batch_answer_template_encoded,
                                          target_encoded_batch['input_ids']], dim=1).to(device)
        # Answer masks
        batch_answer_template_mask = torch.ones(batch_size, answer_template_encoded.shape[1]).to(device)
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       target_encoded_batch['attention_mask']], dim=1)

        return dict(
            question_start_tokens=question_start_tokens,
            question_start_tokens_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask
        )

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
        """
        Calculate task metrics for a task.
        Args:
            outputs: an output from model, can be a tuple of Tensors,
                    a dict with key-value pairs of a single Tensor;
            answers: a Tensor with target values;
            task_metrics: a dictionary (or a torch.nn.ModuleDict) with:
                key - metric name,
                value - a class/function for metric score calculation;

        Returns:
            a dict with:
                key - metric name,
                value - metric score.
        """
        return {}


@dataclass
class ruMostFrequentMCCCodeTaskMulti(MostFrequentMCCCodeTaskMulti):


    def __post_init__(self):
        super().__post_init__()

        self.task_name = "ru_most_frequent_mcc_code_multi"
        self.question_templates = [
            ("Это история транзакций клиента: ",
             ". На ее основе скажи, какой MCC самый частый?"),
            ("Это история транзакций клиента: ",
             ". Выбери на ее основе самый частый MCC код транзакции."),
            ("Дана история транзакций клиента: ",
             ". Исходя из нее, определи, какой был самый частый MCC код транзакции?"),
        ]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.answers_options = [str(i) for i in range(108)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True
        self.num_options = 6  # ground truth + 5 additional options





@dataclass
class ruMostFrequentMCCCodeTaskBinary(MostFrequentMCCCodeTaskBinary):

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "ru_most_frequent_mcc_code_binary"
        self.question_templates = [
            ("Это история транзакций клиента: ",
             ". Правда ли, что %s MCC код транзакции самый частый в истории? Да или Нет?"),
            ("Это история транзакций клиента: ",
             ". Верно ли утверждение, что самый частый в истории MCC код транзакции это - %s? Да или Нет?"),
            ("Дана история транзакций клиента: ",
             ". Исходя из нее, скажи, правда ли что MCC код транзакции - %s самый частый в истории? Да или Нет?"),
        ]
        # all options
        self.answers_options = [str(i) for i in range(108)]
        self.binary_answer_options: Dict[str, str] = {"positive": "Да", "negative": "Нет"}
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True


@dataclass
class ruMostFrequentMCCCodeTaskOpenEnded(MostFrequentMCCCodeTaskOpenEnded):

    def __post_init__(self):
        super().__post_init__()

        self.task_name = "ru_most_frequent_mcc_code_open-ended"
        self.question_templates = [
                ("Это история транзакций клиента: ",
                 ". Какой был самый частый MCC код транзакции?"),
                ("Дана история транзакций клиента: ",
                 ". Исходя из нее, скажи, какой был самый частый MCC код транзакции?"),
                ("Это история транзакций клиента: ",
                 ". На ее основе скажи, какой был самый частый MCC код транзакции?"),
                ("Дана история транзакций клиента: ",
                 ". На ее основе определи самый частый MCC код транзакции."),
            ]

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(108)]
        self.answer_template = ""  # left empty for a first time


