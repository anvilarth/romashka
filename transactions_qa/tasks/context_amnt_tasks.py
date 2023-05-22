import torch
import random
import numpy as np

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

import transformers
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.text.rouge import ROUGEScore

from .task_abstract import AbstractTask
from .numeric_task_abstract import NumericTaskAbstract
from romashka.transactions_qa.utils import get_buckets_info

from romashka.transactions_qa.dataset.data_generator import (
    transaction_features,
    num_features_names
)


@dataclass
class MeanAmountBinnedTaskBinary(NumericTaskAbstract):
    """
    A task for binned Binary QA task: given a discrete or continuous numeric target - Amount,
    discretize it into bins and answer question with binary answer.
    """

    def __post_init__(self):
        self.task_name = "mean_binned_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[mean_binned_AMNT_binary]"
        self.is_open_ended_task = False  # for a default for this task
        self.metrics = {
            "accuracy": BinaryAccuracy()
        }
        self.question_templates = [
            ("This is the client's transaction history: ",
             ". Is the mean value of the amount of all client's transactions "
             "in the range from %s to %s? Yes or No?"),
            ("You are given the client's transaction history: ",
             ". Is it true that the mean value of the amount of all client's transactions "
             "in the range from %s to %s? Yes or No?"),
            ("This is the client's transaction history: ",
             ". Is it true that the mean value of the amount of all client's transactions "
             "is larger then %s but lower then %s? Choose one: Yes or No?")
        ]

        # all options for a target feature
        # self.answers_options: List[str] = [str(i) for i in get_buckets_info(self.target_feature_name,
        #                                                                     "../../assets/dense_features_buckets.pkl")]
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = " "  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available feature value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.answers_options = self._get_buckets_ranges(self.buckets,
                                                        self.feature_min,
                                                        self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.speciaxl_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        # Can be use pre-tokenization: splitting into words/digits/etc.
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        # Can be use pre-tokenization: splitting into words/digits/etc.
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)

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
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "%s")

        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        target_feature_batch = batch['num_features'][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        target_feature_value_range_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.mean([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(float_feature_)  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        target_feature_value_batch = torch.tensor(np.digitize(np.asarray(target_feature_value_batch),
                                                              bins=self.buckets)).to(device)

        # Get buckets ranges of target
        target_feature_value_range_batch = [self.answers_options[target_value]
                                            for target_value in target_feature_value_batch]

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
        for target, target_range, pos_neg_mask in zip(target_feature_value_batch,
                                                      target_feature_value_range_batch,
                                                      pos_neg_target_mask):
            if pos_neg_mask:
                # positive
                question_target_batch.append(question_end % target_range)
            else:
                # negative
                rand_target = None
                while rand_target is None:
                    opt = random.sample(self.answers_options, k=1)[0]
                    if opt != target_range:
                        rand_target = opt
                question_target_batch.append(question_end % rand_target)

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
        # Take only logits for this task
        # predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
        #                                                   skip_special_tokens=True)
        # answers_decoded = self.tokenizer.batch_decode(answers, skip_special_tokens=True)
        #
        # metrics_scores = {}
        # for metric_name, metric_fn in task_metrics.items():
        #     try:
        #         metrics_scores[metric_name] = metric_fn(predictions_decoded, answers_decoded)
        #        self.metrics[metric_name] = task_metrics[metric_name]
        return {}


@dataclass
class MeanAmountNumericTaskBinary(NumericTaskAbstract):
    """
    A task for floating-point Binary QA task: given a continuous numeric target - Amount,
    answer question with a binary answer.
    """

    def __post_init__(self):
        self.task_name = "mean_numeric_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[mean_numeric_AMNT_binary]"
        self.target_feature_index = num_features_names.index(self.target_feature_name)
        self.is_open_ended_task = False  # for a default for this task
        self.metrics = {
            "accuracy": BinaryAccuracy()
        }
        self.question_templates = [
            ("This is the client's transaction history: ",
             ". Is the mean value of the amount of all client's transactions "
             "is %s? Yes or No?"),
            ("You are given the client's transaction history: ",
             ". Is it true that the mean value of the amount of all client's transactions "
             "is %s? Yes or No?"),
            ("This is the client's transaction history: ",
             ". Is it true that the mean value of the amount of all client's transactions "
             "is %s? Choose one: Yes or No?")
        ]

        # all options for a target feature
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.buckets_ranges = self._get_buckets_ranges(self.buckets,
                                                       self.feature_min,
                                                       self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)
        # Note: in this case are not str values!
        self.answers_options = self.buckets_means

        # Or create random options list
        # self.answers_options = self._get_random_options(self.num_answers_options,
        #                                                 self.feature_min,
        #                                                 self.feature_max,
        #                                                 as_strings=True)
        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        # Answer template encoding + strip </s> (EOS) token
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)

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
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "%s")

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.mean([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(round(float_feature_, 3))  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        target_feature_value_bucket_batch = torch.tensor(np.digitize(
            np.asarray(target_feature_value_batch), bins=self.buckets)
        ).to(device)

        # Map to strings
        target_feature_value_batch = list(map(lambda x: str(round(x.item(), 3)), target_feature_value_batch))

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
        for target, target_bucket, pos_neg_mask in zip(target_feature_value_batch,
                                                       target_feature_value_bucket_batch,
                                                       pos_neg_target_mask):
            if pos_neg_mask:
                # positive
                question_target_batch.append(question_end % target)
            else:
                # negative
                rand_target = None
                while rand_target is None:
                    bucket_idx_opt = random.sample(list(range(1, len(self.buckets) + 1)), k=1)[0]
                    if bucket_idx_opt != target_bucket:
                        # as random option get mean value in random bucket (!= target bucket)
                        # Note: buckets are indexed from 1 to N, i.e. [1, N)
                        rand_target = self.answers_options[bucket_idx_opt]
                question_target_batch.append(question_end % str(round(rand_target, 3)))

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
class MeanAmountBinnedTaskOpenEnded(NumericTaskAbstract):
    """
    A task for binned Open-ended QA task: given a discrete or continuous numeric target - Amount,
    discretize it into bins and answer question with bin identifyer.
    """

    def __post_init__(self):
        self.task_name = "mean_binned_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[mean_binned_AMNT_openended]"
        self.target_feature_index = num_features_names.index(self.target_feature_name)
        self.is_open_ended_task = True  # for a default for this task

        self.question_templates = [
            ("This is the client's transaction history: ",
             ". To which range of values does the mean amount of clients' transactions over the whole history belongs to?"),
            ("This is the client's transaction history: ",
             ". To which range of values does the mean value of the amount of clients' transactions belongs to?"),
            ("You are given the client's transaction history: ",
             ". In which range of values falls the mean amount of clients' transactions over the entire history?"),
            ("You are given the client's transaction history: ",
             ". In which range does the mean amount of clients' transactions over history refer to?"),
            ("This is the client's transaction history: ",
             ". In which range does the mean amount of customers' transactions over history refer to?"),
            ("This is the client's transaction history: ",
             ". To which interval does the mean amount of customers' transactions over the entire history belongs?"),
            ("This is the client's transaction history: ",
             ". Which range does the mean amount of clients' transactions belongs to?"),
            ("This is the client's transaction history: ",
             ". Tell me, which range does the mean amount of the clients' transactions belongs to?"),
            ("You are given the client's transaction history: ",
             ". Can you tell which range the mean amount of client transactions belongs to?"),
            ("You are given the client's transaction history: ",
             ". Please indicate the range to which the mean amount of clients' transactions falls into?")
        ]

        # all options for a target feature
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.buckets_ranges = self._get_buckets_ranges(self.buckets,
                                                       self.feature_min,
                                                       self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)
        # Note: in this case are not str values!
        self.answers_options = [str(i) for i in range(1, len(self.buckets) + 1)]

        self.metrics = {
            "accuracy": Accuracy(
                task="multiclass",
                num_classes=len(self.buckets)
            )
        }

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)
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
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        batch_size = batch['mask'].shape[0]

        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "")

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.mean([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(round(float_feature_, 3))  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        target_feature_value_bucket_batch = torch.tensor(np.digitize(
            np.asarray(target_feature_value_batch), bins=self.buckets)
        ).to(device)

        # Map to strings
        target_batch = list(map(lambda x: str(int(x)), target_feature_value_bucket_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
class MeanAmountNumericTaskOpenEnded(NumericTaskAbstract):
    """
    A task for floating-point Open-Ended QA task: given a continuous numeric target - Amount,
    answer question with a numeric answer.
    """

    def __post_init__(self):
        self.task_name = "mean_numeric_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[mean_numeric_AMNT_openended]"
        self.target_feature_index = num_features_names.index(self.target_feature_name)
        self.is_open_ended_task = True  # for a default for this task
        self.metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError()
        }

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". What is the mean amount of clients' transactions?",
            ". Which is the mean amount of clients' transactions?",
            ". Answer the question: what is the mean amount of clients' transactions?",
            ". Answer the question: which is the mean amount of clients' transactions throughout the history?",
            ". Can you please answer the question: what is the mean amount of clients' transactions?",
            ". Identify what was the mean amount of clients' transactions over the whole history?",
            ". Would you answer the question: what is the mean amount of clients' transactions?",
            ". Find out what is the mean amount of clients' transactions?",
            ". Calculate the mean amount of clients' transactions across the entire history?",
            ". Can you calculate the mean amount of transactions of this client?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.buckets_ranges = self._get_buckets_ranges(self.buckets,
                                                       self.feature_min,
                                                       self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)
        # Note: in this case are not str values!
        self.answers_options = [str(i) for i in range(1, len(self.buckets) + 1)]

        # Or create random options list
        # self.answers_options = self._get_random_options(self.num_answers_options,
        #                                                 self.feature_min,
        #                                                 self.feature_max,
        #                                                 as_strings=True)
        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        # target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
        #                                                         padding=True,
        #                                                         return_tensors='pt').to(device)
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)
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
            question_start_tokens_mask=question_start_tokens_mask,  # question_start_attention_mask
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        batch_size = batch['mask'].shape[0]

        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "")

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.mean([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(round(float_feature_, 3))  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        target_feature_value_bucket_batch = torch.tensor(np.digitize(
            np.asarray(target_feature_value_batch), bins=self.buckets)
        ).to(device)

        # Map to strings
        target_batch = list(map(lambda x: str(round(x.item(), 2)), target_feature_value_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
class MinAmountNumericTaskOpenEnded(NumericTaskAbstract):
    """
    A task for floating-point Open-Ended QA task: given a continuous numeric target - Amount,
    answer question with a numeric answer.
    """

    def __post_init__(self):
        self.task_name = "min_numeric_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[min_numeric_AMNT_openended]"
        self.target_feature_index = num_features_names.index(self.target_feature_name)
        self.is_open_ended_task = True  # for a default for this task
        self.metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError()
        }

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". What is the minimum value of the transaction amount occurred throughout the transaction history?",
            ". What is the smallest value of the transaction amount encountered throughout the transaction history?",
            ". Choose the minimum transaction amount that occurred during the whole transaction history.",
            ". Find out what is the minimum transaction amount that occurred during the whole transaction history.",
            ". Can you please answer the question: what is the smallest value of the transaction amount encountered throughout the transaction history?",
            ". Determine the smallest amount of transaction across the entire transactions history?",
            ". Select the smallest transaction amount that encountered across the entire transaction history.",
            ". Choose the minimum amount of transaction that occurred during the history.",
            ". Can you find out which transactions amount is the smallest?",
            ". Answer the question: what is the smallest value of the transaction amount?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.buckets_ranges = self._get_buckets_ranges(self.buckets,
                                                       self.feature_min,
                                                       self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)
        # Note: in this case are not str values!
        self.answers_options = [str(i) for i in range(1, len(self.buckets) + 1)]

        # Or create random options list
        # self.answers_options = self._get_random_options(self.num_answers_options,
        #                                                 self.feature_min,
        #                                                 self.feature_max,
        #                                                 as_strings=True)
        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        # target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
        #                                                         padding=True,
        #                                                         return_tensors='pt').to(device)
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)
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
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        batch_size = batch['mask'].shape[0]

        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "")

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.min([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(round(float_feature_, 3))  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        # target_feature_value_bucket_batch = torch.tensor(np.digitize(
        #     np.asarray(target_feature_value_batch), bins=self.buckets)
        # ).to(device)

        # Map to strings
        target_batch = list(map(lambda x: str(round(x.item(), 2)), target_feature_value_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
class MaxAmountNumericTaskOpenEnded(NumericTaskAbstract):
    """
    A task for floating-point Open-Ended QA task: given a continuous numeric target - Amount,
    answer question with a numeric answer.
    """

    def __post_init__(self):
        self.task_name = "max_numeric_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[max_numeric_AMNT_openended]"
        self.target_feature_index = num_features_names.index(self.target_feature_name)
        self.is_open_ended_task = True  # for a default for this task
        self.metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError()
        }

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". What is the maximum value of the transaction amount occurred throughout the transaction history?",
            ". What is the largest value of the transaction amount encountered throughout the transaction history?",
            ". Choose the maximum transaction amount that occurred during the whole transaction history.",
            ". Find out what is the maximum transaction amount that occurred during the whole transaction history.",
            ". Can you please answer the question: what is the largest value of the transaction amount encountered throughout the transaction history?",
            ". Determine the largest amount of transaction across the entire transactions history?",
            ". Select the largest transaction amount that encountered across the entire transaction history.",
            ". Choose the maximum amount of transaction that occurred during the history.",
            ". Can you find out which transactions amount is the largest?",
            ". Answer the question: what is the largest value of the transaction amount?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.buckets_ranges = self._get_buckets_ranges(self.buckets,
                                                       self.feature_min,
                                                       self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)
        # Note: in this case are not str values!
        self.answers_options = [str(i) for i in range(1, len(self.buckets) + 1)]

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        # target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
        #                                                         padding=True,
        #                                                         return_tensors='pt').to(device)
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)
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
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        batch_size = batch['mask'].shape[0]

        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "")

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.max([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(round(float_feature_, 3))  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        # target_feature_value_bucket_batch = torch.tensor(np.digitize(
        #     np.asarray(target_feature_value_batch), bins=self.buckets)
        # ).to(device)

        # Map to strings
        target_batch = list(map(lambda x: str(round(x.item(), 2)), target_feature_value_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
class LastAmountNumericTaskOpenEnded(NumericTaskAbstract):
    """
    A task for floating-point Open-Ended QA task: given a continuous numeric target - Amount,
    answer question with a numeric answer.
    """

    def __post_init__(self):
        self.task_name = "last_numeric_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[last_numeric_AMNT_openended]"
        self.target_feature_index = num_features_names.index(self.target_feature_name)
        self.is_open_ended_task = True  # for a default for this task
        self.metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError()
        }

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". What is the amount of the last transaction that occurred in history?",
            ". What is the amount of the very last transaction encountered in the transaction history?",
            ". Choose the last transaction amount.",
            ". Select the amount of the most recently occurred transaction.",
            ". Find out what is the amount of last transaction that occurred in history.",
            ". Can you please answer the question: what is the amount of the most recent transaction?",
            ". Determine the amount of the last transaction in history?",
            ". Select the amount of the last transaction that encountered in history.",
            ". Choose the amount of the most recent transaction in the history",
            ". Can you find out of which amount was the most recent transaction?",
            ". Answer the question: what is the amount of the latest transaction?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.buckets_ranges = self._get_buckets_ranges(self.buckets,
                                                       self.feature_min,
                                                       self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)
        # Note: in this case are not str values!
        self.answers_options = [str(i) for i in range(1, len(self.buckets) + 1)]

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run task-specific processing for a full batch of samples.
        Args:
            batch: a dictionary with input data for several samples;
            **kwargs: optional.

        Returns:
            A processed with defined logic batch.
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(batch, question_end=question_end)

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.custom_tokenize(question_start,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
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
        # target_encoded_batch = self.tokenizer.batch_encode_plus(target_batch,
        #                                                         padding=True,
        #                                                         return_tensors='pt').to(device)
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)
        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)
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
            encoder_input_mask=encoder_input_mask,
        )

    def generate_target(self, batch: Any, **kwargs) -> Tuple[List[str], List[str]]:
        """
        Creates target values vector for a batch.
        Args:
            batch: a dict with required for target creation fields;
            **kwargs: **optional**

        Returns: a tuple which contains:
            a question endings - as they (in this task cannot be separated from targets);
            a target values if strings form.
        """
        device = batch['mask'].device
        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        batch_size = batch['mask'].shape[0]

        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "")

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = self.buckets_means[feature_masked[-1]]
            target_feature_value_batch.append(round(float_feature_, 3))  # get a single Tensor value of a feature

        # Convert to corresponding bucket id
        # target_feature_value_bucket_batch = torch.tensor(np.digitize(
        #     np.asarray(target_feature_value_batch), bins=self.buckets)
        # ).to(device)

        # Map to strings
        target_batch = list(map(lambda x: str(round(x, 2)), target_feature_value_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]], **kwargs) -> dict:
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
