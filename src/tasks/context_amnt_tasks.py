import torch
import random
import numpy as np

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

import transformers
from torchmetrics.classification import BinaryAccuracy

from .task_abstract import AbstractTask
from romashka.transactions_qa.utils import get_buckets_info

from romashka.data_generators import (transaction_features,
                                      num_features_names)


@dataclass
class MeanAmountBinnedTaskBinary(AbstractTask):
    """
    A task for binned Binary QA task: given a discrete or continuous numeric target - Amount,
    discretize it into bins and answer question with binary answer.
    """

    tokenizer: transformers.PreTrainedTokenizerBase = None
    buckets: Optional[List[float]] = None

    def __post_init__(self):
        self.task_name = "mean_binned_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.target_feature_index = num_features_names.index(self.target_feature_name)
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
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "../../assets/dense_features_buckets.pkl")
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
            new_tokens = [self.transactions_embeddings_start_token,
                          self.transactions_embeddings_end_token]
            if self.task_special_token is not None:
                new_tokens += [self.task_special_token]
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=new_tokens,
                                   special=False)

    @staticmethod
    def _get_buckets_ranges(buckets: List[float],
                            feature_min: Optional[float] = 0.0,
                            feature_max: Optional[float] = 1.0) -> List[Tuple[float, float]]:
        """
        Creates buckets ranges for real number feature value.
        Returns:
            a list of (min, max) tuples for each bucket.
        """
        bucket_min = feature_min
        buckets_ranges = []

        for bucket_max in buckets:
            buckets_ranges.append((round(bucket_min, 3), round(bucket_max, 3)))
            bucket_min = bucket_max

        buckets_ranges.append((round(bucket_max, 3), feature_max))
        return buckets_ranges

    @staticmethod
    def _get_buckets_means(buckets: List[float],
                           feature_min: Optional[float] = 0.0,
                           feature_max: Optional[float] = 1.0) -> List[float]:
        """
        Creates buckets means for real number feature value.
        Returns:
            a list of mean values for each bucket.
        """
        bucket_min = feature_min
        buckets_means = []

        for bucket_max in buckets:
            buckets_means.append((bucket_min + bucket_max) / 2)
            bucket_min = bucket_max

        buckets_means.append(feature_max)
        return buckets_means

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

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS) !!!
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
            question_start_attention_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask,
        )
    
    def calculate_metrics(self, outputs: Any, answers: torch.Tensor, task_metrics: dict, **kwargs) -> dict:
        #TODO: add metrics calculation here
        return {}


@dataclass
class MeanAmountNumericTaskBinary(AbstractTask):
    """
    A task for floating-point Binary QA task: given a continuous numeric target - Amount,
    answer question with a binary answer.
    """

    tokenizer: transformers.PreTrainedTokenizerBase = None
    buckets: Optional[List[float]] = None

    def __post_init__(self):
        self.task_name = "mean_numeric_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
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
        self.num_answers_options = 1000

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "../../assets/dense_features_buckets.pkl")
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
            new_tokens = [self.transactions_embeddings_start_token,
                          self.transactions_embeddings_end_token]
            if self.task_special_token is not None:
                new_tokens += [self.task_special_token]
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=new_tokens,
                                   special=False)

    @staticmethod
    def _get_random_options(num_options: int,
                            feature_min: Optional[float] = 0.0,
                            feature_max: Optional[float] = 1.0,
                            as_strings: Optional[bool] = False) -> List[Union[str, float]]:
        """
        Creates random options list for real number feature value.
        Returns:
            a list of random floating point options.
        """
        options_step = (feature_max - feature_min) / num_options
        answers_options = np.arange(feature_min, feature_max, options_step, dtype=float)
        np.random.shuffle(answers_options)
        if as_strings:
            answers_options = [str(round(opt, 2)) for opt in answers_options]
        return answers_options

    @staticmethod
    def _get_buckets_ranges(buckets: List[float],
                            feature_min: Optional[float] = 0.0,
                            feature_max: Optional[float] = 1.0) -> List[Tuple[float, float]]:
        """
        Creates buckets ranges for real number feature value.
        Returns:
            a list of (min, max) tuples for each bucket.
        """
        bucket_min = feature_min
        buckets_ranges = []

        for bucket_max in buckets:
            buckets_ranges.append((round(bucket_min, 3), round(bucket_max, 3)))
            bucket_min = bucket_max

        buckets_ranges.append((round(bucket_max, 3), feature_max))
        return buckets_ranges

    @staticmethod
    def _get_buckets_means(buckets: List[float],
                           feature_min: Optional[float] = 0.0,
                           feature_max: Optional[float] = 1.0) -> List[float]:
        """
        Creates buckets means for real number feature value.
        Returns:
            a list of mean values for each bucket.
        """
        bucket_min = feature_min
        buckets_means = []

        for bucket_max in buckets:
            buckets_means.append((bucket_min + bucket_max) / 2)
            bucket_min = bucket_max

        buckets_means.append(feature_max)
        return buckets_means

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

        mask_batch = batch['mask']  # bool Tensor [batch_size, seq_len]
        target_feature_batch = batch['num_features'][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            # Calc direct value as float number
            float_feature_ = np.mean([self.buckets_means[bucket_idx.item()] for bucket_idx in feature_masked])
            target_feature_value_batch.append(round(float_feature_, 3)) # get a single Tensor value of a feature

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

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS) !!!
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
            question_start_attention_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask,
        )

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor, task_metrics: dict, **kwargs) -> dict:
        #TODO: add metrics calculation here
        return {}
