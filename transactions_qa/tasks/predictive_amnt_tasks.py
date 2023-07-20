import torch
import random
import numpy as np

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

from torchmetrics import Perplexity
from torchmetrics.classification import BinaryAccuracy, Accuracy

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from .numeric_task_abstract import NumericTaskAbstract
from romashka.transactions_qa.utils import get_buckets_info
from romashka.transactions_qa.evaluation.eval_processings_utils import (float_splitter,
                                                                        make_float,
                                                                        transform_labels)


@dataclass
class PredNumericAmountTaskBinary(NumericTaskAbstract):
    """
    A task for predictive exact Binary QA task: given a discrete or continuous numeric target - Amount,
    answer question with binary answer.
    """

    def __post_init__(self):
        self.task_name = "pred_numeric_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_numeric_AMNT_binary]"

        self.is_text_task = False
        self.is_binary_task = True
        self.is_open_ended_task = False

        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_outputs: Optional[bool] = False

        self.metrics = {
            "accuracy": BinaryAccuracy()
        }
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". Will the amount of the next transaction be equal to %s? Yes or No?",
            ". Will the next transaction amount be equal to %s? Choose one: Yes or No?",
            ". Is it true that the amount of next transaction will be equal to %s? Yes or No?",
            ". Define whether the following statement is true: in next transaction amount will be equal to %s. "
            "Choose: Yes or No?",
            ". Is it true or false: the amount of the next transaction will be %s? Yes or No?",
            ". Define whether the following statement is correct: in the next transaction amount will be %s. "
            "Choose: Yes or No?",
            ". Identify if the statement that: the amount of the next transaction will be equal to %s, "
            "is correct? Yes or No?",
            ". Determine whether the following statement is true: %s will be the amount of the next transaction"
            ". Choose: Yes or No?",
            ". Is the statement correct: the amount of the next transaction will be %s. "
            "Answer with one of the following options: Yes or No?",
            ". Answer the question whether or not the following statement is true: the amount of the next "
            "transaction will be equal to %s. Yes or No?",
            ". Answer the question: will the amount of the next transaction be equal to %s? "
            "Choose only one of the following options: Yes or No?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

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
                                   special=True)

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
            with_numeric_input=self.numeric_inputs,
            with_numeric_output=self.numeric_outputs
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
            last_feature_index = mask_.sum() - 1
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            last_feature = feature_masked[-1]  # get a single Tensor value of a feature
            float_feature_ = self.buckets_means[last_feature.item()]  # take a mean bucket value of the last feature
            target_feature_value_batch.append(float_feature_)
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # Convert to corresponding bucket id
        target_feature_value_bucket_batch = torch.tensor(np.digitize(
            np.asarray(target_feature_value_batch), bins=self.buckets)
        ).to(device)

        # Map to strings
        target_feature_value_batch = list(map(lambda x: str(round(x, 3)), target_feature_value_batch))

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
                    bucket_idx_opt = random.sample(list(range(1, len(self.buckets))), k=1)[0]
                    if bucket_idx_opt != target_bucket:
                        # as random option get mean value in random bucket (!= target bucket)
                        # Note: buckets are indexed from 1 to N, i.e. [1, N)
                        rand_target_bucket_id = int(self.answers_options[bucket_idx_opt])
                        rand_target = self.buckets_means[rand_target_bucket_id]
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
class PredOverThresholdAmountTaskBinary(NumericTaskAbstract):
    """
    A task for predictive exact Binary QA task: given a discrete or continuous numeric target - Amount,
    answer question with binary answer.
    """

    def __post_init__(self):
        self.task_name = "pred_over_threshold_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.threshold = 0.41

        self.task_special_token = None
        self.task_specific_special_token = "[pred_over_threshold_AMNT_binary]"

        self.is_text_task = False
        self.is_binary_task = True
        self.is_open_ended_task = False

        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_outputs: Optional[bool] = False

        self.metrics = {
            "accuracy": BinaryAccuracy()
        }
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            f". Will the amount of the next transaction be more then {self.threshold}? Yes or No?",
            f". Will the next transaction amount be more then {self.threshold}? Choose one: Yes or No?",
            f". Is it true that the amount of next transaction will be more then {self.threshold}? Yes or No?",
            ". Define whether the following statement is true: in next transaction amount will be more then "
            f"{self.threshold}. Choose: Yes or No?",
            f". Is it true or false: the amount of the next transaction will be more then {self.threshold}? Yes or No?",
            ". Define whether the following statement is correct: in the next transaction amount will be more then "
            f" {self.threshold}. Choose: Yes or No?",
            f". Identify if the statement that: the amount of the next transaction will be more then {self.threshold}, "
            "is correct? Yes or No?",
            f". Determine whether the following statement is true: {self.threshold} will be more then "
            "the amount of the next transaction. Choose: Yes or No?",
            f". Is the statement correct: the amount of the next transaction will be more then {self.threshold}. "
            "Answer with one of the following options: Yes or No?",
            ". Answer the question whether or not the following statement is true: the amount of the next "
            f"transaction will be more then {self.threshold}. Yes or No?",
            f". Answer the question: will the amount of the next transaction be more then {self.threshold}? "
            "Choose only one of the following options: Yes or No?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

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
                                   special=True)

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
            with_numeric_input=self.numeric_inputs,
            with_numeric_output=self.numeric_outputs
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
            last_feature_index = mask_.sum() - 1
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            last_feature = feature_masked[-1]  # get a single Tensor value of a feature
            float_feature_ = self.buckets_means[last_feature.item()]  # take a mean bucket value of the last feature
            target_feature_value_batch.append(float_feature_ > self.threshold)
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(len(target_feature_value_batch))]

        # Target's questions binary [No/Yes]
        target_batch = list(map(lambda x:
                                self.binary_answer_options['positive'] if x
                                else self.binary_answer_options['negative'],
                                target_feature_value_batch))

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
class PredUnderThresholdAmountTaskBinary(NumericTaskAbstract):
    """
    A task for predictive exact Binary QA task: given a discrete or continuous numeric target - Amount,
    answer question with binary answer.
    """

    def __post_init__(self):
        self.task_name = "pred_under_threshold_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.threshold = 0.41

        self.task_special_token = None
        self.task_specific_special_token = "[pred_under_threshold_AMNT_binary]"

        self.is_text_task = False
        self.is_binary_task = True
        self.is_open_ended_task = False

        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_outputs: Optional[bool] = False

        self.metrics = {
            "accuracy": BinaryAccuracy()
        }
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            f". Will the amount of the next transaction be less then {self.threshold}? Yes or No?",
            f". Will the next transaction amount be less then {self.threshold}? Choose one: Yes or No?",
            f". Is it true that the amount of next transaction will be less then {self.threshold}? Yes or No?",
            ". Define whether the following statement is true: in next transaction amount will be less then "
            f"{self.threshold}. Choose: Yes or No?",
            f". Is it true or false: the amount of the next transaction will be less then {self.threshold}? Yes or No?",
            ". Define whether the following statement is correct: in the next transaction amount will be less then "
            f" {self.threshold}. Choose: Yes or No?",
            f". Identify if the statement that: the amount of the next transaction will be less then {self.threshold}, "
            "is correct? Yes or No?",
            f". Determine whether the following statement is true: {self.threshold} will be less then "
            "the amount of the next transaction. Choose: Yes or No?",
            f". Is the statement correct: the amount of the next transaction will be less then {self.threshold}. "
            "Answer with one of the following options: Yes or No?",
            ". Answer the question whether or not the following statement is true: the amount of the next "
            f"transaction will be less then {self.threshold}. Yes or No?",
            f". Answer the question: will the amount of the next transaction be less then {self.threshold}? "
            "Choose only one of the following options: Yes or No?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

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
                                   special=True)

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
            with_numeric_input=self.numeric_inputs,
            with_numeric_output=self.numeric_outputs
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
            last_feature_index = mask_.sum() - 1
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            last_feature = feature_masked[-1]  # get a single Tensor value of a feature
            float_feature_ = self.buckets_means[last_feature.item()]  # take a mean bucket value of the last feature
            target_feature_value_batch.append(float_feature_ < self.threshold)
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        question_target_batch = [question_end for _ in range(len(target_feature_value_batch))]

        # Target's questions binary [No/Yes]
        target_batch = list(map(lambda x:
                                self.binary_answer_options['positive'] if x
                                else self.binary_answer_options['negative'],
                                target_feature_value_batch))

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
class PredNumericAmountTaskOpenEnded(NumericTaskAbstract):
    """
    A task for predictive exact Open-ended QA task: given a discrete or continuous numeric target - Amount,
    answer question with exact numeric answer.
    """

    def __post_init__(self):
        self.task_name = "pred_numeric_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_numeric_AMNT_openended]"

        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True

        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_inputs: Optional[bool] = False
        self.numeric_outputs: Optional[bool] = True

        self.metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "ppl": Perplexity(ignore_index=-100)
        }
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". What is the exact amount of the next transaction based on provided transactions history?",
            ". What is the exact amount of the next transaction?",
            ". Find out the exact exact value of the next transaction's amount.",
            ". Identify the exact amount of the next transaction based on provided transactions history.",
            ". Find out what is the exact amount of next transaction based on provided transactions history.",
            ". Can you please answer the question: what is the exact amount of the next transaction?",
            ". Determine what will be the exact amount of the next transaction based on given transactions history?",
            ". Select the exact amount of the next transaction based on given history.",
            ". Write the exact amount of the next transaction.",
            ". Can you find out the exact amount that will occur in the next transaction?",
            ". Answer the question: what is the exact amount of the next transaction?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

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
                                   special=True)

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

        # Create question targets
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
            with_numeric_input=self.numeric_inputs,
            with_numeric_output=self.numeric_outputs
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
        batch_size = batch['mask'].shape[0]

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            last_feature_index = mask_.sum() - 1
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu"))  # get bucket feature without padding
            if self.is_real:
                # Construct target values from REAL-VALUED input data
                float_feature_ = feature_masked[-1]  # get a single Tensor value of a feature
            else:
                # Construct target values from DISCRETIZED input data
                feature_masked = feature_masked.long()
                last_feature = feature_masked[-1]  # get a single Tensor value of a feature
                float_feature_ = self.buckets_means[last_feature.item()]  # take a mean bucket value of the last feature

            target_feature_value_batch.append(float_feature_)
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # Convert to corresponding bucket id
        if self.is_real:
            target_feature_value_batch = torch.tensor(target_feature_value_batch).to(device)
        else:
            # If needed binned answer
            target_feature_value_bucket_batch = torch.tensor(np.digitize(
                np.asarray(target_feature_value_batch), bins=self.buckets)
            ).to(device)

        # Map to strings
        target_feature_value_batch = list(map(lambda x: str(round(x.item() if isinstance(x, torch.Tensor) else x, 3)),
                                              target_feature_value_batch))

        # Construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_feature_value_batch

    def process_outputs(self, outputs: Any, answers: torch.Tensor,
                        return_logits: Optional[bool] = True,
                        as_strings: Optional[bool] = False) -> Any:
        """
        Processing target text and output text to get the predictions
        """
        # Get predictions as list of strings
        default_value = 0.0
        predictions_decoded = self.tokenizer.batch_decode(outputs['logits'].argmax(2),
                                                          skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(outputs['labels'],
                                              skip_special_tokens=True)

        # In case multiple floating points in numeric answers -> take last one: 0.0.9 => 0.9
        predictions = [make_float(float_splitter(pred)) for pred in predictions_decoded]

        # Clean predicted texts and map them to categorical labels
        predictions = [float(transform_labels(pred,
                                        do_make_numeric=True,
                                        do_clean_text=False,
                                        default_value=default_value))
                       for pred in predictions]

        targets = [float(transform_labels(answer,
                                    do_make_numeric=True,
                                    do_clean_text=False,
                                    default_value=default_value))
                   for answer in targets]

        # Assumed, that floating point features are in provided values range
        predictions = [pred if pred <= self.feature_max else self.feature_max for pred in predictions]
        predictions = [pred if pred >= self.feature_min else self.feature_min for pred in predictions]

        processed_outputs = dict(targets=targets,
                                 predictions=predictions)
        if return_logits:
            processed_outputs['predictions_logits'] = outputs['logits']
            processed_outputs['labels_tokens'] = outputs['labels']

        return processed_outputs

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]],
                          **kwargs) -> dict:
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
        metrics = {}

        processed_outputs = self.process_outputs(outputs, answers, return_logits=True)
        targets = processed_outputs['targets']
        preds = processed_outputs['predictions']
        preds_logits = processed_outputs['predictions_logits'] if 'predictions_logits' in processed_outputs else None
        targets_tokens = processed_outputs['labels_tokens'] if 'predictions_logits' in processed_outputs else None

        try:
            if 'mse' in task_metrics:
                mse = task_metrics['mse'](preds, targets)
                metrics['mse'] = task_metrics['mse']
        except Exception as e:
            print(f"Error during `MSE` metric calculation: {e}")

        try:
            if 'mae' in task_metrics:
                mae = task_metrics['mae'](preds, targets)
                metrics['mae'] = task_metrics['mae']
        except Exception as e:
            print(f"Error during `mae` metric calculation: {e}")

        try:
            if 'ppl' in task_metrics:
                ppl = task_metrics['ppl'](preds_logits, targets_tokens)
                metrics['ppl'] = task_metrics['ppl']
        except Exception as e:
            print(f"Error during `ppl` metric calculation: {e}")

        return metrics


@dataclass
class PredBinnedAmountTaskOpenEnded(NumericTaskAbstract):
    """
    A task for predictive exact Open-ended QA task: given a discrete or continuous numeric target - Amount,
    answer question with a bin index of exact numeric answer.
    """

    def __post_init__(self):
        self.task_name = "pred_binned_amount_open-ended"
        self.target_feature_name = 'amnt'  # [0, 1] range of values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_binned_AMNT_openended]"

        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True

        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_inputs: Optional[bool] = False
        self.numeric_outputs: Optional[bool] = False

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". To which range of values does the amount of the next client's transaction belongs to?"
            ". What is the range of the amount of the next transaction based on provided transactions history?",
            ". What is the index of the values range to which amount of the next transaction belongs to?",
            ". In which range of values falls the amount of the next transaction?",
            ". Find out the range of values of the next transaction's amount.",
            ". Identify the range of values of the next transaction based on provided transactions history.",
            ". Find out what is the amount value range of next transaction based on provided transactions history.",
            ". Can you please answer the question: what is the range of values of the next transaction's amount?",
            ". Determine to which values range will the amount of the next transaction fall into?",
            ". Select the values range of the amount of the next transaction based on given history.",
            ". Write the values range of amount of the next transaction.",
            ". Can you find out to which values range will the next transaction's amount refer to?",
            ". Answer the question: what is the value range of the amount in next transaction?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.answer_template: str = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

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
            "accuracy": Accuracy(task='multiclass', num_classes=len(self.buckets)),
            "ppl": Perplexity(ignore_index=-100)
        }

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=True)

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

        # Create question targets
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
            with_numeric_input=self.numeric_inputs,
            with_numeric_output=self.numeric_outputs
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
        batch_size = batch['mask'].shape[0]

        target_feature_batch = batch[self.target_feature_type][
            self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            last_feature_index = mask_.sum() - 1
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            last_feature = feature_masked[-1]  # get a single Tensor value of a feature
            float_feature_ = self.buckets_means[last_feature.item()]  # take a mean bucket value of the last feature
            target_feature_value_batch.append(float_feature_)
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # Convert to corresponding bucket id
        target_feature_value_bucket_batch = torch.tensor(np.digitize(
            np.asarray(target_feature_value_batch), bins=self.buckets)
        ).to(device)

        # Map to strings
        target_feature_value_bucket_batch = list(map(lambda x: str(int(x)), target_feature_value_bucket_batch))

        # Construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_feature_value_bucket_batch

    def process_outputs(self, outputs: Any, answers: torch.Tensor,
                        return_logits: Optional[bool] = True,
                        as_strings: Optional[bool] = False) -> Any:
        """
        Processing target text and output text to get the predictions
        """
        # Get predictions as list of strings
        default_value = 0
        predictions_decoded = self.tokenizer.batch_decode(outputs['logits'].argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(outputs['labels'],
                                                            skip_special_tokens=True)
        # Clean predicted texts and map them to categorical labels
        predictions_clean = [transform_labels(pred,
                                              do_make_numeric=True,
                                              do_clean_text=False,
                                              default_value=default_value)
                             for pred in predictions_decoded]

        batch_answers_decoded = [transform_labels(answer,
                                                  do_make_numeric=True,
                                                  do_clean_text=False,
                                                  default_value=default_value)
                                 for answer in batch_answers_decoded]

        # Map to available labels
        classes = [int(answer) for answer in self.answers_options]
        predictions_clean = [pred if pred in classes else default_value
                             for pred in predictions_clean]

        # To Tensors
        targets = torch.LongTensor(batch_answers_decoded)
        predictions = torch.LongTensor(predictions_clean)

        processed_outputs = dict(targets=targets,
                                 predictions=predictions)
        if return_logits:
            processed_outputs['predictions_logits'] = outputs['logits']
            processed_outputs['labels_tokens'] = outputs['labels']

        return processed_outputs

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor,
                          task_metrics: Union[torch.nn.ModuleDict, Dict[str, Any]],
                          **kwargs) -> dict:
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
        metrics = {}

        processed_outputs = self.process_outputs(outputs, answers, return_logits=True)
        targets = processed_outputs['targets']
        preds = processed_outputs['predictions']
        preds_logits = processed_outputs['predictions_logits'] if 'predictions_logits' in processed_outputs else None
        targets_tokens = processed_outputs['labels_tokens'] if 'predictions_logits' in processed_outputs else None

        try:
            if 'accuracy' in task_metrics:
                acc = task_metrics['accuracy'](preds, targets)
                metrics['accuracy'] = task_metrics['accuracy']
        except Exception as e:
            print(f"Error during `accuracy` metric calculation: {e}")

        try:
            if 'f1' in task_metrics:
                f1 = task_metrics['f1'](preds, targets)
                metrics['f1'] = task_metrics['f1']
        except Exception as e:
            print(f"Error during `f1` metric calculation: {e}")

        try:
            if 'ppl' in task_metrics:
                ppl = task_metrics['ppl'](preds_logits, targets_tokens)
                metrics['ppl'] = task_metrics['ppl']
        except Exception as e:
            print(f"Error during `ppl` metric calculation: {e}")

        return metrics
