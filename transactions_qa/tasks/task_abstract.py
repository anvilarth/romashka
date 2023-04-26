import dataclasses

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

import transformers
import tokenizers
from torchmetrics.text.rouge import ROUGEScore

from romashka.logging_handler import get_logger

logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)

from romashka.transactions_qa.dataset.data_generator import (transaction_features,
                                                             num_features_names,
                                                             cat_features_names)


@dataclass
class AbstractTask(ABC):
    """
    todo: https://github.com/google-research/xtreme/blob/master/third_party/utils_lareqa.py
    Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    metrics: specifies the metrics to evaluate the task based on them.
    ...
    """
    task_name: Optional[str] = None
    target_feature_name: Optional[str] = None
    target_feature_index: Optional[int] = None
    target_feature_type: Optional[str] = None
    task_specific_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Union[Dict[str, Any], nn.ModuleDict]] = None
    question_templates: Optional[List[Tuple[str, str]]] = None  # (starting, ending)
    answer_template: Optional[str] = None
    answers_options: Optional[List[str]] = None

    tokenizer: transformers.PreTrainedTokenizerBase = None

    seed: Optional[int] = 11
    verbose: Optional[bool] = False
    task_special_token: Optional[str] = None
    transactions_embeddings_start_token: Optional[str] = r"[trx]"
    transactions_embeddings_end_token: Optional[str] = r"[/trx]"
    special_tokens: Optional[List[str]] = None
    add_tokens_to_tokenizer: Optional[bool] = False
    # This option is set to `False` if the answer is binary: yes/no, true/false, or multichoice
    # Otherwise, it is set to `True` to not require any information about available options/targets
    is_open_ended_task: Optional[bool] = True
    multichoice_separator: Optional[str] = " - %s;"
    num_options: Optional[int] = 6  # ground truth + 5 additional options
    is_few_shot: Optional[bool] = False  # whether to provide few examples before question
    n_shot: Optional[int] = 1

    def __post_init__(self):
        # Fill in empty parameters with defaults
        if self.special_tokens is None:
            self.special_tokens = [self.transactions_embeddings_start_token,
                                   self.transactions_embeddings_end_token]
            if self.task_special_token is not None:
                self.special_tokens += [self.task_special_token]
        if self.target_feature_name is not None:
            self.init_feature_index()
        self.task_specific_config = {
            "source_msx_seq_len": 512,
            "target_max_seq_len": 128
        } if self.task_specific_config is None else self.task_specific_config
        # Init metrics
        self.metrics = nn.ModuleDict({"rouge": ROUGEScore()} if self.metrics is None else self.metrics)
        self.question_templates = [
            ("This is the client's transaction history ", "")] \
            if self.question_templates is None else self.question_templates
        self.answer_template = [
            ""  # empty for default
        ] if self.answer_template is None else self.answer_template
        if self.answers_options is None:
            if not self.is_open_ended_task:
                raise ValueError(f"The task is marked as multi choice, but no choices for answers provided!")

    @abstractmethod
    def process_input_batch(self, batch: Dict[str, Any], **kwargs):
        """
        Apply task-specific processing for a batch of data.
        """
        raise NotImplementedError

    # @abstractmethod  todo: later add this for muli-task inside a single batch
    def process_input_sample(self, sample: Any, **kwargs) -> Any:
        """
        Apply task-specific processing for a single data sample.
        """
        raise NotImplementedError

    # @abstractmethod  todo: later add this to separate functionality of target generation
    def generate_target(self, sample: Any, **kwargs) -> Any:
        """
        Generated question/answer-specific target sequence.
        """
        raise NotImplementedError

    def generate_target_question(self, question_end: Any, target_batch: Any, **kwargs) -> Any:
        """
        Generated target question
        """
        raise NotImplementedError

    def process_outputs(self, outputs: Any, answers: torch.Tensor) -> Any:
        """
        Processing target text and output text to get the predictions
        """
        raise NotImplementedError

    def calculate_metrics(self, outputs: Any, answers: torch.Tensor, task_metrics: dict, **kwargs) -> dict:
        """
        Calculate task metrics.
        """
        raise NotImplementedError

    def init_feature_index(self):
        """
        Initialize the feature index.
        """
        if self.target_feature_name in num_features_names:
            self.target_feature_index = num_features_names.index(self.target_feature_name) \
                if self.target_feature_index is None else self.target_feature_index
            self.target_feature_type = 'num_features'

        elif self.target_feature_name in cat_features_names:
            self.target_feature_index = cat_features_names.index(self.target_feature_name) \
                if self.target_feature_index is None else self.target_feature_index
            self.target_feature_type = 'cat_features'
        else:
            raise AttributeError(f"Provided feature name not in available"
                                 f"transactions feature names:\n{transaction_features}")
        logger.info(f"For feature with name: {self.target_feature_name} of type {self.target_feature_type}, "
                    f"set index = {self.target_feature_index}")

    def update_feature_index(self):
        """
        Update the feature index.
        """
        if self.target_feature_name in num_features_names:
            self.target_feature_index = num_features_names.index(self.target_feature_name)
            self.target_feature_type = 'num_features'

        elif self.target_feature_name in cat_features_names:
            self.target_feature_index = cat_features_names.index(self.target_feature_name)
            self.target_feature_type = 'cat_features'
        else:
            raise AttributeError(f"Provided feature name not in available"
                                 f"transactions feature names:\n{transaction_features}")
        logger.info(f"For feature with name: {self.target_feature_name} of type {self.target_feature_type}, "
                    f"set index = {self.target_feature_index}")

    @staticmethod
    def extend_vocabulary(
            tokenizer: transformers.PreTrainedTokenizerBase,
            new_tokens: List[str],
            special: Optional[bool] = True,
            return_ids: Optional[bool] = False,
            model: Optional[transformers.PreTrainedModel] = None) -> Optional[Dict[str, int]]:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        """
        num_added_toks = tokenizer.add_tokens(new_tokens, special_tokens=special)
        logger.info(f"Added to tokenizer: {num_added_toks} tokens: {new_tokens}.")
        if model is not None:
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary,
            # i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
        else:
            logger.info(f"Notice: resize_token_embeddings of a model to adapt to the size of the new vocabulary!")

        # get new tokens ids in tokenizers' vocabulary
        if return_ids:
            return {token: tokenizer(token)['input_ids'][0] for token in new_tokens}



    def custom_tokenize(self, sequence: Union[str, List[str]],
                        **kwargs) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        A custom tokenization for a task. It takes input text (or sequence of texts),
        split them according to pre-defined logic (as `pre-tokenizer`)
        and return with additional (optional) post-processing.
        Args:
            sequence: a str or a List[str] as an input text (or sequence of texts);
            **kwargs: other args for any steps of tokenization pipeline;

        Returns:
            a Dictionary with values as List[int] or torch.Tensor, keys:
                'input_ids' - a tokenized text (or sequence of texts);
                'attention_mask' - an attention mask for tokenized text (or sequence of texts).
        """
        sequence, is_pre_tokenized = self.pre_tokenize(sequence=sequence)
        if not len(sequence) and is_pre_tokenized:
            sequence = ""
        sequence_encoded = self.tokenizer(sequence,
                                          is_split_into_words=True if is_pre_tokenized else False,
                                          **kwargs)
        return sequence_encoded

    def pre_tokenize(self, sequence: Union[str, List[str]]) -> Tuple[Union[str, List[str], List[List[str]]], bool]:
        """
        Pre-tokenization is required for splitting a text into smaller objects.
        For instance, here is a pre-tokenizer that will split on space, punctuation and digits,
        separating numbers in their individual digits.
        Args:
            sequence: a str or a List[str] sequence(-s) of texts;

        Returns:
            a List[str] or List[List[str]] - a list of words on which all input text was separated;
            a boolean flag that pre-tokenization was made.
        """
        is_pre_tokenized = False
        if hasattr(self.tokenizer, "pre_tokenizer") and (self.tokenizer.pre_tokenizer is not None):
            is_pre_tokenized = True
            if isinstance(sequence, str):
                sequence = [pretok_sequence[0]
                            for pretok_sequence in self.tokenizer.pre_tokenizer.pre_tokenize_str(sequence)]
            else:
                pretokenized = [self.tokenizer.pre_tokenizer.pre_tokenize_str(pretok_seq) for pretok_seq in sequence]
                sequence = [[pretokenized_subsequence[0] for pretokenized_subsequence in pretokenized_sequence]
                            for pretokenized_sequence in pretokenized]
        return sequence, is_pre_tokenized


    def update(self, new_attr: Dict[str, Any]):
        """
        Updates the parameters of class with provided {key: value} pair(-s)
        Args:
            new_attr (): a dictionary, where a key is attribute's name, value - a new attribute's value;
        Returns:

        """
        for key, value in new_attr.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"For Task attribute: {key} set value = {value}")
