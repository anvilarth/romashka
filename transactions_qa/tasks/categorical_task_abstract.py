import torch
import random
import numpy as np

from abc import ABC

# DTO
from dataclasses import dataclass
from typing import Optional, Union, List, Any

from romashka.transactions_qa.tasks import AbstractTask


@dataclass
class CategoricalTaskAbstract(AbstractTask, ABC):

    """
    Parent class for all tasks with categorical target feature.
    """
    # Needed to switch model in to numeric mode processing
    numeric_inputs: Optional[bool] = False
    numeric_outputs: Optional[bool] = False

    num_classes: Optional[int] = None
    ignore_class_index: Optional[int] = None
    decision_threshold: Optional[float] = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.target_feature_type = 'cat_features'

    @classmethod
    def sample_random_negative(cls,
                               true_target: Union[torch.Tensor, str],
                               answers_options: List[str],
                               output_dtype: Optional[Union[str, torch.dtype]] = torch.int64) -> torch.Tensor:
        if isinstance(true_target, torch.Tensor):
            if len(true_target.size()) != 0:  # not a scalar
                raise AttributeError(f"True target is not a scalar: {true_target}")
            true_target = str(true_target.long().item())  # as an integer number string

        # negative
        rand_target = None
        while rand_target is None:
            opt = random.sample(answers_options, k=1)[0]
            if opt != true_target:
                rand_target = opt

        rand_target = torch.as_tensor([int(rand_target)], dtype=output_dtype)
        return rand_target

    def process_input_multichoice(self, sample: Any, **kwargs) -> Any:
        """
        Creates a multichoice sequence of [input prompts + Q + Answer variant]
        for each of available answer variants.
        The prediction of model should be provided if form of score (PPL/CE ...) for each of input sequences.
        An answer variant with the lowest score would be chosen then as a final answer.
        Args:
            sample: an input sample;
            **kwargs: ***
        Returns:
                a List of Dicts with created sequences and supplementary info (masks, etc.).
        """
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)

        if self.task_special_token is not None:
            question_start = self.task_special_token + " " + question_start
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = sample['mask'].device

        # Create question targets as concatenation of "question end + target (true/random) + ?"
        # and targets as string targets representation, for binary task: Yes/No options
        question_target_batch, target_batch = self.generate_target(sample, question_end=question_end)

        true_target_idx = self.answers_options.index(target_batch[0])
        all_targets_options = self.answers_options
        batch_size = len(all_targets_options)

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
        transactions_embedding_mask = sample['mask'].repeat(batch_size, 1).to(device)

        encoder_input_mask = torch.cat(
            [question_start_tokens_mask,
             transactions_embedding_mask,
             question_end_tokens_mask.repeat(batch_size, 1).to(device)],
            dim=1)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        # add [:, :-1] for no EOS tokens - ?
        # Answer template encoding + strip </s> (EOS) token

        targets_options_encoded = self.custom_tokenize(all_targets_options,
                                                       return_tensors='pt',
                                                       padding=True,
                                                       truncation=True).to(device)

        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       return_attention_mask=False)['input_ids'][:, :-1].to(device)
        # Answer template encoding + target tokens + EOS token
        answer_template_encoded = answer_template_encoded.repeat(
            targets_options_encoded['input_ids'].size(0), 1)

        batch_answer_encoded = torch.cat([answer_template_encoded,
                                          targets_options_encoded['input_ids']], dim=1).long().to(device)

        # Answer masks
        batch_answer_template_mask = torch.ones(targets_options_encoded['attention_mask'].size(0),
                                                answer_template_encoded.shape[1]).to(device)
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       targets_options_encoded['attention_mask']], dim=1)

        return dict(
            question_start_tokens=question_start_tokens,
            question_start_tokens_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=targets_options_encoded['input_ids'],  # GT target + options
            target_attention_mask=targets_options_encoded['attention_mask'],  # GT target + options
            true_target_idx=true_target_idx,
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            encoder_input_mask=encoder_input_mask,
            with_numeric_input=self.numeric_inputs,
            with_numeric_output=self.numeric_outputs
        )