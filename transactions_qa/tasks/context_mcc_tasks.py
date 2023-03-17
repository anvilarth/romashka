import torch
import random
import pandas as pd
from abc import ABC, abstractmethod

# DTO
import dataclasses
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional)

import transformers
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.classification.f_beta import F1Score

from .task_abstract import AbstractTask


@dataclass
class MostFrequentMCCCode(AbstractTask):
    tokenizer: transformers.PreTrainedTokenizerBase = None

    def __init__(self):
        self.task_name = "most_frequent_mcc_code"
        self.target_feature_name = 'mcc'  # 108 unique values
        self.is_binary_task = False  # for a default for this task
        self.metrics = {
            "rouge": ROUGEScore()
        }
        if self.is_binary_task:
            self.question_templates = [
                ("This is the client's transaction history ",
                 " Is %d MCC code is the most frequent? Yes or No?"),
                ("You are given the client's transaction history ",
                 " Is %d MCC code is the most frequent? Choose one: Yes or No?"),
            ]
        else:
            self.question_templates = [
                ("This is the client's transaction history ",
                 " Which MCC code is the most frequent?"),
                ("This is the client's transaction history ",
                 " Select the most frequent MCC code."),
                ("You are given the client's transaction history ",
                 " Choose the most frequent MCC code."),
            ]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.answers_options = [str(i) for i in range(108)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True
        self.num_options = 6  # ground truth + 5 additional options

    def __post_init__(self):
        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer = self.tokenizer,
                                   new_tokens = [self.transactions_embeddings_start_token,
                                                 self.transactions_embeddings_end_token],
                                   special = False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]

        mask_batch = batch['mask']
        target_feature_batch = batch['cat_features'][self.target_feature_index]

        # Construct target values
        target_feature_value_batch = []
        for feature_, mask_ in zip(target_feature_batch, mask_batch):
            feature_ = torch.masked_select(feature_, mask = mask_)
            codes, cnt = torch.unique(feature_, return_counts = True)
            most_freq_feature = codes[torch.argmax(cnt)]
            target_feature_value_batch.append(most_freq_feature)

        # Map to strings
        target_feature_value_batch = list(map(lambda x: str(x), target_feature_value_batch))

        # for binary task randomly sample True and False examples from batch
        # and construct target sequences
        target_batch = []
        question_target_batch = []  # as strings

        if self.is_binary_task:
            # Mask
            pos_neg_target_mask = torch.randint(0, 2, (len(target_feature_value_batch),), dtype = torch.int).bool()

            # Target's questions binary
            target_batch = list(map(lambda x: "Yes" if x else "No", pos_neg_target_mask))

            # ground truth target (int/str), mask (bool)
            for target, pos_neg_mask in zip(target_feature_value_batch, pos_neg_target_mask):
                if pos_neg_mask:
                    # positive
                    question_target_batch.append(question_end % target)
                else:
                    # negative
                    rand_target = None
                    while rand_target is None:
                        opt = random.sample(self.answers_options, k = 1)[0]
                        if opt != target:
                            rand_target = opt
                    question_target_batch.append(question_end % rand_target)

        else:
            target_batch_options = []
            for gt_target in target_feature_value_batch:
                target_options = {gt_target}
                while len(target_options) < self.num_options:
                    target_options.add(random.sample(self.answers_options, k = 1)[0])

                # shuffle
                target_options = random.sample(list(target_options), k = len(target_options))
                # connect with separator
                target_options = self.multichoice_separator.join(target_options)
                target_batch_options.append(target_options)

            question_target_batch = [question_end + " OPTIONS: " + target_options for target_options in
                                     target_batch_options]

            # Target's questions numeric/categorical answers as str
            target_batch = target_feature_value_batch

        # Encode
        # question_start  -> 'start str <trx>'
        # question_target_bin/question_target_batch  -> '</trx> + end str + OPTIONS:... / Yes or No'
        # target_batch -> feature values as str ('15')

        question_start_tokens = self.tokenizer.encode(question_start, return_tensors = 'pt').to(
            device)  # single tensor + </s>
        question_target_encoded_batch = self.tokenizer(question_target_batch,
                                                       padding = True,
                                                       truncation = True,
                                                       return_tensors = 'pt').to(device)
        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor)

        target_encoded_batch = [self.tokenizer.encode(target,
                                                      return_tensors = 'pt')[:, :-1].to(device)
                                for target in target_batch]  # list of tensors, each token_ids (no eos token!)
        return dict(
            question_start_tokens = question_start_tokens,
            question_end_tokens = question_target_encoded_batch['input_ids'],
            question_end_attention_mask = question_target_encoded_batch['attention_mask'],
            targets_encoded = target_encoded_batch
        )
        # To embedding of LM
        # question_start_embeddings = lm_model.encoder.embed_tokens(
        #     question_start_tokens)  # call for (embed_tokens): Embedding(32128, 512)
        # question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)
        #
        # question_end_embeddings_batch = lm_model.encoder.embed_tokens(
        #     question_target_encoded_batch['input_ids'])  # call for (embed_tokens): Embedding(32128, 512)

    def process_input_sample(self, sample: Any, **kwargs) -> Any:
        pass
