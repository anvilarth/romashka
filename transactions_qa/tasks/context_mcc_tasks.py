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

from .task_abstract import AbstractTask


@dataclass
class MostFrequentMCCCodeTask(AbstractTask):

    tokenizer: transformers.PreTrainedTokenizerBase = None

    def __post_init__(self):
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
                 ". Which MCC code is the most frequent?"),
                ("This is the client's transaction history ",
                 ". Select the most frequent MCC code."),
                ("You are given the client's transaction history ",
                 ". Choose the most frequent MCC code."),
            ]
        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.answers_options = [str(i) for i in range(108)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True
        self.num_options = 6  # ground truth + 5 additional options

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=[self.transactions_embeddings_start_token,
                                               self.transactions_embeddings_end_token],
                                   special=False)

    def process_input_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Construct templates
        question_start, question_end = random.choice(self.question_templates)
        question_start = question_start + self.transactions_embeddings_start_token
        question_end = self.transactions_embeddings_end_token + question_end

        device = batch['mask'].device
        batch_size = batch['mask'].shape[0]
        # print(f"Task.process_input_batch():\ton device: {device}, with batch_size: {batch_size}")

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
        target_batch = []
        question_target_batch = []  # as strings

        if self.is_binary_task:
            # Mask [0/1]
            pos_neg_target_mask = torch.randint(0, 2, (len(target_feature_value_batch),), dtype=torch.int).bool()

            # Target's questions binary [No/Yes]
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
                        opt = random.sample(self.answers_options, k=1)[0]
                        if opt != target:
                            rand_target = opt
                    question_target_batch.append(question_end % rand_target)

        else:
            target_batch_options = []  # a list of str target options
            for gt_target in target_feature_value_batch:
                target_options = {gt_target}
                while len(target_options) < self.num_options:
                    target_options.add(random.sample(self.answers_options, k=1)[0])

                # shuffle
                target_options = random.sample(list(target_options), k=len(target_options))
                # connect with separator
                target_options = "".join([self.multichoice_separator % target for target in target_options])
                print(f"Multichoice target_options: {target_options} with true option: {gt_target}")
                target_batch_options.append(target_options)

            question_target_batch = [question_end + " OPTIONS:" + target_options for target_options in
                                     target_batch_options]
            print(f"question_target_batch: {question_target_batch}")
            # Target's questions numeric/categorical answers as str
            target_batch = target_feature_value_batch

        # Encode
        # question_start  -> 'start str <trx>'
        # question_target_bin/question_target_batch  -> '</trx> + end str + OPTIONS:... / Yes or No'
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
        question_start_tokens_mask = torch.ones(question_start_tokens.size()).repeat(batch_size, 1)
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
        batch_answer_template_mask = torch.ones(batch_size, answer_template_encoded.shape[1])
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       target_encoded_batch['attention_mask']], dim=1).to(device)

        return dict(
            question_start_tokens=question_start_tokens,
            question_start_attention_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask'],
            # answer_template_tokens=batch_answer_template_encoded,
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask
        )

    def process_input_sample(self, sample: Any, **kwargs) -> Any:
        pass

    def generate_target(self, sample: Any, **kwargs) -> Any:
        pass
