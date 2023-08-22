import torch

import random
import datetime

# DTO
from dataclasses import dataclass
from typing import (Dict, Any, Optional, Union)

from torchmetrics.text import BLEUScore

from .categorical_task_abstract import CategoricalTaskAbstract

from romashka.transactions_qa.dataset.data_generator import (transaction_features,
                                                             num_features_names,
                                                             cat_features_names)


@dataclass
class PredDateTaskOpenEnded(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "pred_date_open-ended"
        self.target_feature_name = 'date'  # 53 unique values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_date_openended]"

        self.year = 2019
        self.num_classes = 365  # as days in a year
        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True
        self.metrics = torch.nn.ModuleDict({
            "bleu_1": BLEUScore(n_gram=1)
        })
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]
        self.ending_prompts = [
            ". What is a date of the next client's transaction?"
            " Answer a date in DD/MM format, where DD - is a day and MM - is a month."
            " For example 31 April is a 31/04."
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(self.num_classes)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

        self.hour_feature_index = cat_features_names.index('hour')
        self.day_of_week_feature_index = cat_features_names.index('day_of_week')
        self.week_of_year_feature_index = cat_features_names.index('weekofyear')

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.special_tokens,
                                   special=True)

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

        day_of_week_feature_batch = batch['cat_features'][self.day_of_week_feature_index]  # [batch_size, seq_len]
        week_of_year_feature_batch = batch['cat_features'][self.week_of_year_feature_index]  # [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (dow_feature_, woy_feature_, mask_) in enumerate(zip(day_of_week_feature_batch,
                                                                    week_of_year_feature_batch,
                                                                    mask_batch)):
            dow_feature_masked = torch.masked_select(dow_feature_.to("cpu"),
                                                     mask=mask_.to("cpu")).long()  # get feature without padding
            woy_feature_masked = torch.masked_select(woy_feature_.to("cpu"),
                                                     mask=mask_.to("cpu")).long()  # get feature without padding
            joined_feature_masked = torch.cat([dow_feature_masked.unsqueeze(0),
                                               woy_feature_masked.unsqueeze(0)], dim=0)
            codes, cnt = torch.unique(joined_feature_masked.permute(-1, 0), return_counts=True, dim=0)
            most_freq_feature = codes[torch.argmax(cnt)].long()  # get a single Tensor value of a feature
            target_feature_value_batch.append(most_freq_feature.to(device))
            # Mask last feature to predict it!
            last_feature_index = mask_.sum() - 1
            batch['mask'][i, last_feature_index] = 0

        # Target's questions numeric/categorical answers as str
        # Convert to DD/MM format
        target_batch = []
        day_of_year_feature_batch = (target_feature_value_batch[:, 1] * 7) + target_feature_value_batch[:, 0]

        for doy in day_of_year_feature_batch:
            datetime_object = datetime.datetime(self.year, 1, 1) + datetime.timedelta(doy.item() - 1)

            # get month and day of the month
            month = str(datetime_object.month)
            if len(month) < 2:
                month = "0" + month
            day_of_the_month = str(datetime_object.day)
            if len(day_of_the_month) < 2:
                day_of_the_month = "0" + day_of_the_month
            target = f"{day_of_the_month}/{month}"
            target_batch.append(target)

        # Construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        # Encode
        # question_start  -> '[task_special_token] + start str [trx]'
        # question_target_batch  -> '[/trx] + end str.'
        # target_batch -> feature values as str ('15')

        # single tensor without </s> (EOS), but only for encoder-decoder !!!
        question_start_tokens = self.tokenizer.encode(question_start,
                                                      add_special_tokens=True,
                                                      return_tensors='pt')
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.tokenizer(question_target_batch,
                                                       padding=True,
                                                       truncation=True,
                                                       add_special_tokens=False,
                                                       return_attention_mask=True,
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
                                                                add_special_tokens=False,
                                                                return_tensors='pt').to(device)
        batch_answer_mask = target_encoded_batch['attention_mask']

        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.tokenizer.encode(self.answer_template,
                                                        add_special_tokens=False,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        answer_template_mask = answer_template_encoded['attention_mask'].to(device)
        answer_template_encoded = answer_template_encoded['input_ids'].to(device)

        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        batch_answer_template_mask = answer_template_mask.repeat(batch_size, 1)

        # Answer template encoding + target tokens + EOS token
        batch_answer_encoded = torch.cat([batch_answer_template_encoded,
                                          target_encoded_batch['input_ids']], dim=1).to(device)
        # Answer masks
        batch_answer_template_mask = torch.ones(batch_size, answer_template_encoded.shape[1]).to(device)
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       batch_answer_mask], dim=1)

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

    def process_outputs(self, outputs: Any, answers: torch.Tensor, as_strings: Optional[bool] = False) -> Any:
        """
        Processing target text and output text to get the predictions
        """
        # Get predictions as list of strings
        default_value = 0
        predictions_decoded = self.tokenizer.batch_decode(outputs['logits'].argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(outputs['labels'],
                                                            skip_special_tokens=True)
        # todo: add postprocessing
        return batch_answers_decoded, predictions_decoded

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
        try:
            targets, preds = self.process_outputs(outputs, answers)

            if 'accuracy' in task_metrics:
                acc = task_metrics['accuracy'](preds, targets)
                metrics['accuracy'] = task_metrics['accuracy']

            if 'f1' in task_metrics:
                f1 = task_metrics['f1'](preds, targets)
                metrics['f1'] = task_metrics['f1']

            if 'bleu_1' in task_metrics:
                bleu_1 = task_metrics['bleu_1'](preds, [targets])
                metrics['bleu_1'] = task_metrics['bleu_1']

        except Exception as e:
            print(f"Error during metrics calculation: {e}")

        return metrics