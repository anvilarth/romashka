import torch
import random

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import AUROC
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score, F1Score, Accuracy)
from romashka.transactions_qa.tasks.categorical_task_abstract import CategoricalTaskAbstract
from romashka.transactions_qa.evaluation.eval_processings_utils import transform_labels


@dataclass
class MostFrequentCountryTaskOpenEnded(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "most_frequent_country_open-ended"
        self.target_feature_name = 'country'  # 24 unique values

        self.task_special_token = None
        self.task_specific_special_token = "[most_freq_country_openended]"

        self.num_classes = 25
        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True
        self.metrics = torch.nn.ModuleDict({
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
            "f1": F1Score(task="multiclass", num_classes=self.num_classes)
        })
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". Which country is the most frequent?",
            ". Answer the question: which country is the most frequent?",
            ". Choose the most frequent country.",
            ". Select the most frequent country.",
            ". Answer the question: which country is the most frequent within "
            "clients' transactions throughout the history?",
            ". Find out which country is the most frequent in history?",
            ". Which country is the most frequent within clients' transactions throughout the history?",
            ". Identify which country is the most frequent within clients' transactions?",
            ". Can you find out which country is the most frequent?",
            ". Can you find out which country is the most frequent?"
            ". Would you answer the question: which  country is the most frequent based "
            "on the client's transaction history?"
        ]
        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(1, self.num_classes)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

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

        return targets, predictions

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

        except Exception as e:
            print(f"Error during metrics calculation: {e}")

        return metrics


@dataclass
class LastCountryTaskOpenEnded(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "last_country_open-ended"
        self.target_feature_name = 'country'  # 24 unique values

        self.task_special_token = None
        self.task_specific_special_token = "[last_country_openended]"

        self.num_classes = 25
        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True
        self.metrics = torch.nn.ModuleDict({
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
            "f1": F1Score(task="multiclass", num_classes=self.num_classes)
        })

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". What is the country of the last transaction that occurred in history?",
            ". What is the country of the very last transaction encountered in the transaction history?",
            ". Choose the last transaction country.",
            ". Select the country of the most recently occurred transaction.",
            ". Find out what is the country of last transaction that occurred in history.",
            ". Can you please answer the question: what is the country of the most recent transaction?",
            ". Determine the country of the last transaction in history?",
            ". Select the country of the last transaction that encountered in history.",
            ". Choose the country of the most recent transaction in the history",
            ". Can you find out of which country was the most recent transaction?",
            ". Answer the question: what is the country of the latest transaction?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(1, self.num_classes)]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

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
        target_feature_batch = batch['cat_features'][self.target_feature_index]  # Tensor [batch_size, seq_len]

        # Construct target values
        target_feature_value_batch = []
        for i, (feature_, mask_) in enumerate(zip(target_feature_batch, mask_batch)):
            feature_masked = torch.masked_select(feature_.to("cpu"),
                                                 mask=mask_.to("cpu")).long()  # get feature without padding
            last_feature = feature_masked[-1]  # get a single Tensor value of a feature
            target_feature_value_batch.append(last_feature.to(device))

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

        return targets, predictions

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

        except Exception as e:
            print(f"Error during metrics calculation: {e}")

        return metrics