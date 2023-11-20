import torch
import random
import numpy as np

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

import transformers
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError, Perplexity

from .numeric_task_abstract import NumericTaskAbstract
from romashka.transactions_qa.utils import get_buckets_info
from romashka.transactions_qa.evaluation.eval_processings_utils import (float_splitter,
                                                                        make_float,
                                                                        transform_labels)
from romashka.transactions_qa.dataset.data_generator import (
    transaction_features,
    num_features_names
)

@dataclass
class PredHourDiffTaskOpenEnded(NumericTaskAbstract):
    """
    A task for predictive exact Open-ended QA task: given a discrete or continuous numeric target - hour_diff,
    answer question with exact numeric answer.
    """

    def __post_init__(self):
        self.task_name = "pred_hour_diff_open-ended"
        self.target_feature_name = 'hour_diff'  # [0, 8000+] range of values, but crop to [0, 95]

        self.task_special_token = None
        self.task_specific_special_token = "[pred_numeric_hour_diff_openended]"

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
            ". What is the difference in hours from the current to the next customer's transaction?"
            " Answer a number from the range from 0 to 95."
            " In case the answer is larger then 95, answer 95 as maximum significant difference."
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.answer_template: str = "Answer is"
        self.add_tokens_to_tokenizer = True

        # Required to specify available fetaure value range
        self.feature_min = 0.0
        self.feature_max = 95.0

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
        # For Decoders:
        # [<s> + task_special_token + Q_start + [trx] + transactions_tokens + [/trx] + Q_end + Answer + </s>]
        question_start_tokens = self.custom_tokenize(question_start,
                                                     add_special_tokens=True,
                                                     return_tensors='pt')['input_ids']
        if question_start_tokens[:, -1] == self.tokenizer.eos_token_id:
            question_start_tokens = question_start_tokens[:, :-1]
        question_start_tokens = question_start_tokens.to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             add_special_tokens=False,
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
                                                    add_special_tokens=True,
                                                    truncation=True)
        target_encoded_ids = target_encoded_batch['input_ids'].to(device)
        batch_answer_mask = target_encoded_batch['attention_mask'].to(device)
        if target_encoded_batch['input_ids'][0, 0] == self.tokenizer.bos_token_id:
            target_encoded_ids = target_encoded_ids[:, 1:]  # strip EOS from beginnings, but keep EOS
            batch_answer_mask = batch_answer_mask[:, 1:]

        # Answer template encoding + strip </s> (EOS) token
        answer_template_encoded = self.custom_tokenize(self.answer_template,
                                                       return_tensors='pt',
                                                       add_special_tokens=False,
                                                       return_attention_mask=True)
        answer_template_mask = answer_template_encoded['attention_mask'].to(device)
        answer_template_encoded = answer_template_encoded['input_ids'].to(device)

        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        batch_answer_template_mask = answer_template_mask.repeat(batch_size, 1)

        # Answer template encoding + target tokens + EOS token
        batch_answer_encoded = torch.cat([batch_answer_template_encoded,
                                          target_encoded_ids], dim=1).long().to(device)
        # Answer mask
        batch_answer_mask = torch.cat([batch_answer_template_mask,
                                       batch_answer_mask], dim=1).long().to(device)

        return dict(
            question_start_tokens=question_start_tokens,
            question_start_tokens_mask=question_start_tokens_mask,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            question_end_attention_mask=question_target_encoded_batch['attention_mask'],
            target_tokens=target_encoded_ids,
            target_attention_mask=batch_answer_mask,
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
            # batch['mask'][i, last_feature_index] = 0
            self.mask_single_transaction(batch, i, last_feature_index, 0)

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

        processed_outputs = dict(targets=torch.FloatTensor(targets),
                                 predictions=torch.FloatTensor(predictions))
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