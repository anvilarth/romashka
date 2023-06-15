import torch
import random
import numpy as np

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy
from romashka.transactions_qa.tasks.categorical_task_abstract import CategoricalTaskAbstract
from romashka.transactions_qa.utils import get_buckets_info

from romashka.transactions_qa.evaluation.eval_processings_utils import map_prediction_to_answer


@dataclass
class PredDefaultTaskBinary(CategoricalTaskAbstract):
    """
    A task for predictive Binary QA task: given a transactions history predict the client's default happening,
    answer question with binary answer.
    """

    def __post_init__(self):
        self.task_name = "pred_default_binary"
        self.target_feature_name = 'label'  # [0, 1] range of values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_DEFAULT_binary]"

        self.is_text_task = False
        self.is_binary_task = True
        self.is_open_ended_task = False

        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_outputs: Optional[bool] = False

        self.metrics = {
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        }
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = [
            ". Will this client have an overdue loan? Yes or No?",
            ". Will this client be overdue on the credit? Choose one: Yes or No?",
            ". Will this client have a credit default? Yes or No?",
            ". Is it true that this client will have a credit default? Yes or No?",
            ". Define whether the following statement is true: this client will have an overdue on the credit. "
            "Choose: Yes or No?",
            ". Is it true or false: this client will have a credit default? Yes or No?",
            ". Define whether the following statement is correct: this client will have a credit default. "
            "Choose: Yes or No?",
            ". Identify if the statement that: this client will be overdue on the credit, "
            "is correct? Yes or No?",
            ". Determine whether the following statement is true: this client will have a credit default"
            ". Choose: Yes or No?",
            ". Is the statement correct: this client will have a credit default. "
            "Answer with one of the following options: Yes or No?",
            ". Answer the question whether or not the following statement is true: "
            "this client will have a credit default. Yes or No?",
            ". Answer the question: will this client be overdue on the credit? "
            "Choose only one of the following options: Yes or No?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = " "  # left empty for a first time
        self.add_tokens_to_tokenizer = True
        self.answers_options = [str(i) for i in range(2)]

        super().__post_init__()

        self.update_feature_index()
        print(f"\nTask `{self.task_name}` has\n\tTarget feature type: `{self.target_feature_type}` "
              f"\n\tTarget feature index: `{self.target_feature_index}`")

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
        batch_size = batch['mask'].shape[0]

        # Get target value
        target_feature_value_batch = batch[self.target_feature_type]  # with size of: [batch_size,]

        # Target's questions binary [No/Yes]
        target_batch = list(map(lambda x:
                                self.binary_answer_options['positive'] if x
                                else self.binary_answer_options['negative'],
                                target_feature_value_batch))

        # Construct target sequences
        # Use a default formatted question end template
        question_end = kwargs.get("question_end", "%s")
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

    def process_outputs(self, outputs: Any, answers: torch.Tensor) -> Any:
        """
        Processing target text and output text to get the predictions
        """
        # Get predictions as list of strings
        predictions_decoded = self.tokenizer.batch_decode(outputs['logits'].argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(outputs['labels'],
                                                            skip_special_tokens=True)
        # Map to answers
        predictions_decoded = [map_prediction_to_answer(t.lower(),
                                                        list(self.binary_answer_options.values()),
                                                        'no') for t in predictions_decoded]
        batch_answers_decoded = [map_prediction_to_answer(t.lower(),
                                                          list(self.binary_answer_options.values()),
                                                          'no') for t in batch_answers_decoded]
        target2index_mapping = {'yes': 1, 'no': 0}
        targets = torch.Tensor([target2index_mapping.get(answer, 0) for answer in batch_answers_decoded])
        predictions = torch.Tensor([target2index_mapping.get(pred, 0) for pred in predictions_decoded])

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

            if 'auc' in task_metrics:
                task_metrics['auc'](preds, targets)
                metrics['auc'] = task_metrics['auc']

            if 'accuracy' in task_metrics:
                task_metrics['accuracy'](preds, targets)
                metrics['accuracy'] = task_metrics['accuracy']

        except Exception as e:
            print(f"Error during metrics calculation: {e}")

        return metrics


