import torch
import random

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

from torchmetrics import Perplexity
from torchmetrics.classification import F1Score, Accuracy
from romashka.transactions_qa.tasks.categorical_task_abstract import CategoricalTaskAbstract
from romashka.transactions_qa.evaluation.eval_processings_utils import (map_prediction_to_answer,
                                                                        transform_labels)


@dataclass
class PredOpKindTaskOpenEnded(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "pred_operation_kind_open-ended"
        self.target_feature_name = 'operation_kind'  # 7 unique values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_operation_kind_openended]"

        self.num_classes = 8  # from 1 to 7 inclusive
        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True
        self.metrics = torch.nn.ModuleDict({
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
            "f1": F1Score(task="multiclass", num_classes=self.num_classes),
            "ppl": Perplexity(ignore_index=-100)
        })

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]
        self.ending_prompts = [
            ". What is the operation kind of the next transaction?"
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". What is the operation kind of the next transaction based on the provided transaction history?"
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Choose in which operation kind the upcoming transaction will be made."
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Select the operation kind of the next transaction."
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Find out what is the operation kind of upcoming transaction."
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Can you please answer the question: in which operation kind the next transaction will be made?"
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Determine the operation kind of the next transaction."
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Select the operation kind of the upcoming transaction based on provided history."
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Choose the operation kind of the next transaction."
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Can you find out in which operation kind will be the next transaction?"
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
            ". Answer the question: what is the operation kind of the upcoming transaction?"
            " Answer an index of an operation kind starting from 1 to 7 inclusive.",
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(1, self.num_classes)]
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
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
        question_end = self.transactions_embeddings_end_token + question_end + "\nThe answer is: "

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
        question_start_tokens = question_start_tokens.repeat(batch_size, 1).to(device)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        question_target_encoded_batch = self.custom_tokenize(question_target_batch,
                                                             return_tensors='pt',
                                                             padding=True,
                                                             truncation=True,
                                                             return_attention_mask=True
                                                             ).to(device)
        # Full input
        encoder_input = torch.cat([question_start_tokens, question_target_encoded_batch['input_ids']], 1)

        # Attention masks
        # already for full batch
        question_start_tokens_mask = torch.ones(question_start_tokens.size()).to(device)
        question_end_tokens_mask = question_target_encoded_batch['attention_mask']
        encoder_input_mask = torch.cat(
            [question_start_tokens_mask, question_end_tokens_mask],
            dim=1)

        # as dict(input_ids: torch.Tensor, attention_mask: torch.Tensor), padded to max_seq_len in batch
        # add [:, :-1] for no EOS tokens - ?
        # Answer template encoding + strip </s> (EOS) token
        target_encoded_batch = self.custom_tokenize(target_batch,
                                                    return_tensors='pt',
                                                    padding=True,
                                                    truncation=True).to(device)

        return dict(
            input_ids=encoder_input,
            attention_mask=encoder_input_mask,
            target_tokens=target_encoded_batch['input_ids'],
            target_attention_mask=target_encoded_batch['attention_mask']
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

        captions = batch['captions']  # as List[[str]] of shape [batch_size, 1, cap_len]

        # Construct target values
        target_feature_value_batch = []
        question_endings_batch = []
        for i, (feature_, mask_, cap_) in enumerate(zip(target_feature_batch, mask_batch, captions)):
            last_feature = feature_[-1]
            target_feature_value_batch.append(last_feature.to(device))
            # Construct target sequences
            # remove last transaction
            cap_ = "\n".join(cap_[0].split("\n")[:-1])
            question_endings_batch.append(cap_ + '\n' + question_end)

        # Map to strings
        target_batch = list(map(lambda x: str(x.item()), target_feature_value_batch))

        return question_endings_batch, target_batch

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