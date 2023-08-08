import torch
import random

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

from torchmetrics import AUROC, Perplexity
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score, F1Score, Accuracy)
from romashka.transactions_qa.tasks.categorical_task_abstract import CategoricalTaskAbstract

from romashka.transactions_qa.model.generation_utils import isin
from romashka.transactions_qa.evaluation.eval_processings_utils import (map_prediction_to_answer, transform_labels)


@dataclass
class PredHourTaskBinary(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "pred_hour_binary"
        self.target_feature_name = 'hour'  # 24 unique values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_hour_binary]"

        self.num_classes = 24
        self.is_text_task = False
        self.is_binary_task = True
        self.is_open_ended_task = False

        self.metrics = {
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score()
        }

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]
        self.ending_prompts = [
            ". Will the the hour of the next transaction be equal to %s? Yes or No?",
            ". Will the upcoming transaction hour of the day be equal to %s? Choose one: Yes or No?",
            ". Is it true that the hour of next transaction will be equal to %s? Yes or No?",
            ". Define whether the following statement is true: in next transaction the hour of day will be equal to %s."
            "bChoose: Yes or No?",
            ". Is it true or false: the hour of the upcoming transaction will be %s? Yes or No?",
            ". Define whether the following statement is correct: in the next transaction hour will be %s. "
            "Choose: Yes or No?",
            ". Identify if the statement that: the hour of the next transaction will be equal to %s, "
            "is correct? Yes or No?",
            ". Determine whether the following statement is true: %s will be the hour of the day in the upcoming "
            "transaction. Choose: Yes or No?",
            ". Is the statement correct: the hour of the next transaction will be %s. "
            "Answer with one of the following options: Yes or No?",
            ". Answer the question whether or not the following statement is true: the hour of the next "
            "transaction will be equal to %s. Yes or No?",
            ". Answer the question: will the hour of the upcoming transaction be equal to %s? "
            "Choose only one of the following options: Yes or No?"
        ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(self.num_classes)]
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answers2tokens = {answer: self.tokenizer.encode(answer_word, add_special_tokens=False)[0]
                               for answer, answer_word in self.binary_answer_options.items()}
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
        batch_size = batch['mask'].shape[0]

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
            target_feature_value_batch.append(last_feature.to(device))
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # Map to strings
        target_feature_value_batch = list(map(lambda x: str(x.item()), target_feature_value_batch))

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
        for target, pos_neg_mask in zip(target_feature_value_batch,
                                                       pos_neg_target_mask):
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

        return question_target_batch, target_batch

    def process_outputs(self, outputs: Any, answers: torch.Tensor, as_strings: Optional[bool] = False) -> Any:
        """
        Processing target text and output text to get the predictions
        """
        # Get predictions as list of strings
        if as_strings:
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

        else:

            # Get predictions as probabilities of binary answer
            probabilities_over_vocab = torch.nn.functional.softmax(outputs['logits'], dim=2)

            # answer structure: [..., answer_token, ..., </s>]
            targets_tokens = answers[isin(answers, torch.LongTensor(list(self.answers2tokens.values())))]
            targets = (targets_tokens == self.answers2tokens['positive']).long()

            answer_tokens_indexes = torch.nonzero(isin(outputs['logits'].argmax(2),
                                                       torch.LongTensor(list(self.answers2tokens.values()))),
                                                  as_tuple=True)

            preds_probs, preds = torch.max(probabilities_over_vocab, -1)
            positive_probs = probabilities_over_vocab[answer_tokens_indexes][:, self.answers2tokens['positive']]
            negative_probs = probabilities_over_vocab[answer_tokens_indexes][:, self.answers2tokens['negative']]
            pos_neg_probs = torch.cat([positive_probs.unsqueeze(-1), negative_probs.unsqueeze(-1)], 1)
            predictions = torch.sigmoid(pos_neg_probs[:, 0] - pos_neg_probs[:, 1])

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

            if 'f1' in task_metrics:
                task_metrics['f1'](preds, targets)
                metrics['f1'] = task_metrics['f1']

        except Exception as e:
            print(f"Error during metrics calculation: {e}")

        return metrics


@dataclass
class PredHourTaskOpenEnded(CategoricalTaskAbstract):

    def __post_init__(self):
        self.task_name = "pred_hour_open-ended"
        self.target_feature_name = 'hour'  # 108 unique values

        self.task_special_token = None
        self.task_specific_special_token = "[pred_hour_openended]"

        self.num_classes = 24
        self.is_text_task = False
        self.is_binary_task = False
        self.is_open_ended_task = True
        self.metrics = torch.nn.ModuleDict({
            'accuracy': Accuracy(task='multiclass',
                                 threshold=self.decision_threshold,
                                 average='weighted',
                                 ignore_index=self.ignore_class_index,
                                 num_classes=self.num_classes),
            "ppl": Perplexity(ignore_index=-100)
        })

        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]
        self.ending_prompts = [
            ". On which hour will the client make the next transaction?"
            " Answer as an index of an hour in 24-hour format, starting from 0 to 23 inclusive."
        ]
        # self.ending_prompts = [
        #     ". On which hour will the client make the next transaction?"
        #     ". What is the hour of the next transaction?",
        #     ". What is the hour of the next transaction based on the provided transaction history?",
        #     ". Choose the hour of upcoming transaction.",
        #     ". Select the hour of the day on which client will make the next transaction?",
        #     ". Find out what is the hour of upcoming transaction.",
        #     ". Determine the hour of the next transaction.",
        #     ". Choose the hour of the next transaction based on provided history.",
        #     ". Can you find out on which hour will be the next transaction?",
        #     ". Answer the question: on which hour will the client make the next transaction?",
        #     ". Answer the following question: on which hour of the day will the client make the next transaction?",
        # ]

        self.question_templates = self.generate_question_templates(self.starting_prompts,
                                                                   self.ending_prompts)

        # all options for a target feature - it is not actually required here, but still
        self.answers_options = [str(i) for i in range(self.num_classes)]
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
        batch_size = batch['mask'].shape[0]

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
            target_feature_value_batch.append(last_feature.to(device))
            # Mask last feature to predict it!
            batch['mask'][i, last_feature_index] = 0

        # Map to strings
        target_batch = list(map(lambda x: str(x.item()), target_feature_value_batch))

        # Construct target sequences
        question_target_batch = [question_end for _ in range(batch_size)]  # as strings

        return question_target_batch, target_batch

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