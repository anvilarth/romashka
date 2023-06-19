import re
import random
import inflect

from tkinter import TRUE
import numpy as np
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from src.transactions_qa.layers.connector import (make_linear_connector,
                                                       make_recurrent_connector)
from src.tasks.task_abstract import AbstractTask
from src.utils.logging_handler import get_logger
from src.transactions_qa.utils import (get_split_indices, prepare_splitted_batch, collate_batch_dict, get_exponent_number, get_mantissa_number)
from src.transactions_qa.layers.numerical_head import LinearHead, MLPHead
from src.models.components.perceiver_pytorch.perceiver_model import Attention
from src.models.components.embedding import NumEmbedding

from transformers import Adafactor, GenerationConfig
from transformers.optimization import AdafactorSchedule

from copy import deepcopy

class TransactionQAModel(pl.LightningModule):
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 tasks: List[AbstractTask],
                 connector: Optional[nn.Module] = None,
                 connector_input_size: Optional[int] = None,
                 connector_output_size: Optional[int] = None,
                 checkpoint_dir: Optional[str] = None,
                 do_freeze_tm: Optional[bool] = True,
                 do_freeze_lm: Optional[bool] = False,
                 do_freeze_connector: Optional[bool] = False,
                 optimizer_name: Optional[str] = 'AdamW',
                 scheduler_name: Optional[str] = 'linear_schedule_with_warmup',
                 learning_rate: Optional[float] = 5e-5,
                 weight_decay: Optional[float] = 0.0,
                 scale_parameter: Optional[bool] = False,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000,
                 use_numerical_input: Optional[bool] = False,
                 use_numerical_output: Optional[bool] = False,
                 verbose_for_debug: Optional[bool] = False,
                 num_head: Optional[str] = 'linear',
                 numerical_context: Optional[str] = 'context',
                 mantissa_weight: Optional[float] = 1.0,
                 number2text: Optional[bool] = False,
                 ):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.transaction_model = transaction_model
        self.connector = make_linear_connector(
            input_size=connector_output_size,
            output_size=connector_input_size,
            embedding_model=self.transaction_model,
            autoregressive_model=self.language_model) \
            if connector is None else connector

        self.use_numerical_input = use_numerical_input
        self.use_numerical_output = use_numerical_output
        self.numerical_context = numerical_context
        self.mantissa_weight = mantissa_weight
        self.mantissa_loss = nn.L1Loss()
        self.exponent_loss = nn.CrossEntropyLoss()
        self.number_engine = inflect.engine()
        self.numbers2text = number2text

        dim = self.connector.get_input_size()

        self.num_embedding = None
        if use_numerical_input:
            self.input_num_token = self.tokenizer('<extra_id_0>', add_special_tokens=False, return_tensors='pt').input_ids.item()
            buckets = torch.linspace(1, 10, steps=20)
            self.num_embedding = NumEmbedding(dim, buckets)

        if use_numerical_output:
            self.num_context = Attention(dim, dim, 1, dim)
            self.agg_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            self.num_context = nn.Identity()

        if num_head == 'linear':
            self.exponent_head = LinearHead(dim, 17)
            self.mantissa_head = LinearHead(dim, 1)
            
        elif num_head == 'mlp':
            self.exponent_head = MLPHead(dim, 17)
            self.mantissa_head = MLPHead(dim, 1)

        else:
            raise NotImplementedError

        self.training_mode = 'lm'
        
        self.tasks = tasks
        self.checkpoint_dir = checkpoint_dir

        self._logger.info(f"Setuping metrics.")
        self.val_metrics = nn.ModuleDict({task.task_name: deepcopy(task.metrics) for task in self.tasks})
        self.test_metrics = nn.ModuleDict({task.task_name: deepcopy(task.metrics) for task in self.tasks})          

        print(self.val_metrics)
        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scale_parameter = scale_parameter
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name

        self.do_freeze_tm: bool = do_freeze_tm
        self.do_freeze_lm: bool = do_freeze_lm
        self.do_freeze_connector: bool = do_freeze_connector
        self._is_multitask: bool = False
        self._is_encoder_decoder: bool = False

        self._verbose_for_debug: bool = verbose_for_debug
        self._prepare_model()

        self.save_hyperparameters(ignore=['tasks', '_logger', 'columns',
                                          'transaction_model', 'language_model', 'connector',
                                          'log_eval_predictions_table', 'log_eval_steps_counter'])

        # ✨ W&B: Create a Table to store predictions for each test step
        self.columns = ["epoch", "step #", "task",
                        "question", "prediction", "truth",
                        "transactions_history_lengths"]

        self.log_eval_predictions_table = wandb.Table(columns=self.columns)
        self.log_eval_steps_counter = 0
        self.num_eval_batches_to_log = 10

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Returns:
             **Dictionary**, with an "optimizer" key, and (optionally) a "lr_scheduler"
              key whose value is a single LR scheduler or `lr_scheduler_config`.
        """
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        if self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay,
                                        eps=1e-8)

        elif self.optimizer_name == 'Adafactor':
            optimizer = Adafactor(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                scale_parameter=self.scale_parameter,
                relative_step=False,
                warmup_init=False,
            )
        else:
            raise NotImplementedError

        if self.scheduler_name == 'linear_schedule_with_warmup':
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=self.warmup_steps,
                                                                 num_training_steps=self.training_steps
                                                                 )  # was: 10_000 * 20
        elif self.scheduler_name == "adafactor":
            scheduler = AdafactorSchedule(optimizer, self.learning_rate)
        
        else:
            raise NotImplementedError

        self._logger.info(f"Training with {self.optimizer_name}-lr={self.learning_rate} and {self.scheduler_name}.")

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _prepare_model(self):
        if not len(self.tasks):
            raise AttributeError(f"For training at least one task should be specified!")
        elif len(self.tasks) > 1:
            self._is_multitask = True
            self._logger.info(f"Running in `multi task` setting with {len(self.tasks)} tasks provided.")
        else:
            self._is_multitask = False
            self._logger.info(f"Running in `single task` setting"
                              f"with a single task: {self.tasks[0].task_name} provided.")

        # Figure out what the model type passed% encoder-decoder / decoder-only
        self._set_model_type()

        # In case any of tasks extends initial tokenizer vocab with additional tokens
        self._resize_text_embeddings()

        # Freezing some weights
        if self.do_freeze_tm:
            self._logger.info(f"Freezing transaction model's parameters...")
            for param in self.transaction_model.parameters():
                param.requires_grad = False

        if self.do_freeze_lm:
            self._logger.info(f"Freezing language model's parameters...")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if self.do_freeze_connector:
            self._logger.info(f"Freezing connector layer's parameters...")
            for param in self.connector.parameters():
                param.requires_grad = False

    def _set_model_type(self):
        # For encoder-decoder models
        if hasattr(self.language_model, "encoder"):
            self._is_encoder_decoder = True
        # For decoder-only
        elif hasattr(self.language_model, "transformer"):
            self._is_encoder_decoder = False
        else:
            raise NotImplementedError(f"Unknown model type: {type(self.language_model)}")

        self._logger.info(f"Language model type: `{'encoder-decoder' if self._is_encoder_decoder else 'decoder'}`")

    def _resize_text_embeddings(self):
        # For encoder-decoder models
        if self._is_encoder_decoder:
            init_embeddings = self.language_model.encoder.get_input_embeddings()
        # For decoder-only
        else:
            init_embeddings = self.language_model.transformer.get_input_embeddings()

        self._logger.info(f"LM initial `num_embeddings`: {init_embeddings.num_embeddings}, "
                          f"`embedding_dim`: {init_embeddings.embedding_dim}")

        self.language_model.resize_token_embeddings(len(self.tokenizer))

        # For encoder-decoder models
        if self._is_encoder_decoder:
            resized_embedds = self.language_model.encoder.get_input_embeddings()
            # For decoder-only
        else:
            resized_embedds = self.language_model.transformer.get_input_embeddings()

        self._logger.info(f"LM resized `num_embeddings`: {resized_embedds.num_embeddings}, "
                          f"`embedding_dim`: {resized_embedds.embedding_dim}")

    def tokenize_texts(self, batch):
        batch_size = batch['mask'].shape[0]
        device = batch['mask'].device
        transactions_embedding_mask = batch['mask']

        ### Tokenizing question start
        
        question_start_encoded = self.tokenizer(batch['question_start_string'],
                                        padding=True,
                                        return_tensors='pt',
                                        add_special_tokens=False).to(device)

        question_start_tokens, question_start_tokens_mask = question_start_encoded.input_ids, question_start_encoded.attention_mask
        ### Tokenizing question end

        num_values = None

        if self.numbers2text:
            batch['question_end_string'] = [re.sub("\d+\.\d+", lambda x: self.number_engine.number_to_words(x.string[x.start():x.end()], decimal='dot'), string) for string in batch['question_end_string']]
       
        elif self.use_numerical_input:
            detected_nums = [re.findall("\d+\.\d+", question) for question in batch['question_end_string']]
            batch['question_end_string'] = [re.sub("\d+\.\d+", '<extra_id_0>', string) for string in batch['question_end_string']]

            values = []
            for value_list in detected_nums:
                for value in value_list:
                    values.append(float(value))

            num_values = torch.tensor(values, device=device)

        question_target_encoded_batch = self.tokenizer(batch['question_end_string'],
                                                    padding=True,
                                                    return_tensors='pt',
                                                    add_special_tokens=False).to(device)

        question_end_tokens_mask = question_target_encoded_batch.attention_mask


        encoder_input_mask = torch.cat(
                                        [question_start_tokens_mask, 
                                        transactions_embedding_mask, 
                                        question_end_tokens_mask], dim=1
        )

        ###
        # Not using add_skip_special_tokens to keet tensor in Long format
        answer_template_encoded = self.tokenizer(batch['answer_start_string'],
                                                        padding=True,
                                                        return_tensors='pt'
                                                        ).to(device)

        target_encoded_batch = self.tokenizer(batch['answer_target_string'],
                                        padding=True,
                                        return_tensors='pt',
                                        add_special_tokens=False).to(device)

        batch_answer_encoded = torch.cat([answer_template_encoded.input_ids[:, :-1],
                                          target_encoded_batch.input_ids], dim=1)
        
        batch_answer_mask = torch.cat([answer_template_encoded.attention_mask[:, :-1],
                                       target_encoded_batch.attention_mask], dim=1)

        eos_mask = torch.ones(batch_size, 1, dtype=torch.long, device=device) 
        eos_tokens = eos_mask * self.tokenizer.eos_token_id

        return dict(
            question_start_tokens=question_start_tokens,
            question_end_tokens=question_target_encoded_batch['input_ids'],
            target_tokens=target_encoded_batch['input_ids'],
            answer_tokens=batch_answer_encoded,  # template + targets
            answer_mask=batch_answer_mask,
            question_start_tokens_mask=question_start_tokens_mask,
            question_end_tokens_mask=question_end_tokens_mask,
            encoder_input_mask=encoder_input_mask,
            eos_tokens_mask=eos_mask,
            eos_tokens=eos_tokens,
            num_values=num_values,
        )
        
    def construct_lm_input(self,qa_batch, batch):
        question_start_embeddings_batch = self.language_model.encoder.embed_tokens(
            qa_batch['question_start_tokens'])  # call for (embed_tokens): Embedding(32128, 512)
        # question_end_tokens: torch.Size([batch_size, len(max_question_end_tokens)])

        question_end_embeddings_batch = self.language_model.encoder.embed_tokens(
                qa_batch['question_end_tokens'])

        if self.num_embedding is not None:
            if self.num_embedding.mantissa_embedding.buckets.device != batch['mask'].device:
                self.num_embedding.mantissa_embedding.buckets = self.num_embedding.mantissa_embedding.buckets.to(batch['mask'].device)
                self.num_embedding.mantissa_embedding.matrix = self.num_embedding.mantissa_embedding.matrix.to(batch['mask'].device)
                self.num_embedding.mantissa_embedding.bucket_sizes = self.num_embedding.mantissa_embedding.bucket_sizes.to(batch['mask'].device)

        if self.use_numerical_input:
            numerical_mask = qa_batch['question_end_tokens'] == self.input_num_token
            question_end_embeddings_batch[numerical_mask] += self.num_embedding(qa_batch['num_values'])
             # call for (embed_tokens): Embedding(32128, 512)
        

        transactions_embeddings = self.transaction_model.get_embs(batch)[0]

        # next pass them to connector == linear mapping -> to LM inner dim
        transactions_embeddings = self.connector(transactions_embeddings)

        # Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)
        if 'encoder_input_mask' in qa_batch:
            encoder_input_mask = qa_batch['encoder_input_mask']

        else:
            encoder_input_mask = torch.cat(
                [qa_batch['question_start_tokens_mask'],
                batch['mask'],
                qa_batch['question_end_tokens_mask']],dim=1
                )

        # Create answers + masks for LM's decoder inputs
        batch_answers = qa_batch['answer_tokens']
        # was: torch.cat([qa_batch['answer_template_tokens'], qa_batch['target_tokens']], dim=1)
        batch_answers_mask = qa_batch['answer_mask']
        # torch.cat([qa_batch['answer_mask'], qa_batch['target_attention_mask']], dim=1)

        return dict(inputs_embeds=encoder_input,
                    attention_mask=encoder_input_mask,
                    labels=batch_answers,
                    decoder_attention_mask=batch_answers_mask,
                    question_end_embeds=question_end_embeddings_batch,
                    question_end_mask=qa_batch['question_end_tokens_mask']
        )
    
    def get_few_shot_batch(self, qa_batch, batch):
        question_start_embeddings_batch = self.language_model.encoder.embed_tokens(qa_batch['question_start_tokens'])  
        # removing <EOS> token
        question_end_embeddings_batch = self.language_model.encoder.embed_tokens(qa_batch['question_end_tokens']) 
        question_answer_embeddings_batch = self.language_model.encoder.embed_tokens(qa_batch['answer_tokens']) 
        question_eos_embeddings_batch = self.language_model.encoder.embed_tokens(qa_batch['eos_tokens']) 

        # Get transaction embedding
        transactions_embeddings = self.transaction_model.get_embs(batch)[0]
        transactions_embeddings = self.connector(transactions_embeddings)

        # Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                    transactions_embeddings,
                                    question_end_embeddings_batch,
                                    question_answer_embeddings_batch], dim=1)

        encoder_input_mask = torch.cat([
            qa_batch['question_start_tokens_mask'],
            batch['mask'],
            qa_batch['question_end_tokens_mask'],
            qa_batch['answer_mask'],
            qa_batch['eos_tokens_mask']],dim=1
            )

        # Create answers + masks for LM's decoder inputs
        batch_answers = qa_batch['answer_tokens'][-1]
        # was: torch.cat([qa_batch['answer_template_tokens'], qa_batch['target_tokens']], dim=1)
        batch_answers_mask = qa_batch['answer_mask'][-1]

        res = encoder_input.flatten(0,1)[:-(question_answer_embeddings_batch.shape[1] + qa_batch['eos_tokens_mask'].shape[1])].unsqueeze(0)
        return dict(inputs_embeds=res,
                    attention_mask=encoder_input_mask,
                    labels=batch_answers,
                    decoder_attention_mask=batch_answers_mask
        )

    def model_step(self, batch, generate=False, task_idx: Optional[int] = None) -> Tuple[Any, torch.Tensor]:
        # Sample single task
        # if task_idx is None:
        #     task_idx = 0
        # task = self.tasks[task_idx]


        if task_idx is not None:
            new_batch = self.tasks[task_idx].process_input_batch(batch)

            if len(new_batch) == 0:
                return None, None
        else: 
            NUM_TASKS = len(self.tasks)
            splitted = get_split_indices(batch, len(self.tasks))
            new_batch = np.random.choice(NUM_TASKS, size=len(splitted), replace=False)
            task_ids = np.random.choice(NUM_TASKS, size=len(splitted), replace=False)

            batches = []
            for i, split_indices in enumerate(splitted):
                task = self.tasks[task_ids[i]]
                subbatch = prepare_splitted_batch(batch, split_indices)
                tmp_batch = task.process_input_batch(subbatch)
                if len(tmp_batch) == 0:
                    continue

                batches.append(tmp_batch)
            
            if len(batches) == 0:
                return None, None
            new_batch = collate_batch_dict(batches)

        qa_batch = self.tokenize_texts(new_batch)

        transactions_history_lengths = batch['mask'].sum(1)

        if self.training_mode == 'few_shot':
            model_input = self.get_few_shot_batch(qa_batch, batch)
        elif self.training_mode == 'lm':
            model_input = self.construct_lm_input(qa_batch, batch)
        else:
            raise NotImplementedError

        # if self.numerical_context == 'context' and self.use_numerical:
        #     batch_size = len(model_input['labels'])
            
        #     agg_token = self.agg_token.repeat(batch_size, 1, 1)
        #     model_input['inputs_embeds'] = torch.cat([model_input['inputs_embeds'], agg_token], dim=1)
        #     num_mask = torch.ones(batch_size, 1, device=model_input['attention_mask'].device)
        #     model_input['attention_mask'] = torch.cat([model_input['attention_mask'], num_mask], dim=1)


        # Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
     
        lm_outputs = self.language_model(inputs_embeds=model_input['inputs_embeds'],
                                         attention_mask=model_input['attention_mask'],
                                         labels=model_input['labels'],
                                         decoder_attention_mask=model_input['decoder_attention_mask'],
                                         output_hidden_states=True,
                                        )
        if self.use_numerical_output:
            # TODO Fix if we will predict several numbers
            num_token_mask = (model_input['labels'] == self.tokenizer.convert_tokens_to_ids('<NUM>')).any(dim=1)
            if self.numerical_context == 'context':
                embedding  = lm_outputs.encoder_last_hidden_state[:, -1][num_token_mask]
            
            elif self.numerical_context == 'simple':
                context = lm_outputs.encoder_last_hidden_state[num_token_mask]
                embedding = lm_outputs['decoder_hidden_states'][-1][num_token_mask]

                if embedding.shape[1] != 1:
                    raise NotImplementedError
                embedding = self.num_context(embedding, context, mask=model_input['attention_mask'])[:, 0]
            
            else:
                raise NotImplementedError

            exponent = self.exponent_head(embedding)
            mantissa = self.mantissa_head(embedding).squeeze(-1)
            
            lm_outputs['exponent'] = exponent
            lm_outputs['mantissa'] = mantissa
            
            # On our torch version torch.pow(10, 0) = 0, but torch.pow(10, 0.0) = 1 WTF
            lm_outputs['preds'] = 10 ** (exponent.argmax(-1).float() - 8) * mantissa
            
        # Return question as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        lm_outputs['question_encoded'] = torch.cat([qa_batch['question_start_tokens'],
                                                    qa_batch['question_end_tokens']], dim=1)
        # Experimental !
        lm_outputs['transactions_history_lengths'] = transactions_history_lengths

        # lm_outputs['question_start_input_size'] = question_start_embeddings_batch.size(1)
        # lm_outputs['question_end_input_size'] = question_end_embeddings_batch.size(1)
        # lm_outputs['transactions_input_size'] = transactions_embeddings.size(1)
        lm_outputs['total_input_size'] = model_input['inputs_embeds'].size(1)
        lm_outputs['label'] = new_batch['raw_labels']

        return lm_outputs, model_input['labels']

    def training_step(self, batch, batch_idx: Optional[int] = None) -> Optional[Any]:
        r"""
        Compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch: The output of torch.utils.data.DataLoader. A tensor, tuple or list.
            batch_idx (`int`): Integer displaying index of this batch

        Return:
            Any of.
        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
        """
        # Sample a random single task
        task_idx = random.sample(list(range(len(self.tasks))), k=1)[0]
        task_name = self.tasks[task_idx].task_name

        outputs, answer = self.model_step(batch, task_idx=task_idx)
        if outputs is None:
            return None

        loss = outputs.loss

        if self.use_numerical_output:
            mantissa = outputs['mantissa']
            exponent = outputs['exponent']

            true_exponent = get_exponent_number(outputs['label']).long() + 8
            true_mantissa = get_mantissa_number(outputs['label'])

            mantissa_loss = self.mantissa_loss(mantissa / 10, true_mantissa / 10)
            exponent_loss = self.exponent_loss(exponent, true_exponent)

            loss += self.mantissa_weight * mantissa_loss + exponent_loss

        logging_dict = {
            'train_loss': loss,
            f'{task_name}_train_loss': loss,
        }
        if self._verbose_for_debug:
            additional_logging_dict = self._collect_additional_info(outputs)
            if additional_logging_dict is not None and len(additional_logging_dict):
                logging_dict = dict(list(logging_dict.items()) + list(additional_logging_dict.items()))

        self.log_dict(logging_dict)
        # self._logger.info(f"Train step results:\n{logging_dict}")
        return loss

    def _collect_additional_info(self, outputs: Any) -> Dict[str, Any]:
        """
        Collect additional information from model's outputs for debugging.
        Calling this function is optional - it can be removed without ane troubles.
        Args:
            outputs: model's step outputs;
        Returns:
            a collected dictionary with information about a step.
        """
        logging_dict = {}
        if 'question_start_input_size' in outputs:
            logging_dict['question_start_input_size'] = outputs['question_start_input_size']

        if 'question_end_input_size' in outputs:
            logging_dict['question_end_input_size'] = outputs['question_end_input_size']

        if 'transactions_input_size' in outputs:
            logging_dict['transactions_input_size'] = outputs['transactions_input_size']

        if 'total_input_size' in outputs:
            logging_dict['total_input_size'] = outputs['total_input_size']

        # if 'transactions_history_lengths' in outputs:
        #     logging_dict['transactions_history_lengths'] = outputs['transactions_history_lengths'].detach()

        if self._verbose_for_debug:
            print(f"Additional info from a step:")
            print(logging_dict)
        return logging_dict

    def validation_step(self, batch,
                        batch_idx: Optional[int],
                        dataloader_idx: Optional[int] = None, **kwargs) -> Optional[Any]:
        r"""
        Operates on a single batch of data from the validation set.
        Args:
            batch: The output of your torch.utils.data.DataLoader.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used)

        Return:
            - Any object or value
        """
        # Sample a random single task
        task_idx = random.sample(list(range(len(self.tasks))), k=1)[0]
        task = self.tasks[task_idx]

        outputs, batch_answers = self.model_step(batch, task_idx=task_idx)
        
        if outputs is None:
            return None

        loss = outputs.loss

        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(batch_answers,
                                                            skip_special_tokens=True)

        batch_questions_decoded = self.tokenizer.batch_decode(outputs.question_encoded.detach().cpu(),
                                                              skip_special_tokens=True)

        # for i, (pred, answer, question) in enumerate(zip(predictions_decoded, batch_answers_decoded, batch_questions_decoded)):
        #     print(f"\t#{i}{question}:\tpredicted: {pred}, answer: {answer}")


        # Calc metrics
        try: 
            metrics_scores = task.calculate_metrics(outputs, batch_answers, self.val_metrics[task.task_name], stage='val_')
        except Exception as e:
            self._logger.error(f"error occurred during task metric calculation:\n{e}")

        logging_dict = {
            'val_loss': loss,
            f'{task.task_name}_val_loss': loss
        }
        if self._verbose_for_debug:
            additional_logging_dict = self._collect_additional_info(outputs)
            if additional_logging_dict is not None and len(additional_logging_dict):
                logging_dict = dict(list(logging_dict.items()) + list(additional_logging_dict.items()))

        logging_dict = dict(list(logging_dict.items()) + list(metrics_scores.items()))
        # self._logger.info(f"Validation step results:\n{logging_dict}")
        self.log_dict(
            logging_dict,
            batch_size=batch_answers.size(0)
        )

        # Log predictions on validation set
        if self.log_eval_steps_counter < self.num_eval_batches_to_log:
            self.log_predictions(logits=outputs.logits.detach().cpu(),
                                 answers=batch_answers.detach().cpu(),
                                 questions=outputs.question_encoded.detach().cpu(),
                                 transactions_history_lengths=outputs['transactions_history_lengths'].detach().cpu(),
                                 predictions_table=self.log_eval_predictions_table,
                                 log_counter=self.log_eval_steps_counter,
                                 outputs=outputs)
            self.log_eval_steps_counter += 1
        return loss

    def test_step(self, batch:Any, batch_idx: int, dataloader_idx: int = 0):
        batch_size = batch['mask'].shape[0]

        for task_idx, task in enumerate(self.tasks):
            try:
                predictions = self._predict_step_task(batch,
                                                      batch_idx=batch_idx,
                                                      task_idx=task_idx,
                                                      verbose=False,
                                                      calculate_metrics=True)

                if predictions:
                    self.log_dict(
                            predictions['metrics'],
                            batch_size=batch_size,
                        )

            except Exception as e:
                self._logger.error(f"Error occurred during task `{task.task_name}` evaluation:\n{e}")
            


                
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Step function called during Trainer.predict().

        TODO: to use pytorch_lightning.callbacks.BasePredictionWriter callback
        to write the predictions to disk or database after each batch or on epoch end.

        Args:
            batch: Current batch.
            batch_idx: Index of current batch.
            dataloader_idx: Index of the current dataloader.
        Return:
            Predicted output
        """
        # Predict for each task !!!
        # Calc metrics
        tasks_predictions = {}
        for task_idx, task in enumerate(self.tasks):
            try:
                predictions = self._predict_step_task(batch,
                                                      batch_idx=batch_idx,
                                                      task_idx=task_idx,
                                                      verbose=True,
                                                      calculate_metrics=False)

                tasks_predictions[task.task_name] = predictions
            except Exception as e:
                self._logger.error(f"Error occurred during task `{task.task_name}` evaluation:\n{e}")

        return tasks_predictions

    def _predict_step_task(self, batch: Any, task_idx: int,
                           calculate_metrics: Optional[bool] = False,
                           verbose: Optional[bool] = False,
                           batch_idx: Optional[int] = 0) -> Dict[str, Any]:
        """
        Predict for single task.
        Args:
            batch: Current batch;
            task_idx: selected task index;
            batch_idx: Index of current batch.

        Returns:
            results: as dictionary, where:
                keys are - metrics / predictions / answers / questions.
        """
        task = self.tasks[task_idx]
        outputs, batch_answers = self.model_step(batch, task_idx=task_idx)
        
        if outputs is None:
            return dict()

        # as list of strings
        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(batch_answers,
                                                            skip_special_tokens=True)
        batch_questions_decoded = self.tokenizer.batch_decode(outputs.question_encoded.detach().cpu(),
                                                              skip_special_tokens=True)

        if verbose:
            print("----- Prediction step -----")
            for i, (pred, answer, question) in enumerate(
                    zip(predictions_decoded, batch_answers_decoded, batch_questions_decoded)):
                print(f"\t#{i}{question}:\tpredicted: {pred}, answer: {answer}")

        # Calc metrics
        metrics_scores = {}
        if calculate_metrics:
            try:
                metrics_scores = task.calculate_metrics(outputs, batch_answers, self.test_metrics[task.task_name], stage='test_')
            except Exception as e:
                self._logger.error(f"error occurred during task metric `{metric_name}` calculation:\n{e}")

        return dict(
            predictions=predictions_decoded,
            answers=batch_answers_decoded,
            questions=batch_questions_decoded,
            metrics=metrics_scores,
            batch_idx=batch_idx
        )

    def on_validation_epoch_start(self) -> None:
        print(f"\n----------- Validation start ----------\n")
        # Reset log counter
        self.log_eval_steps_counter = 0

    def on_fit_end(self) -> None:
        # ✨ W&B: Log predictions table to wandb
        wandb.log({"val_predictions": self.log_eval_predictions_table})
        # was directly to W&B: wandb.log({"val_predictions": self.log_eval_predictions_table})
        # ✨ W&B: Mark the run as complete (useful for multi-cell notebook)

    def log_predictions(self,
                        logits: torch.Tensor,
                        answers: torch.Tensor,
                        questions: torch.Tensor,
                        predictions_table: wandb.Table,
                        log_counter: int,
                        transactions_history_lengths: Optional[torch.Tensor] = [],
                        task_name: Optional[str] = "default",
                        outputs: Optional[torch.Tensor] = None):
        
    
        if self.use_numerical_output:
            predictions_decoded = list(map(lambda x: str(round(x.item(), 2)), outputs['preds']))
            answers_decoded = list(map(lambda x: str(round(x.item(), 2)), outputs['label']))

        else:
            predictions_decoded = self.tokenizer.batch_decode(logits.argmax(2),
                                                        skip_special_tokens=True)
        
            answers_decoded = self.tokenizer.batch_decode(answers,
                                                          skip_special_tokens=True)
        
        questions_decoded = self.tokenizer.batch_decode(questions,
                                                        skip_special_tokens=True)

        print(f"Validation predictions vs. answers, batch #{log_counter}:")

        # columns = ["epoch", "step #", "task", "question", "prediction", "truth", "transactions_history_lengths"]
        for i, (pred, answer, question) in enumerate(zip(predictions_decoded, answers_decoded, questions_decoded)):
            print(f"\t#{i}:\tpredicted: {pred}, answer: {answer}")
            predictions_table.add_data(self.current_epoch,
                                       "_".join([str(log_counter), str(i)]),
                                       task_name,
                                       question,
                                       pred,
                                       answer,
                                       transactions_history_lengths[i] if len(transactions_history_lengths) else 0)
