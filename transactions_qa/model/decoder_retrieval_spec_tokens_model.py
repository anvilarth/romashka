import numpy as np
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.utils import seed_everything
from romashka.transactions_qa.model.decoder_model import DecoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.tasks.task_token_updater import collect_task_specific_tokens
from romashka.transactions_qa.model.generation_utils import isin
from romashka.transactions_qa.losses.infonce_loss import InfoNCE
from romashka.transactions_qa.utils import (maybe_autocast, mask_lm_labels_padding)
from romashka.transactions_qa.layers.initialization import (init_embeddings_with_tensor,
                                                            init_parameter_with_tensor)

from romashka.transactions_qa.model.generation_utils import USE_HF_GENERATE


class DecoderRetrievalSpecTokensModel(DecoderSimpleModel):
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 tasks: Optional[List[AbstractTask]] = None,
                 connector: Optional[nn.Module] = None,
                 connector_input_size: Optional[int] = None,
                 connector_output_size: Optional[int] = None,
                 do_freeze_tm: Optional[bool] = True,
                 do_freeze_lm: Optional[bool] = False,
                 do_freeze_lm_embeddings: Optional[bool] = False,
                 do_freeze_connector: Optional[bool] = False,
                 min_ret_tokens: Optional[int] = 50,  # equals to max transactions history size
                 max_ret_tokens: Optional[int] = 150,  # equals to min transactions history size
                 n_retrieval_layers: Optional[List[int]] = None,
                 embeddings_dropout_p: Optional[float] = 0.1,
                 retrieval_loss_scale: Optional[float] = 1.0,
                 text_loss_scale: Optional[float] = 1.0,
                 add_temporal_embeddings: Optional[bool] = False,
                 transactions_embeddings_start_token: Optional[str] = r"[trx]",
                 transactions_embeddings_end_token: Optional[str] = r"[/trx]",
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):

        self._transactions_embeddings_start_token = transactions_embeddings_start_token
        self._transactions_embeddings_end_token = transactions_embeddings_end_token

        self.min_ret_tokens = min_ret_tokens
        self.max_ret_tokens = max_ret_tokens
        self._ret_tokens_template = "[RET_%s]"
        self.tasks = tasks if tasks is not None else []

        # self.projection_layers = nn.ModuleList([])
        self._n_retrieval_layers = n_retrieval_layers
        self._embeddings_dropout_p = embeddings_dropout_p

        self._retrieval_loss_scale = retrieval_loss_scale
        self._text_loss_scale = text_loss_scale

        self._add_temporal_embeddings = add_temporal_embeddings

        super().__init__(language_model=language_model,
                         transaction_model=transaction_model,
                         tokenizer=tokenizer,
                         connector=connector,
                         connector_input_size=connector_input_size,
                         connector_output_size=connector_output_size,
                         do_freeze_tm=do_freeze_tm,
                         do_freeze_lm=do_freeze_lm,
                         do_freeze_lm_embeddings=do_freeze_lm_embeddings,
                         do_freeze_connector=do_freeze_connector,
                         generation_config=generation_config,
                         is_debug=is_debug)

    def _prepare_model(self):
        super()._prepare_model()
        self.register_buffer("whitespace_token_id",
                             torch.Tensor(self.tokenizer.encode(' ', add_special_tokens=False)).long())

        self.lm_mean_embedding = self._create_mean_lm_embedding()

        self._create_trainable_parameters()
        self._create_losses()

        # Check if language model is frozen, optionally freeze
        self._logger.info(f"Check language model's parameters to be frozen...")
        for param_name, param in self.language_model.named_parameters():
            if param.requires_grad:
                self._logger.warning(f"Parameter `{param_name}` of LM requires grad, freezing..")
                param.requires_grad = False

        # Additionally re-assign embeddings
        self._set_language_model_embedding_func()

        # Check total trainable parameters
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        self._logger.info(f"Totally trainable parameters: {len(trainable_parameters)} from {len(parameters)}")

    def _create_trainable_parameters(self):
        """
        Creates trainable parameters for:
            - transactions embeddings start/end tokens: [trx] / [/trx];
            - retrieval tokens: RET_0 ... RET_N;
        Note: those parameters need to be passed to separate optimizer (with connector & projections layers)!
        i.e:
            opt = Adafactor(
                list(projection.parameters())
                + [trns_start_embedding, trns_end_embedding], lr=1e-2, relative_step=False)
        """
        if self.tokenizer.padding_side != 'right':
            self._logger.error("Tokenizer padding size should be set to `right`!")
            self.tokenizer.padding_side = 'right'

        if not self.tokenizer.add_eos_token:
            self._logger.error("Tokenizer `add_eos_token` should be set to `True`!")
            self.tokenizer.add_eos_token = True

        # Create transactions embeddings start/end tokens: [trx] / [/trx] and trainable parameters for them
        self._create_surrounding_parameters()

        # Create retrieval tokens: RET_0 ... RET_N in tokenizers vocabulary and mappings token <-> id
        self._create_retrieval_parameters()

        # Create trainable task-specific tokens
        self._create_trainable_task_special_tokens()

        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

        # Create projection layers from LM output hidden states to shared dim for loss calculation
        self._create_projection_layers()

        # Creates position embeddings layers (optionally)
        if self._add_temporal_embeddings:
            self._create_position_parameters()

    def _create_surrounding_parameters(self):
        """
        Creates trainable parameters for:
            - transactions embeddings start/end tokens: [trx] / [/trx];
        Note: those parameters need to be passed to separate optimizer (with connector & projections layers)!
        i.e:
            opt = Adafactor(
                list(projection.parameters())
                + [trns_start_embedding, trns_end_embedding], lr=1e-2, relative_step=False)
        """
        # Check if transactions embeddings start/end tokens, exists in tokenizers' vocabulary,
        # add them if not exist and get their indexes
        self.transactions_special_tokens_ids_mapping = AbstractTask.extend_vocabulary(
            new_tokens=[self._transactions_embeddings_start_token,
                        self._transactions_embeddings_end_token],
            tokenizer=self.tokenizer,
            return_ids=True,
            special=True
        )

        # Init transactions injection tokens ids
        self.transactions_start_token_id = self.transactions_special_tokens_ids_mapping.get(
            self._transactions_embeddings_start_token
        )
        self.transactions_end_token_id = self.transactions_special_tokens_ids_mapping.get(
            self._transactions_embeddings_end_token
        )

        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # Transactions embeddings start/end
        self.transactions_start_embedding = nn.Parameter(
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)),
            requires_grad=True).to(self.params_precision)
        self.transactions_end_embedding = nn.Parameter(
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)),
            requires_grad=True).to(self.params_precision)

        init_parameter_with_tensor(self.transactions_start_embedding, self.lm_mean_embedding)
        init_parameter_with_tensor(self.transactions_end_embedding, self.lm_mean_embedding)
        self._logger.info(f"Initialized trainable parameters for transactions embeddings start/end tokens.")

    def _create_retrieval_parameters(self):
        """
        Creates trainable parameters for:
            - retrieval tokens: RET_0 ... RET_N;
        Note: those parameters need to be passed to separate optimizer (with connector & projections layers)!
        i.e:
            opt = Adafactor(
                list(projection.parameters())
                + [trns_start_embedding, trns_end_embedding]
                + [ret_embeddings], lr=1e-2, relative_step=False)
        """
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ret_tokens = [self._ret_tokens_template % str(i) for i in range(self.max_ret_tokens)]

        # Check if transactions retrieval tokens, exists in tokenizers' vocabulary,
        # add them if not exist and get their indexes
        self.transactions_ret_tokens2ids_mapping = AbstractTask.extend_vocabulary(
            new_tokens=self.ret_tokens,
            tokenizer=self.tokenizer,
            return_ids=True,
            special=True
        )
        self._logger.info(f"Retrieval tokens added to tokenizer: {len(self.ret_tokens)}\ntokens: {self.ret_tokens}.")

        # Init transactions retrieval tokens ids -> token names mapping
        self.transactions_ret_ids2tokens_mapping = {token_id: token for token, token_id
                                                    in self.transactions_ret_tokens2ids_mapping.items()}

    def _create_trainable_task_special_tokens(self):
        """
        Creates trainable parameters for task special tokens.
        """
        self.task_special_tokens = collect_task_specific_tokens(self.tasks)
        self.register_buffer("task_special_tokens_ids",
                             self.tokenizer(self.task_special_tokens,
                                            add_special_tokens=False,
                                            padding=False,
                                            return_tensors='pt')['input_ids'].flatten())
        self._logger.info(f"Collected {len(self.task_special_tokens)} task special tokens: {self.task_special_tokens} "
                          f"with corresponding ids: {self.task_special_tokens_ids}")
        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # Embeddings for special tokens
        self.task_special_tokens_embeddings = nn.Embedding(num_embeddings=self.task_special_tokens_ids.size(0),
                                                           embedding_dim=params_dim).to(self.params_precision)
        init_embeddings_with_tensor(self.task_special_tokens_embeddings, self.lm_mean_embedding)

    def _create_position_parameters(self):
        """
        Creates position embeddings layers as the indicator of temporal information
        to the representations from different events in sequence.
        """
        self.temp_position_embedding = nn.Embedding(self.max_ret_tokens, 384)
        self._logger.info(f"Created position embeddings layers for maximum {self.max_ret_tokens} positions.")

    def _create_projection_layers(self):
        """
        Creates a linear mappings from language model hidden dimensionality
        to shared embeddings dimensionality for RET tokens loss calculation.
        """
        # List of indexes of hidden states to take for information extraction
        if self._n_retrieval_layers is None:
            self._n_retrieval_layers = [-1]

        self.projection_layers = nn.ModuleList([])

        shared_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            shared_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            shared_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where shared dimensionality for retrieval loss calculation"
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        for layer_idx in self._n_retrieval_layers:
            # Last layer hidden states
            if layer_idx == -1 or layer_idx == self.language_model.config.num_hidden_layers:
                if self.language_model_arch_type == "OPT":
                    in_dim = self.language_model.config.word_embed_proj_dim
                else:  # for GPT-like
                    in_dim = self.language_model.config.hidden_size
                # Maps from LM hidden_size -> shared dim
                text_fc = [nn.Linear(in_dim, shared_dim),  # , dtype=torch.float16
                           nn.Dropout(self._embeddings_dropout_p)]
                self.projection_layers.append(nn.Sequential(*text_fc))
            # Take representation from any middle layer
            elif layer_idx < self.language_model.config.num_hidden_layers:
                text_fc = [nn.Linear(self.language_model.config.hidden_size, shared_dim),  # , dtype=torch.float16
                           nn.Dropout(self._embeddings_dropout_p)]
                self.projection_layers.append(nn.Sequential(*text_fc))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only'
                    f' has {self.language_model.config.num_hidden_layers} layers.')

    def _create_losses(self):
        # Use CE for general QA text loss
        self.qa_loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

        # Use CE for RET tokens generation loss
        self.ret_CE_loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        # Use contrastive loss for embeddings comparison
        self.ret_NCE_loss_fn = InfoNCE()

    def has_start_token(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether transactions injection start token id already contained in given ids.
        """
        return (input_tokens_ids == self.transactions_start_token_id).sum() > 0

    def replace_start_token(self, input_tokens_ids: Union[List[int], torch.Tensor],
                            input_embeddings: torch.Tensor):
        """
        Replace transactions injection start tokens' embedding with trainable parameter.
        """
        mask = input_tokens_ids == self.transactions_start_token_id
        input_embeddings[mask] = self.transactions_start_embedding.to(
            input_embeddings.device).to(input_embeddings.dtype)

    def has_end_token(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether transactions injection end token id already contained in given ids.
        """
        return (input_tokens_ids == self.transactions_end_token_id).sum() > 0

    def replace_end_token(self, input_tokens_ids: Union[List[int], torch.Tensor],
                          input_embeddings: torch.Tensor):
        """
        Replace transactions injection end tokens' embedding with trainable parameter.
        """
        mask = input_tokens_ids == self.transactions_end_token_id
        input_embeddings[mask] = self.transactions_end_embedding.to(
            input_embeddings.device).to(input_embeddings.dtype)

    def has_task_tokens(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether any task special token id already contained in given ids.
        """
        return isin(input_tokens_ids, self.task_special_tokens_ids.to(input_tokens_ids.device)).sum() > 0

    def replace_task_tokens(self,
                            input_tokens_ids: Union[List[int], torch.Tensor],
                            input_embeddings: torch.Tensor):
        """
        Replace task special tokens' embedding with trainable parameters.
        """
        mask = isin(input_tokens_ids, self.task_special_tokens_ids.to(input_tokens_ids.device))
        embs = self.task_special_tokens_embeddings(input_tokens_ids[mask] -
                                                   min(self.task_special_tokens_ids.to(input_tokens_ids.device)))
        input_embeddings[mask] = embs

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                output_attentions: Optional[bool] = True,
                output_hidden_states: Optional[bool] = True,
                is_train: Optional[bool] = True) -> Any:
        """
        Passes input batch through:
        1) Sequence embedder model (transactions model);
        2) Connector
        3) Collate CLM input sequences and pass through LM decoder
        Args:
            batch: a prepared with chosen task batch of items;
            output_attentions: whether to output attention maps;
            output_hidden_states: whether to output LM hidden states;
            is_train: whether to pass to LM forward input labels or not;

        Returns:
            LM model's outputs with added labels (if `is_train` was set).
        """
        # 1) Get transactions embeddings for initial batch
        # return: Tuple[
        # torch.Tensor, - embeddings
        # torch.Tensor - mask
        # ]
        transaction_mask = batch['mask']
        batch_size = transaction_mask.size(0)
        device = transaction_mask.device

        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # 1.2) If created position embeddings, apply them first
        if hasattr(self, "temp_position_embedding"):
            batch_size, seq_len = transactions_embeddings.size()[:2]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            transactions_position_embeddings = self.temp_position_embedding(position_ids)
            transactions_embeddings = transactions_embeddings + transactions_position_embeddings

        # 2) Next pass them to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_embeddings = self.connector(transactions_embeddings,
                                                     mask=transactions_embeddings_mask)
        elif self.inspect_forward_signature("input_text_ids", self.connector) and \
             self.inspect_forward_signature("input_text_attention_mask", self.connector):
            # using Instruct QFormer model
            transactions_embeddings = self.connector(
                embeds=transactions_embeddings,
                # strip 2 tokens from the start of Q, as "[/trx]" + "."
                input_text_ids=batch['question_end_tokens'][:, 2:],
                input_text_attention_mask=batch['question_end_attention_mask'][:, 2:]
            )

        else:
            transactions_embeddings = self.connector(transactions_embeddings)

        # 3) Questions: to embedding of LM
        # torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)

        # if it already ends with [trx]
        question_start_attention_mask = batch['question_start_tokens_mask']

        # if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)

        question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # Replace task special tokens embeddings with trainable parameters
        if self.has_task_tokens(question_start_tokens_batch):
            self.replace_task_tokens(question_start_tokens_batch, question_start_embeddings_batch)

        # Question ends: to embedding of LM
        # 3.1) Strip paddings from questions endings!!!
        question_end_tokens_mask = batch['question_end_attention_mask'].bool()  # 1 - token, 0 == pad
        answer_tokens_mask = batch['answer_mask'].bool()

        question_end_tokens_full = []
        for i in range(question_end_tokens_mask.size(0)):
            # question without padding
            question_end_tokens_ = batch['question_end_tokens'][i][question_end_tokens_mask[i]]
            # answers without padding
            answer_ = batch['answer_tokens'][i][answer_tokens_mask[i]]
            full_question_end_tokens_ = torch.cat([question_end_tokens_,
                                                   # self.whitespace_token_id.to(device),
                                                   # check to not to insert <eos> before answer tokens!!!
                                                   answer_,
                                                   # add EOS token to train model to finish generation
                                                   self.eos_token_id.to(device)
                                                   ], dim=0)
            question_end_tokens_full.append(full_question_end_tokens_)

        # 3.2) Pad to max q+a length
        question_end_tokens_full_mask = []
        max_question_answer_len = max([len(qa) for qa in question_end_tokens_full])
        for i in range(question_end_tokens_mask.size(0)):
            n_padds = max_question_answer_len - question_end_tokens_full[i].size(0)

            question_end_mask = torch.full((max_question_answer_len,), 1).to(device)  # 1 for all tokens, [seq_len]
            question_end_mask[question_end_tokens_full[i].size(0):] = 0  # 0 for paddings
            question_end_tokens_full_mask.append(question_end_mask)

            question_end_tokens_full[i] = torch.cat(
                [
                    question_end_tokens_full[i],
                    torch.full((n_padds,), self.tokenizer.pad_token_id).to(device)
                ], dim=0)

        # 3.3) Cat back into batch
        question_end_tokens_full = torch.stack(question_end_tokens_full).long()
        question_end_tokens_full_mask = torch.stack(question_end_tokens_full_mask).long()

        # Get LLM embeddings
        question_end_embeddings_batch = self.language_model_tokens_embedding_func(question_end_tokens_full)

        # 3.4) Fill with trainable parameters
        # if it already starts with [/trx]
        if self.has_end_token(question_end_tokens_full):
            self.replace_end_token(question_end_tokens_full, question_end_embeddings_batch)

        # otherwise prepend it to the start of question sequence
        else:
            # todo: add one more sample to attention_mask also !!!
            question_end_embeddings_batch = torch.cat([
                self.transactions_end_embedding[None, None].repeat(batch_size, 1, 1),
                question_end_embeddings_batch], dim=1)

        # Get general LM's input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        input_embedds = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)
        input_mask = torch.cat([
            # Q start mask -> all 1
            torch.full(question_start_embeddings_batch.size()[:2], 1).long().to(device),
            # Transactions mask  -> all 1
            torch.full(transactions_embeddings.size()[:2], 1).long().to(device),
            # Q end + Answers mask (some are masked with 0)
            question_end_tokens_full_mask
        ], dim=1)

        # First transactions token
        transactions_start_i = question_start_embeddings_batch.size(1)
        # Last transactions token
        transactions_end_i = transactions_start_i + transactions_embeddings.size(1)

        # Create transactions tokens ids + mask padding in transactions history
        max_transactions_size = transactions_embeddings.size(1)
        transactions_text_tokens = [self._ret_tokens_template % i for i in range(max_transactions_size)]

        transactions_tokens = torch.Tensor([
            self.tokenizer.convert_tokens_to_ids(self._ret_tokens_template % i)
            for i in range(max_transactions_size)]).long().repeat(batch_size, 1).to(device)

        # Check if transactions history embedding size was reduced, then we cannot mask it
        if transactions_tokens.size(-1) == transactions_embeddings_mask.size(-1):
            transactions_tokens.masked_fill_(transactions_embeddings_mask == 0,
                                             self.tokenizer.pad_token_id)
        # 5) Labels
        #  Label = [-100 * (question_start_tokens_len - 1)
        #             <trns>,
        #             -100 * transactions_tokens_len
        #             </trns>,
        #             -100 * (question_end_tokens_len - 1)]

        question_end_labels = question_end_tokens_full.clone()
        for i in range(batch_size):
            answer_tokens_len = batch['answer_tokens'][i].size(0) + 1  # + 1 for whitespace token
            question_end_labels[i, 1:-answer_tokens_len] = -100

        text_labels = torch.cat([
            # -100 * (question_start_tokens_len - 1)
            torch.full((batch_size, batch['question_start_tokens'].size(1) - 1), -100).to(device),  # --> empty
            # <trns> token
            batch['question_start_tokens'][:, -1].repeat(batch_size, 1),
            # transactions_tokens
            # transactions_tokens.to(device),
            # -100 * transactions_tokens
            torch.full_like(transactions_tokens, -100).to(device),
            question_end_tokens_full[:, 0].unsqueeze(-1),  # </trns> to [batch_size, 1]
            # -100 * (question_end_tokens_len - answer_len) + answer tokens
            question_end_labels[:, 1:]
        ], dim=1)

        # print(f"Labels decoded:")
        # for labs in text_labels:
        #     print(f"{self.tokenizer.decode([t for t in labs if t != -100], skip_special_tokens=False)}")

        # 6) Pass through LM
        self._device_type = self.language_model.device.type  # 'cuda' or 'cpu'
        with torch.autocast(device_type=self._device_type):
            # contains: ['loss', 'logits', 'past_key_values', 'last_hidden_state']
            # `logits` of size: [batch_size, max_pred_len, vocab_size]
            lm_outputs = self.language_model(inputs_embeds=input_embedds,
                                             attention_mask=input_mask,
                                             labels=text_labels if is_train else None,
                                             output_attentions=output_attentions,
                                             output_hidden_states=output_hidden_states)

        # 7) Calculate retrival loss
        try:
            ret_loss_outputs = self._compute_retrieval_loss_fromage(lm_outputs,
                                                                    ret_start_i=transactions_start_i,
                                                                    ret_end_i=transactions_end_i,
                                                                    ret_embeddings=transactions_embeddings,
                                                                    output_hidden_states=True)
        except Exception as e:
            self._logger.error(f"Contrastive loss error: {e}")
            ret_loss_outputs = {'loss': torch.Tensor([0.0]).to(lm_outputs.loss.device)}

        # Re-scale losses
        total_loss = lm_outputs.loss * self._text_loss_scale + \
                     ret_loss_outputs.get('loss') * self._retrieval_loss_scale

        # join two output dicts
        outputs = dict()
        outputs["logits"] = lm_outputs.logits.contiguous().float()

        outputs["text_loss"] = lm_outputs.loss * self._text_loss_scale
        ret_loss = ret_loss_outputs.pop('loss')
        outputs["retrieval_loss"] = ret_loss * self._retrieval_loss_scale

        outputs["unscaled_text_loss"] = lm_outputs.loss
        outputs["unscaled_retrieval_loss"] = ret_loss

        if output_attentions:
            outputs["attentions"] = lm_outputs.attentions
        if output_hidden_states:
            outputs["hidden_states"] = lm_outputs.hidden_states
        outputs['loss'] = total_loss
        for key, val in ret_loss_outputs.items():
            outputs[key] = val

        if is_train:
            outputs['labels'] = text_labels
        return outputs

    def _compute_retrieval_loss(self,
                                outputs: Dict[str, torch.Tensor],
                                ret_start_i: int, ret_end_i: int,
                                ret_embeddings: torch.Tensor,
                                output_hidden_states: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive retrieval loss based on InfoNCE loss implementation.
        Args:
            outputs: a LLM outputs, containing: 'logits', 'hidden_states', etc.
            ret_start_i: a starting index of transactions embeddings injection;
            ret_end_i: an ending index of transactions embeddings injection (non-inclusive);
            ret_embeddings: a reference embeddings (i.e. target embeddings);
            output_hidden_states: whether to output hidden_states for retrieval tokens;
            output_logits: whether to output logits for retrieval tokens;

        Returns:
            a dict, containing 'loss' and,  optionally, 'last_hidden_state' and 'ret_tokens_logits'.
        """
        # 1) Extract hidden states and pass them through projection layers
        ret_hidden_states = []
        start_hidden_states = []
        end_hidden_states = []

        # As a projection_layers can be used: projection_layers or lm_connector
        for idx, projection_layer in zip(self._n_retrieval_layers, self.projection_layers):  # [lm_connector]
            ret_hidden_states.append(
                projection_layer(outputs.hidden_states[idx][..., ret_start_i:ret_end_i, :])
            )  # (bs, trns_history_seq_len, 768)

            start_hidden_states.append(
                outputs.hidden_states[idx][..., ret_start_i - 1, :]
            )  # (bs, 768)
            end_hidden_states.append(
                outputs.hidden_states[idx][..., ret_start_i, :]
            )  # (bs, 768)

        # start_tokens == 10
        # transactions_tokens == [50 ... 150]
        # Q end tokens == seq_len - 10 - transactions_tokens -> == question_end_tokens_full.size(-1)

        # 2) Add hidden states together
        collected_last_hidden_state = torch.stack(ret_hidden_states, dim=-1).sum(dim=-1)
        collected_start_hidden_states = torch.stack(start_hidden_states, dim=-1).sum(dim=-1)
        collected_end_hidden_states = torch.stack(end_hidden_states, dim=-1).sum(dim=-1)

        # 4) Calculate Contrastive loss
        loss_outputs = dict()
        try:
            ret_loss_accumulated = []
            for trx_i in range(collected_last_hidden_state.size(1)):
                ret_token_loss = self.ret_NCE_loss_fn(ret_embeddings[:, trx_i, :],
                                                      collected_last_hidden_state[:, trx_i, :])
                ret_loss_accumulated.append(ret_token_loss)

            ret_loss_accumulated = torch.stack(ret_loss_accumulated).sum()
            loss_outputs['loss'] = ret_loss_accumulated
        except Exception as e:
            self._logger.error(f"!!! Exceptiom occurred during retrieval loss calculation:\n{e}")
            loss_outputs['loss'] = torch.zeros((1,), dtype=torch.float32).to(collected_last_hidden_state.device)

        if output_hidden_states:
            loss_outputs['last_hidden_state'] = collected_last_hidden_state
        return loss_outputs

    def _compute_retrieval_loss_fromage(self,
                                        outputs: Dict[str, torch.Tensor],
                                        ret_start_i: int, ret_end_i: int,
                                        ret_embeddings: torch.Tensor,
                                        output_hidden_states: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive retrieval loss based on InfoNCE loss implementation.
        Args:
            outputs: a LLM out
        Calculate contrastive retrieval loss based on InfoNCE loss implementation.
        Args:
            outputs: a LLM outputs, containing: 'logits', 'hidden_states', etc.
            ret_start_i: a starting indexes of retrieval tokens sequence;
            ret_end_i: an ending indexes of retrieval tokens sequence (non-inclusive);
            ret_embeddings: a reference embeddings (i.e. target embeddings);
            output_hidden_states: whether to output hidden_states for retrieval tokens;
            output_logits: whether to output logits for retrieval tokens;

        Returns:
            a dict, containing 'loss' and,  optionally, 'last_hidden_state' and 'ret_tokens_logits'.
        puts, containing: 'logits', 'hidden_states', etc.

        Returns:
            a dict, containing 'loss' and,  optionally, 'last_hidden_state' and 'ret_tokens_logits'.
        """
        # 1) Extract hidden states and pass them through projection layers
        ret_hidden_states = []

        # As a projection_layers can be used: projection_layers or lm_connector
        for idx, projection_layer in zip(self._n_retrieval_layers, self.projection_layers):  # [lm_connector]
            hidd_state = outputs.hidden_states[idx]  # size: [bs, seq_len, hidd_size]
            batch_ret_hidd_states = []
            for ret_i in range(hidd_state.size(0)):
                batch_ret_hidd_states.append(hidd_state[ret_i,
                                             ret_start_i:ret_end_i,
                                             :].unsqueeze(0))
            batch_ret_hidd_states = torch.cat(batch_ret_hidd_states, dim=0)
            ret_hidden_states.append(
                projection_layer(maybe_autocast(batch_ret_hidd_states, projection_layer[0].weight.dtype))
            )  # (bs, trns_history_seq_len, 768)

        # 2) Add hidden states together
        collected_last_hidden_state = torch.stack(ret_hidden_states, dim=-1).sum(dim=-1)

        # 3) Normalize embeddings
        ret_embeddings_norm = (ret_embeddings / ret_embeddings.norm(dim=-1, keepdim=True))
        collected_last_hidden_state_norm = (
                collected_last_hidden_state / collected_last_hidden_state.norm(dim=-1, keepdim=True))

        # 4) cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        ret_embeddings_norm = logit_scale * ret_embeddings_norm

        logits_per_sample = ret_embeddings_norm @ collected_last_hidden_state.permute(0, 2, 1)
        logits_per_query = logits_per_sample.permute(0, 2, 1)

        targets = torch.linspace(0, ret_embeddings.size(1), ret_embeddings.size(1), dtype=int)
        targets = targets.unsqueeze(0).repeat(ret_embeddings.size(0), 1).to(
            ret_embeddings.device)  # as size: [bs, n_queries]

        # Contrastive loss: 32 queries vs. 32 queries
        # as mean of:
        #  1) similarities RET tokens last hidden states <-> queries
        #  2) similarities queries <-> RET tokens last hidden states
        loss_contrastive = (torch.nn.functional.cross_entropy(logits_per_sample, targets)  # label_smoothing=0.1
                            + torch.nn.functional.cross_entropy(logits_per_query, targets)  # label_smoothing=0.1
                            ) / 2

        loss_outputs = dict(loss=loss_contrastive)
        if output_hidden_states:
            loss_outputs['last_hidden_state'] = collected_last_hidden_state
        return loss_outputs

    def generate(self,
                 questions: Union[str, List[str], torch.Tensor],
                 transactions_batch: Dict[str, torch.Tensor],
                 prefix_prompt: Optional[Union[str, torch.Tensor]] = "",
                 answer_template: Optional[str] = "",
                 max_new_tokens: Optional[int] = None,
                 min_new_tokens: Optional[int] = None,
                 top_p: Optional[float] = 1.0,
                 temperature: Optional[float] = 0.0,
                 suggestions: Optional[int] = 1,
                 diversity_penalty: Optional[float] = 0.0,
                 filter_value: Optional[float] = -float('Inf'),
                 allowed_token_ids: Optional[List[int]] = None,
                 hidden_dims_indexes: Optional[List[int]] = None,
                 stopping_criteria: Optional[Any] = None,
                 use_custom_generation: Optional[bool] = False,
                 seed: Optional[int] = 11):
        """
        Generates answers for questions.
        Args:
            questions: Union[str, List[str], torch.Tensor] - a question(-s) to answer. Can be passed as:
                str: a single question in string representation;
                List[str]: a list of questions in string representation;
                torch.Tensor: a single tokenized question or multiple questions;
            transactions_batch: Dict[str, torch.Tensor] - a batch for transactions model;
            prefix_prompt: Union[str, torch.Tensor] - a prefix for transactions embeddings. Can be passed as:
                str: a prefix in string representation;
                torch.Tensor: a tokenized prefix;
            answer_template: str - an answer template prefix to add to the question ending;
            max_new_tokens: int - the maximum number of tokens to generate,
                                    ignoring the number of tokens in the question;
            min_new_tokens: int - the minimum number of tokens to generate,
                                    ignoring the number of tokens in the question;
            top_p: float - If set to float < 1, only the most probable tokens with probabilities
                            that add up to top_p or higher are kept for generation;
            temperature: float - The value used to module the next token probabilities;
            suggestions: TBD
            diversity_penalty: TBD
            filter_value: float - a value to assign to tokens that should never be generated;
            allowed_token_ids: List[int] - a list of token ids that must be generated;
            hidden_dims_indexes: List[int] - a list of hidden layers' indexes from
                                        which to take hidden states for embedding creation.
                                        Default set to -1 - so only last layer's hidden states would be used;
            stopping_criteria: a class instance / callable that can be used to change
                                when to stop generation (other than EOS token).
                                It should return a boolean flag when all batch samples are successfully finished;
            seed: int - a seed for generation;

        Returns:
            A dict with keys:
             - generated_texts - a list of generated text tokens sequences for each batch sample;
             - output_embeddings - a list of embeddings for sequences, collected from selected hidden layers' states;
             - output_logits - a list of logits generated for each batch sample on each step.
        """
        seed_everything(seed)
        self.eval()  # freeze all at once

        device = self.language_model.device
        if device.type != 'cpu':
            torch.cuda.empty_cache()

        # Transactions
        transactions_history_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(
            transactions_batch
        )
        vocab_size = self.language_model.vocab_size
        batch_size = transactions_history_embeddings.size(0)

        # Fill empty parameters
        hidden_dims_indexes = hidden_dims_indexes if hidden_dims_indexes is not None else [-1]

        # Encode everything
        # Starting prompts
        # In case single question in string form
        if isinstance(prefix_prompt, str):
            prefix_prompt_tokens = self.tokenizer.encode(prefix_prompt,
                                                         add_special_tokens=False,
                                                         return_tensors='pt').long().to(device)
        elif isinstance(prefix_prompt, torch.Tensor):
            prefix_prompt_tokens = prefix_prompt.long().to(device)
        else:
            raise AttributeError(f"Unable to use prefix prompt in provided form: {type(prefix_prompt)}!")

        prefix_prompt_embeddings = self.language_model_tokens_embedding_func(prefix_prompt_tokens)

        # if it already ends with [trx]
        if self.has_start_token(prefix_prompt_tokens):
            self.replace_start_token(prefix_prompt_tokens, prefix_prompt_embeddings)

        # Replace task special tokens embeddings with trainable parameters
        if self.has_task_tokens(prefix_prompt_tokens):
            self.replace_task_tokens(prefix_prompt_tokens, prefix_prompt_embeddings)

        prefix_prompt_embeddings_batch = prefix_prompt_embeddings.repeat(batch_size, 1, 1)
        prefix_prompt_mask_batch = torch.full(prefix_prompt_embeddings_batch.size()[:2], 1).long().to(device)

        # Question
        # In case single question in string form
        if isinstance(questions, str):
            question_tokens = self.tokenizer.encode(questions,
                                                    add_special_tokens=False,
                                                    return_tensors='pt').long().to(device)
            question_mask = torch.full(question_tokens.size(), 1).long().to(device)
            question_embeddings = self.language_model_tokens_embedding_func(question_tokens)
            question_embeddings_batch = question_embeddings.repeat(batch_size, 1, 1)

        # In case questions in tokenized form of tensors
        elif isinstance(questions, torch.Tensor):
            question_tokens = questions
            question_mask = torch.full(questions.size(), 1).long().to(device)
            question_embeddings = self.language_model_tokens_embedding_func(questions)
            if question_embeddings.size(0) != batch_size:
                # If it was a single question
                question_embeddings_batch = question_embeddings.repeat(batch_size, 1, 1)
            else:
                question_embeddings_batch = question_embeddings

        # In case a list of string questions provided
        elif isinstance(questions, list) and isinstance(questions[0], str):
            question_tokens = self.tokenizer.encode(questions,
                                                    padding=True,
                                                    add_special_tokens=False,
                                                    return_tensors='pt').long().to(device)
            question_mask = torch.full(question_tokens.size(), 1).long().to(device)
            question_embeddings = self.language_model_tokens_embedding_func(question_tokens)
            question_embeddings_batch = question_embeddings
        else:
            raise AttributeError(f"Unable to use questions in provided form: {type(questions)}!")

        # if it already starts with [/trx]
        if self.has_end_token(question_tokens):
            self.replace_end_token(question_tokens, question_embeddings)

        # 2) Pass transaction embeddings to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_history_embeddings = self.connector(transactions_history_embeddings,
                                                     mask=transactions_embeddings_mask)
        elif self.inspect_forward_signature("input_text_ids", self.connector) and \
                self.inspect_forward_signature("input_text_attention_mask", self.connector):
            # using Instruct QFormer model
            transactions_history_embeddings = self.connector(
                embeds=transactions_history_embeddings,
                # strip 2 tokens from the start of Q, as "[/trx]" + "."
                input_text_ids=question_tokens[:, 2:],
                input_text_attention_mask=question_mask[:, 2:]
            )

        else:
            transactions_history_embeddings = self.connector(transactions_history_embeddings)

        transactions_seq_len = transactions_history_embeddings.size(1)

        # Answer template --> embeddings
        answer_template_tokens = self.tokenizer.encode(answer_template,
                                                       add_special_tokens=False,
                                                       return_tensors='pt').long().to(device)
        # If empty template (to prevent errors in embeddings)
        if not answer_template_tokens.size(1):
            answer_template_tokens = self.whitespace_token_id.to(device)

        answer_template_embeddings = self.language_model_tokens_embedding_func(answer_template_tokens)
        answer_template_embeddings_batch = answer_template_embeddings.repeat(batch_size, 1, 1)

        # Concat all together
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        input_embedds = torch.cat([prefix_prompt_embeddings_batch,
                                   transactions_history_embeddings,
                                   question_embeddings_batch,
                                   answer_template_embeddings_batch], dim=1).to(device)
        input_mask = torch.cat([prefix_prompt_mask_batch,
                                torch.full(transactions_history_embeddings.size()[:2], 1).long().to(device),
                                question_mask,
                                torch.full(answer_template_embeddings_batch.size()[:2], 1).long().to(device)
                                ], dim=1).to(device)

        if (self.language_model.config.architectures[0] in USE_HF_GENERATE) and not use_custom_generation:
            return self._hf_generate(input_embedds=input_embedds,
                                     input_mask=input_mask,
                                     temperature=temperature,
                                     min_new_tokens=min_new_tokens,
                                     max_new_tokens=max_new_tokens,
                                     top_p=top_p,
                                     hidden_dims_indexes=hidden_dims_indexes,
                                     allowed_token_ids=allowed_token_ids,
                                     stopping_criteria=stopping_criteria,
                                     filter_value=filter_value,
                                     seed=seed,
                                     device=device)
        else:
            return self._custom_generate(input_embedds=input_embedds,
                                         temperature=temperature,
                                         min_new_tokens=min_new_tokens,
                                         max_new_tokens=max_new_tokens,
                                         top_p=top_p,
                                         suggestions=suggestions,
                                         diversity_penalty=diversity_penalty,
                                         hidden_dims_indexes=hidden_dims_indexes,
                                         allowed_token_ids=allowed_token_ids,
                                         stopping_criteria=stopping_criteria,
                                         filter_value=filter_value,
                                         seed=seed,
                                         device=device)

    def _hf_generate(self, input_embedds: torch.Tensor,
                     input_mask: torch.Tensor,
                     max_new_tokens: Optional[int] = None,
                     min_new_tokens: Optional[int] = None,
                     top_p: Optional[float] = 1.0,
                     temperature: Optional[float] = 0.0,
                     repetition_penalty: Optional[float] = 1.0,
                     length_penalty: Optional[float] = 1.0,
                     filter_value: Optional[float] = -float('Inf'),
                     allowed_token_ids: Optional[List[int]] = None,
                     hidden_dims_indexes: Optional[List[int]] = None,
                     stopping_criteria: Optional[Any] = None,
                     seed: Optional[int] = 11,
                     device: Union[torch.device, str] = "cpu"):
        """
        Generate with HF utilities.
        Args:
            input_embedds: input embeddings for LLM;
            input_mask: attention mask for input embeddings;
            max_new_tokens: int - the maximum number of tokens to generate,
                                    ignoring the number of tokens in the question;
            min_new_tokens: int - the minimum number of tokens to generate,
                                    ignoring the number of tokens in the question;
            top_p: float - If set to float < 1, only the most probable tokens with probabilities
                            that add up to top_p or higher are kept for generation;
            temperature: float - The value used to module the next token probabilities;
            repetition_penalty: float - a penalty for tokens sequences repetition;
                                The parameter for repetition penalty. 1.0 means no penalty.
            length_penalty: float -  Exponential penalty to the length that is used with beam-based generation.
                It is applied as an exponent to the sequence length, which in turn is used to divide the score
                of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
                length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
            filter_value: float - a value to assign to tokens that should never be generated;
            allowed_token_ids: List[int] - a list of token ids that must be generated;
            hidden_dims_indexes: List[int] - a list of hidden layers' indexes from
                                        which to take hidden states for embedding creation.
                                        Default set to -1 - so only last layer's hidden states would be used;
            stopping_criteria: a class instance / callable that can be used to change
                                when to stop generation (other than EOS token).
                                It should return a boolean flag when all batch samples are successfully finished;
            seed: int - a seed for generation;
            device: a device to generate on.

        Returns:
            torch.Tensor - generated token ids.
        """
        generation_config = transformers.GenerationConfig(max_new_tokens=max_new_tokens,
                                                          min_new_tokens=min_new_tokens,
                                                          do_sample=False,  # use greedy decoding
                                                          num_beams=1,
                                                          temperature=temperature,
                                                          top_p=top_p,
                                                          repetition_penalty=repetition_penalty,
                                                          length_penalty=length_penalty,
                                                          eos_token_id=self.tokenizer.eos_token_id,
                                                          bos_token_id=self.tokenizer.bos_token_id)
        with torch.no_grad():  # no tracking history
            generated_tokens_ids = self.language_model.generate(
                inputs_embeds=input_embedds,
                attention_mask=input_mask,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return generated_tokens_ids

    def _custom_generate(self, input_embedds: torch.Tensor,
                         max_new_tokens: Optional[int] = None,
                         min_new_tokens: Optional[int] = None,
                         top_p: Optional[float] = 1.0,
                         temperature: Optional[float] = 0.0,
                         suggestions: Optional[int] = 1,
                         diversity_penalty: Optional[float] = 0.0,
                         filter_value: Optional[float] = -float('Inf'),
                         allowed_token_ids: Optional[List[int]] = None,
                         hidden_dims_indexes: Optional[List[int]] = None,
                         stopping_criteria: Optional[Any] = None,
                         seed: Optional[int] = 11,
                         device: Union[torch.device, str] = "cpu"):
        vocab_size = self.language_model.vocab_size

        embeddings = input_embedds.clone().to(device)
        output_embeddings = []
        output_logits = []
        out = None

        with torch.no_grad():  # no tracking history
            # Generate max number of tokens if stopping criterion is not triggered
            i = 0
            for _ in range(max_new_tokens):
                i += 1
                output = self.language_model(inputs_embeds=embeddings, output_hidden_states=True)

                # Collect and sum the hidden states.
                hidden_states = []
                for idx in hidden_dims_indexes:
                    hidden_states.append(output.hidden_states[idx])
                # Add hidden states together.
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, T, 256)
                last_embedding = last_hidden_state / last_hidden_state.norm(dim=-1, keepdim=True)
                output_embeddings.append(last_embedding)

                logits = output.logits[:, -1, :]  # take only last token logits (N, vocab_size)

                # If we need to restrict model to predict only some tokens
                if allowed_token_ids is not None:
                    logits = logits[:, allowed_token_ids]

                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)

                past_key_values = output.past_key_values
                # get next token
                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / temperature

                    # Apply top-p filtering.
                    if top_p < 1.0:
                        assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1),
                                                        dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value

                    token_weights = logits.exp()  # (N, vocab_size)
                    next_token = torch.multinomial(token_weights, 1)  # (N, 1)

                # Concat with previous embeddings
                next_token = next_token.long().to(device)

                if allowed_token_ids is not None:
                    next_token = torch.tensor(allowed_token_ids).to(device)[next_token]
                out_of_vocab_indexes = torch.where(next_token > vocab_size)
                next_token[out_of_vocab_indexes] = self.tokenizer.eos_token_id \
                    if hasattr(self.tokenizer, "eos_token_id") else 0

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                self._logger.info(f"Output decoded: {[self.tokenizer.decode(token) for token in out]}")
                next_embedding = self.language_model_tokens_embedding_func(next_token)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)

                # If stopping criteria triggered for all samples in batch
                # and all samples in batch reached min number of new tokens
                if (stopping_criteria is not None) \
                        and stopping_criteria(out) \
                        and (i >= min_new_tokens):
                    self._logger.warning(f"Stopping criteria triggered!")
                    break

        return dict(generated_texts=out,
                    output_embeddings=output_embeddings,
                    output_logits=output_logits)
