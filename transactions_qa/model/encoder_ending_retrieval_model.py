from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.utils import mask_padding
from romashka.transactions_qa.model.generation_utils import isin
from romashka.transactions_qa.model.encoder_model import EncoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.losses.infonce_loss import InfoNCE


class EncoderEndingRetrievalModel(EncoderSimpleModel):
    """
    Encoder-decoder model with multiple retrieval token appended to the very end of the sequence (before answers).
    """
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
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
                 transactions_embeddings_start_token: Optional[str] = r"[trx]",
                 transactions_embeddings_end_token: Optional[str] = r"[/trx]",
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):

        self._transactions_embeddings_start_token = transactions_embeddings_start_token
        self._transactions_embeddings_end_token = transactions_embeddings_end_token

        self.min_ret_tokens = min_ret_tokens
        self.max_ret_tokens = max_ret_tokens
        self._ret_tokens_template = "[RET_%s]"

        # self.projection_layers = nn.ModuleList([])
        self._n_retrieval_layers = n_retrieval_layers
        self._embeddings_dropout_p = embeddings_dropout_p

        self._retrieval_loss_scale = retrieval_loss_scale
        self._text_loss_scale = text_loss_scale

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
        self.register_buffer("eos_token_id",
                             torch.Tensor([self.tokenizer.eos_token_id]).long())

        self._create_trainable_parameters()
        self._create_losses()

        # Additionally re-assign embeddings
        self._set_language_model_embedding_func()

        # Check if language model is frozen, optionally freeze
        self._logger.info(f"Check language model's parameters to be frozen...")
        for param_name, param in self.language_model.named_parameters():
            if param.requires_grad:
                self._logger.warning(f"Parameter `{param_name}` of LM requires grad, freezing..")
                param.requires_grad = False

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
        # Create transactions embeddings start/end tokens: [trx] / [/trx] and trainable parameters for them
        self._create_surrounding_parameters()
        # Create retrieval tokens: RET_0 ... RET_N in tokenizers vocabulary and mappings token <-> id
        self._create_retrieval_parameters()

        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

        # Create projection layers from LM output hidden states to shared dim for loss calculation
        self._create_projection_layers()

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
            # model=self.language_model,  # -> optionally
            return_ids=True
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
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)), requires_grad=True)
        self.transactions_end_embedding = nn.Parameter(
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)), requires_grad=True)
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

        self.ret_tokens = [self._ret_tokens_template % str(i) for i in range(self.max_ret_tokens)]

        # Check if transactions retrieval tokens, exists in tokenizers' vocabulary,
        # add them if not exist and get their indexes
        self.transactions_ret_tokens2ids_mapping = AbstractTask.extend_vocabulary(
            new_tokens=self.ret_tokens,
            tokenizer=self.tokenizer,
            # model=self.language_model,  # -> optionally
            return_ids=True
        )
        self._logger.info(f"Retrieval tokens added to tokenizer: {len(self.ret_tokens)}\ntokens: {self.ret_tokens}.")

        # Init transactions retrieval tokens ids -> token names mapping
        self.transactions_ret_ids2tokens_mapping = {token_id: token for token, token_id
                                                    in self.transactions_ret_tokens2ids_mapping.items()}
        # Needed to select embedding by it's index from range [0, max_ret_tokens],
        # which doesn't equal to LLM tokenizer's vocabulary ids of RET tokens
        # self.ret_tokens_ids = list(self.transactions_ret_ids2tokens_mapping.keys())
        self.register_buffer("ret_tokens_ids",
                             torch.LongTensor(list(self.transactions_ret_ids2tokens_mapping.keys())).long())
        self.register_buffer("transactions_ret_start_id", min(self.ret_tokens_ids).long())

        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # RET embeddings 0...n_ret
        self.ret_embeddings = torch.nn.Embedding(num_embeddings=self.max_ret_tokens,
                                                 embedding_dim=params_dim)

    def _create_projection_layers(self):
        """
        Creates a linear mappings from language model hidden dimensionality
        to shared embeddings dimensionality for rET tokens loss calculation.
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
                text_fc = [nn.Linear(in_dim, shared_dim),
                           nn.Dropout(self._embeddings_dropout_p)]
                self.projection_layers.append(nn.Sequential(*text_fc))
            # Take representation from any middle layer
            elif layer_idx < self.language_model.config.num_hidden_layers:
                text_fc = [nn.Linear(self.language_model.config.hidden_size, shared_dim),
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
        input_embeddings[mask] = self.transactions_start_embedding

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
        input_embeddings[mask] = self.transactions_end_embedding

    def has_eos_token(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether EOS token id already contained in given ids.
        """
        return (input_tokens_ids == self.tokenizer.eos_token_id).sum() > 0

    def exclude_eos_token(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Cleans EOS tokens from given ids.
        """
        if self.has_eos_token(input_tokens_ids):
            mask = ~torch.eq(input_tokens_ids, self.tokenizer.eos_token_id)
            input_tokens_ids = input_tokens_ids[mask]
        return input_tokens_ids

    def has_ret_tokens(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether transactions retrieval token id already contained in given ids.
        """
        return isin(input_tokens_ids, self.ret_tokens_ids).sum() > 0

    def replace_ret_tokens(self, input_tokens_ids: Union[List[int], torch.Tensor],
                           input_embeddings: torch.Tensor):
        """
        Replace retrieval tokens' embedding with trainable parameters.
        """
        mask = isin(input_tokens_ids, self.ret_tokens_ids)
        embs = self.ret_embeddings(self.ret_tokens_ids - self.transactions_ret_start_id)
        embs = embs.repeat(input_tokens_ids.size(0), 1)
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
        # transactions model requires: ['mask', 'cat_features', 'num_features', 'meta_features']
        # return: Tuple[
        # torch.Tensor, - embeddings
        # torch.Tensor - mask
        # ]
        transaction_mask = batch['mask']
        batch_size = transaction_mask.size(0)
        device = transaction_mask.device

        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # 2) Next pass them to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_embeddings = self.connector(transactions_embeddings,
                                                     mask=transactions_embeddings_mask)
        else:
            transactions_embeddings = self.connector(transactions_embeddings)

        # 3) Questions: to embedding of LM - torch.Size([1, len(question_start_tokens))
        # torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_start_attention_mask = batch['question_start_tokens_mask']

        # if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)

        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # 4) Question ends: to embedding of LM
        # 4.1) Strip paddings from questions endings!!!
        question_end_tokens_mask = batch['question_end_attention_mask'].bool()  # 1 - token, 0 == pad
        question_end_tokens_full = []
        for i in range(question_end_tokens_mask.size(0)):
            # question without padding
            question_end_tokens_ = batch['question_end_tokens'][i][question_end_tokens_mask[i]]
            # clear EOS from the end of sequence
            question_end_tokens_ = self.exclude_eos_token(question_end_tokens_)
            full_question_end_tokens_ = torch.cat([question_end_tokens_,
                                                   self.whitespace_token_id,
                                                   # check to not to insert <eos> before answer tokens!!!
                                                   # insert RET_0 .. RET_N tokens here
                                                   self.ret_tokens_ids,  # .to(device)
                                                   # finish with EOS token
                                                   self.eos_token_id,
                                                   ], dim=0)
            question_end_tokens_full.append(full_question_end_tokens_)

        # 4.2) Pad to max q+a length
        max_question_answer_len = max([len(qa) for qa in question_end_tokens_full])
        for i in range(question_end_tokens_mask.size(0)):
            n_padds = max_question_answer_len - question_end_tokens_full[i].size(0)
            # Pad from the side which is required by tokenizer
            if self.tokenizer.padding_side == 'right':
                question_end_tokens_full[i] = torch.cat(
                    [question_end_tokens_full[i],
                     torch.full((n_padds,), self.tokenizer.pad_token_id).to(device),
                     ], dim=0)
            else:
                question_end_tokens_full[i] = torch.cat(
                    [torch.full((n_padds,), self.tokenizer.pad_token_id).to(device),
                     question_end_tokens_full[i],
                     ], dim=0)

        # 4.3) Cat back into batch
        question_end_tokens_full = torch.stack(question_end_tokens_full).long()

        # Get LLM embeddings
        question_end_embeddings_batch = self.language_model_tokens_embedding_func(question_end_tokens_full)

        # 5) Create new attention mask
        question_end_attention_mask = (~mask_padding(question_end_tokens_full)).long()

        # 6) Fill with trainable parameters
        # 6.1) Fill injection ending tokens embeddings with trainable parameters
        # if it already starts with [/trx]
        if self.has_end_token(question_end_tokens_full):
            self.replace_end_token(question_end_tokens_full, question_end_embeddings_batch)

        # 6.2) Fill injection ending tokens embeddings with trainable parameters
        if self.has_ret_tokens(question_end_tokens_full):
            self.replace_ret_tokens(question_end_tokens_full,
                                    question_end_embeddings_batch)

        # 7) Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)
        if ('encoder_input_mask' in batch) \
                and (batch['encoder_input_mask'].size(1) == encoder_input.size(1)):
            encoder_input_mask = batch['encoder_input_mask']

        else:
            # Check if transactions history embedding size was reduced, then we cannot mask it
            if transactions_embeddings.size(1) == transactions_embeddings_mask.size(-1):
                encoder_input_mask = torch.cat(
                    [question_start_attention_mask,
                     batch['mask'],
                     question_end_attention_mask], dim=1
                )
            else:
                encoder_input_mask = torch.cat(
                    [question_start_attention_mask,
                     torch.ones(transactions_embeddings.size()[:2], dtype=batch['mask'].dtype, device=device),
                     question_end_attention_mask], dim=1
                )

        # First transactions token
        transactions_start_i = question_start_embeddings_batch.size(1)
        # Last transactions token
        transactions_end_i = transactions_start_i + transactions_embeddings.size(1)

        # First retrieval token
        ret_start_indexes = torch.nonzero(question_end_tokens_full == self.ret_tokens_ids[0], as_tuple=True)[-1]
        # Last retrieval token
        ret_end_indexes = torch.nonzero(question_end_tokens_full == self.ret_tokens_ids[-1], as_tuple=True)[-1]

        # 5) Labels
        # Create answers + masks for LM's decoder inputs
        batch_answers = batch['answer_tokens']
        batch_answers_mask = batch['answer_mask']

        # Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        lm_outputs = self.language_model(inputs_embeds=encoder_input,
                                         attention_mask=encoder_input_mask,
                                         labels=batch_answers,
                                         output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states,
                                         decoder_attention_mask=batch_answers_mask)
        # Create answers + masks for LM's decoder inputs
        lm_outputs['answer_tokens'] = batch_answers

        # 7) Calculate retrival loss
        ret_loss_outputs = self._compute_retrieval_loss(lm_outputs,
                                                        ret_start_i=ret_start_indexes,
                                                        ret_end_i=ret_end_indexes,
                                                        ret_embeddings=transactions_embeddings,
                                                        output_hidden_states=True)

        # Re-scale losses
        total_loss = lm_outputs.loss * self._text_loss_scale + \
                     ret_loss_outputs.get('loss') * self._retrieval_loss_scale

        # join two output dicts
        outputs = dict()
        outputs["logits"] = lm_outputs.logits

        outputs["text_loss"] = lm_outputs.loss * self._text_loss_scale
        ret_loss = ret_loss_outputs.pop('loss')
        outputs["retrieval_loss"] = ret_loss * self._retrieval_loss_scale

        outputs["unscaled_text_loss"] = lm_outputs.loss
        outputs["unscaled_retrieval_loss"] = ret_loss

        if output_attentions:
            outputs["attentions"] = lm_outputs.attentions
        if output_hidden_states:
            outputs["hidden_states"] = lm_outputs.encoder_hidden_states
        outputs['loss'] = total_loss
        for key, val in ret_loss_outputs.items():
            outputs[key] = val

        if is_train:
            outputs['labels'] = batch_answers
        if self._is_debug:
            outputs['input_embeddings'] = encoder_input  # for debug purposes
            question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
            outputs['question_encoded'] = torch.cat([question_start_tokens_batch,
                                                     batch['question_end_tokens']], dim=1)
            # Experimental !
            transactions_history_lengths = transaction_mask.sum(1)
            outputs['transactions_history_lengths'] = transactions_history_lengths

            outputs['question_start_input_size'] = question_start_embeddings_batch.size(1)
            outputs['question_end_input_size'] = question_end_embeddings_batch.size(1)
            outputs['transactions_input_size'] = transactions_embeddings.size(1)
            outputs['total_input_size'] = encoder_input.size(1)
        return outputs

    def _compute_retrieval_loss(self,
                                outputs: Dict[str, torch.Tensor],
                                ret_start_i: Union[List[int], torch.Tensor],
                                ret_end_i: Union[List[int], torch.Tensor],
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

        # -> will take only last and pass it to single (for a while) projection layer
        last_hidden_state = outputs['encoder_hidden_states'][-1]

        ret_hidden_states = []
        for i in range(last_hidden_state.size(0)):
            # -> Currently use only one projection layer == for single hidden layer
            ret_hidden_states.append(
                self.projection_layers[0](last_hidden_state[i, ret_start_i[i]:(ret_end_i[i] + 1), :]).unsqueeze(0)
            ) # (bs, trns_history_seq_len, 768)

        # 2) Add hidden states together
        collected_last_hidden_state = torch.stack(ret_hidden_states, dim=0).squeeze()

        # 4) Calculate Contrastive loss
        ret_loss_accumulated = []
        for trx_i in range(collected_last_hidden_state.size(1)):
            ret_token_loss = self.ret_NCE_loss_fn(ret_embeddings[:, trx_i, :],
                                                  collected_last_hidden_state[:, trx_i, :])
            ret_loss_accumulated.append(ret_token_loss)

        # Use reduce with `mean` instead of `sum` (in previous Retrival model)
        ret_loss_accumulated = torch.stack(ret_loss_accumulated).mean()

        loss_outputs = dict(loss=ret_loss_accumulated)
        if output_hidden_states:
            loss_outputs['last_hidden_state'] = collected_last_hidden_state
        return loss_outputs