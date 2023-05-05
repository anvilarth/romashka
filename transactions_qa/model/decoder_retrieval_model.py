from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.utils import seed_everything
from romashka.transactions_qa.model.decoder_model import DecoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask

from romashka.transactions_qa.losses.infonce_loss import InfoNCE
from romashka.transactions_qa.utils import (mask_padding, mask_lm_labels_padding)


class DecoderRetrievalModel(DecoderSimpleModel):
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
        device = transaction_mask.device

        batch_size = transaction_mask.size(0)
        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # 2) Next pass them to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_embeddings = self.connector(transactions_embeddings,
                                                     mask=transactions_embeddings_mask)
        else:
            transactions_embeddings = self.connector(transactions_embeddings)

        # 3) Questions: to embedding of LM
        # torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)

        # if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)
        # otherwise append it to the end of starting sequence
        else:
            # todo: add one more sample to attention_mask also !!!
            question_start_embeddings = torch.cat([question_start_embeddings,
                                                   self.transactions_start_embedding[None, None]], dim=1)

        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # Question ends: to embedding of LM
        # 3.1) Strip paddings from questions endings!!!
        question_end_tokens_mask = batch['question_end_attention_mask'].bool()  # 1 - token, 0 == pad

        question_end_tokens_full = []
        for i in range(question_end_tokens_mask.size(0)):
            # question without padding
            question_end_tokens_ = batch['question_end_tokens'][i][question_end_tokens_mask[i]]
            answer_ = batch['answer_tokens'][i]
            full_question_end_tokens_ = torch.cat([question_end_tokens_,
                                                   self.whitespace_token_id.to(device),
                                                   answer_,
                                                   # eos_token_id.to(device)
                                                   ], dim=0)
            question_end_tokens_full.append(full_question_end_tokens_)

        # 3.2) Pad to max q+a length
        max_question_answer_len = max([len(qa) for qa in question_end_tokens_full])
        for i in range(question_end_tokens_mask.size(0)):
            n_padds = max_question_answer_len - question_end_tokens_full[i].size(0)
            question_end_tokens_full[i] = torch.cat(
                [torch.full((n_padds,), self.tokenizer.pad_token_id).to(device),
                 question_end_tokens_full[i],
                 ], dim=0)

        # 3.3) Cat back into batch
        question_end_tokens_full = torch.stack(question_end_tokens_full).long()

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

        transactions_tokens.masked_fill_(transactions_embeddings_mask == 0,
                                         self.tokenizer.pad_token_id)
        # 5) Labels
        #  5.3) Label = [-100 * (question_start_tokens_len - 1)
        #             <trns>,
        #             -100 * transactions_tokens_len
        #             </trns>,
        #             -100 * (question_end_tokens_len - 1)]

        #  5.4) Label = [<pad> * (question_start_tokens_len - 1)
        #             <trns>,
        #             transactions_tokens, <pad> * num_trns_history_paddings,
        #             </trns>,
        #             <pad> * (question_end_tokens_len - 1)]

        question_end_labels = question_end_tokens_full.clone()
        for i in range(batch_size):
            answer_tokens_len = batch['answer_tokens'][i].size(0)  # + 1  # + 1 for whitespace token
            question_end_labels[i, 1:-answer_tokens_len] = -100

        text_labels = torch.cat([
            # <pad> * (question_start_tokens_len - 1)
            torch.full((batch_size, batch['question_start_tokens'].size(1) - 1), self.tokenizer.pad_token_id).to(
                device),  # --> empty
            # <trns> token
            batch['question_start_tokens'][:, -1].repeat(batch_size, 1),
            # transactions_tokens, < pad > * num_trns_history_paddings,
            transactions_tokens.to(device),
            question_end_tokens_full[:, 0].unsqueeze(-1),  # </trns> to [batch_size, 1]
            # <pad> * (question_end_tokens_len - 1)
            question_end_labels[:, 1:]
        ], dim=1)

        labels_masked = mask_lm_labels_padding(text_labels,
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               mask_value=-100)

        # 6) Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        lm_outputs = self.language_model(inputs_embeds=input_embedds,
                                         labels=labels_masked if is_train else None,
                                         output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states)

        # 7) Calculate retrival loss
        ret_loss_outputs = self._compute_retrieval_loss(lm_outputs,
                                                        ret_start_i=transactions_start_i,
                                                        ret_end_i=transactions_end_i,
                                                        ret_embeddings=transactions_embeddings,
                                                        output_hidden_states=True,
                                                        output_logits=True)

        # Re-scale losses
        total_loss = lm_outputs.loss * self._text_loss_scale + \
                     ret_loss_outputs.get('loss') * self._retrieval_loss_scale

        # join two output dicts
        outputs = dict()
        outputs["logits"] = lm_outputs.logits
        outputs["text_loss"] = lm_outputs.loss
        outputs["retrieval_loss"] = ret_loss_outputs.pop('loss')
        if output_attentions:
            outputs["attentions"] = lm_outputs.attentions
        if output_hidden_states:
            outputs["hidden_states"] = lm_outputs.hidden_states
        outputs['loss'] = total_loss
        for key, val in ret_loss_outputs.items():
            outputs[key] = val

        if is_train:
            outputs['labels'] = labels_masked
        if self._is_debug:
            outputs['input_embeddings'] = input_embedds  # for debug purposes
            question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
            outputs['question_encoded'] = torch.cat([question_start_tokens_batch,
                                                     batch['question_end_tokens']], dim=1)
            # Experimental !
            transactions_history_lengths = transaction_mask.sum(1)
            outputs['transactions_history_lengths'] = transactions_history_lengths

            outputs['question_start_input_size'] = question_start_embeddings_batch.size(1)
            outputs['question_end_input_size'] = question_end_embeddings_batch.size(1)
            outputs['transactions_input_size'] = transactions_embeddings.size(1)
            outputs['total_input_size'] = input_embedds.size(1)
        return outputs

    def _compute_retrieval_loss(self,
                                outputs: Dict[str, torch.Tensor],
                                ret_start_i: int, ret_end_i: int,
                                ret_embeddings: torch.Tensor,
                                output_hidden_states: Optional[bool] = False,
                                output_logits: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive retrival loss based on InfoNCE loss implementation.
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

        # 3) Collect also logits for start/end and retrieval tokens
        trx_start_token_logits = outputs.logits[:, ret_start_i - 1, :].to(torch.float32)
        trx_end_token_logits = outputs.logits[:, ret_end_i, :].to(torch.float32)
        ret_tokens_logits = outputs.logits[:, ret_start_i:ret_end_i, :].to(torch.float32)

        # 4) Calculate Contrastive loss
        ret_loss_accumulated = []
        for trx_i in range(collected_last_hidden_state.size(1)):
            ret_token_loss = self.ret_NCE_loss_fn(ret_embeddings[:, trx_i, :],
                                                  collected_last_hidden_state[:, trx_i, :])
            ret_loss_accumulated.append(ret_token_loss)

        ret_loss_accumulated = torch.stack(ret_loss_accumulated).sum()

        loss_outputs = dict(loss=ret_loss_accumulated)
        if output_hidden_states:
            loss_outputs['last_hidden_state'] = collected_last_hidden_state
        if output_logits:
            loss_outputs['ret_tokens_logits'] = ret_tokens_logits
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

        # 2) Next pass them to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_history_embeddings = self.connector(transactions_history_embeddings,
                                                     mask=transactions_embeddings_mask)
        else:
            transactions_history_embeddings = self.connector(transactions_history_embeddings)

        vocab_size = self.language_model.vocab_size
        batch_size = transactions_history_embeddings.size(0)
        transactions_seq_len = transactions_history_embeddings.size(1)

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
        prefix_prompt_embeddings_batch = prefix_prompt_embeddings.repeat(batch_size, 1, 1)

        # Question
        # In case single question in string form
        if isinstance(questions, str):
            question_tokens = self.tokenizer.encode(questions,
                                                    add_special_tokens=False,
                                                    return_tensors='pt').long().to(device)
            question_embeddings = self.language_model_tokens_embedding_func(question_tokens)
            question_embeddings_batch = question_embeddings.repeat(batch_size, 1, 1)

        # In case questions in tokenized form of tensors
        elif isinstance(questions, torch.Tensor):
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
            question_embeddings = self.language_model_tokens_embedding_func(question_tokens)
            question_embeddings_batch = question_embeddings
        else:
            raise AttributeError(f"Unable to use questions in provided form: {type(questions)}!")

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

        # if any([model_type in self.language_model.config.architectures[0] for model_type in USE_HF_GENERATE]):
        #     pass
        # else:
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