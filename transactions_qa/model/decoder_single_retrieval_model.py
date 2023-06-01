from typing import (List, Optional, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.model.decoder_model import DecoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.utils import (mask_padding, mask_lm_labels_padding)


class DecoderSingleRetrievalModel(DecoderSimpleModel):
    """
    Decoder-only model with single retrieval token appended to the very end of the sequence (before answers).
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
                 retrieval_loss_temperature: Optional[float] = 1.0,
                 transactions_embeddings_start_token: Optional[str] = r"[trx]",
                 transactions_embeddings_end_token: Optional[str] = r"[/trx]",
                 generation_config: Optional[Dict[str, Any]] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu',
                 is_debug: Optional[bool] = False):

        self.device = device

        self._transactions_embeddings_start_token = transactions_embeddings_start_token
        self._transactions_embeddings_end_token = transactions_embeddings_end_token

        self.min_embeddings_injection_tokens = min_ret_tokens
        self.max_embeddings_injection_tokens = max_ret_tokens
        self.ret_token = "[RET]"

        self._n_retrieval_layers = n_retrieval_layers
        self._embeddings_dropout_p = embeddings_dropout_p
        self.ret_temperature = retrieval_loss_temperature

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
            - retrieval token (single here!): [RET];
        """
        # Create transactions embeddings start/end tokens: [trx] / [/trx] and trainable parameters for them
        self._create_surrounding_parameters()
        # Create retrieval token: [RET] in tokenizers vocabulary and mappings token <-> id
        self._create_retrieval_parameters()

        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

        # Create projection layers from LM output hidden states to shared dim for loss calculation
        self._create_projection_layers()

    def _create_surrounding_parameters(self):
        """
        Creates trainable parameters for:
            - transactions embeddings start/end tokens: [trx] / [/trx];
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
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)),
            requires_grad=True)
        self.transactions_end_embedding = nn.Parameter(
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)),
            requires_grad=True)
        self._logger.info(f"Initialized trainable parameters for transactions embeddings start/end tokens.")

    def _create_retrieval_parameters(self):
        """
        Creates trainable parameters for:
            - retrieval token: [RET];
        """
        # Check if transactions retrieval tokens, exists in tokenizers' vocabulary,
        # add them if not exist and get their indexes
        self.transactions_ret_token2id_mapping = AbstractTask.extend_vocabulary(
            new_tokens=[self.ret_token],
            tokenizer=self.tokenizer,
            return_ids=True
        )
        self._logger.info(f"Retrieval tokens added to tokenizer: {len(self.ret_token)}\ntoken: {self.ret_token}.")

        # Init transactions retrieval token id
        self.ret_token_id = self.transactions_ret_token2id_mapping.get(self.ret_token)
        self.register_buffer("ret_token_id_tensor", torch.Tensor([self.ret_token_id]).long())

        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # Create trainable embedding for RET token
        self.ret_embedding = nn.Parameter(
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)),
            requires_grad=True)

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
        self.ret_loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

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

    def has_ret_token(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether transactions retrieval token id already contained in given ids.
        """
        return (input_tokens_ids == self.ret_token_id).sum() > 0

    def has_eos_token(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether EOS token id already contained in given ids.
        """
        return (input_tokens_ids == self.tokenizer.eos_token_id).sum() > 0

    def replace_ret_token(self, input_tokens_ids: Union[List[int], torch.Tensor],
                          input_embeddings: torch.Tensor):
        """
        Replace retrieval embeddings with trainable parameters.
        """
        mask = input_tokens_ids == self.ret_token_id
        input_embeddings[mask] = self.ret_embedding

    def update_trainable_embeddings(self,
                                    start_token_embedding: torch.Tensor,
                                    end_token_embedding: torch.Tensor,
                                    ret_token_embedding: torch.Tensor):
        """
        If LM model embeddings are frozen, then implicitly update some vectors in embeddings weights.
        Args:
            start_token_embedding: an updated start sequence embeddings injection token's embeddings vector;
            end_token_embedding: an updated end sequence embeddings injection token's embeddings vector;
            ret_token_embedding: an updated [RET] embeddings token's embeddings vector;
        """
        # Update embeddings
        with torch.no_grad():
            # for T5-like
            self.language_model.encoder.embed_tokens.weight[self.transactions_start_token_id] = start_token_embedding
            self.language_model.encoder.embed_tokens.weight[self.transactions_end_token_id] = end_token_embedding
            self.language_model.encoder.embed_tokens.weight[self.ret_token_id] = ret_token_embedding
            self._logger.debug(f"Trainable embeddings vectors updated")

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

        # 3) Update trainable embeddings in LM Embedding layer
        # self.update_trainable_embeddings(start_token_embedding=self.transactions_start_embedding,
        #                                  end_token_embedding=self.transactions_end_embedding,
        #                                  ret_token_embedding=self.ret_embedding)

        # 4) Questions: to embedding of LM - torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_start_attention_mask = batch['question_start_tokens_mask']
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # Insert trainable parameter if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)

        # 5) Question ends: to embedding of LM
        # 5.1) Strip paddings from questions endings!!!
        question_end_tokens_mask = batch['question_end_attention_mask'].bool()  # 1 - token, 0 == pad

        question_end_tokens_full = []
        for i in range(question_end_tokens_mask.size(0)):
            # question without padding
            question_end_tokens_ = batch['question_end_tokens'][i][question_end_tokens_mask[i]]
            answer_ = batch['answer_tokens'][i]
            full_question_end_tokens_ = torch.cat([question_end_tokens_,
                                                   self.whitespace_token_id.to(device),
                                                   # check to not to insert <eos> before answer tokens!!!
                                                   # insert RET token here
                                                   self.ret_token_id_tensor.to(device),
                                                   self.whitespace_token_id.to(device),
                                                   answer_], dim=0)
            question_end_tokens_full.append(full_question_end_tokens_)

        # 5.2) Pad to max q+a length
        max_question_answer_len = max([len(qa) for qa in question_end_tokens_full])
        for i in range(question_end_tokens_mask.size(0)):
            n_padds = max_question_answer_len - question_end_tokens_full[i].size(0)
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

        # 5.3) Concatenate back into batch
        question_end_tokens_full = torch.stack(question_end_tokens_full).long()

        # 5.4) Create new attention mask
        question_end_attention_mask = (~mask_padding(question_end_tokens_full)).long()

        # Question ends: to embedding of LM
        # question_end_tokens: torch.Size([batch_size, len(max_question_end_tokens)])
        question_end_embeddings_batch = self.language_model_tokens_embedding_func(question_end_tokens_full)

        # Fill injection ending tokens embeddings with trainable parameters
        if self.has_end_token(question_end_tokens_full):
            self.replace_end_token(question_end_tokens_full, question_end_embeddings_batch)

        # Fill injection ending tokens embeddings with trainable parameters
        if self.has_ret_token(question_end_tokens_full):
            self.replace_ret_token(question_end_tokens_full, question_end_embeddings_batch)

        # 6) LM input
        # 6.1) Get general LM's encoder input as:
        # Q_start_tokens + [trx] + TRNS_embeddings + [/trx] + Q_end_tokens + [RET] + " " + answer_tokens
        input_embedds = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)

        # 6.2) Update attention mask
        if ('encoder_input_mask' in batch) \
                and (batch['encoder_input_mask'].size(1) == input_embedds.size(1)):
          input_attention_mask = batch['encoder_input_mask']

        else:
            # Check if transactions history embedding size was reduced, then we cannot mask it
            if transactions_embeddings.size(1) == transactions_embeddings_mask.size(-1):
                input_attention_mask = torch.cat(
                    [question_start_attention_mask,
                     batch['mask'],
                     question_end_attention_mask], dim=1
                )
            else:
                input_attention_mask = torch.cat(
                    [question_start_attention_mask,
                     torch.ones(transactions_embeddings.size()[:2], dtype=batch['mask'].dtype, device=device),
                     question_end_attention_mask], dim=1
                )

        # 7) Labels
        # 7.1) Create answers vectors
        #  a) Label = [-100 * (question_start_tokens_len - 1)
        #             <trns>,  -> train!
        #             -100 * transactions_tokens_len
        #             </trns>,  -> train!
        #             -100 * (question_end_tokens_len - 1),
        #             answer_tokens  -> train!
        # ]

        #  b) Label = [<pad> * (question_start_tokens_len - 1)
        #             <trns>,  -> train!
        #             transactions_tokens, <pad> * num_trns_history_paddings,
        #             </trns>,  -> train!
        #             <pad> * (question_end_tokens_len - 1)
        #             answer_tokens  -> train!
        # ]

        #  c) Label = [<pad> * (question_start_tokens_len - 1)
        #             <trns>,  -> train!
        #             transactions_tokens, <pad> * num_trns_history_paddings,
        #             </trns>,  -> train!
        #             <pad> * (question_end_tokens_len - 1),
        #             [RET], whitespace token,  -> train!
        #             answer_tokens  -> train!
        # ]

        # Here use c)
        question_end_labels = question_end_tokens_full.clone()
        for i in range(batch_size):
            answer_tokens_len = batch['answer_tokens'][i].size(0) + 2  # + 1 for whitespace token
            question_end_labels[i, 1:-answer_tokens_len] = -100

        # 7.2) Concatenate all parts together
        text_labels = torch.cat([
            # <pad> * (question_start_tokens_len - 1)
            torch.full((batch_size, batch['question_start_tokens'].size(1) - 1), self.tokenizer.pad_token_id).to(
                device),  # --> empty
            # <trns> token
            batch['question_start_tokens'][:, -1].repeat(batch_size, 1),
            # transactions_tokens, < pad > * num_trns_history_paddings,
            torch.full(transactions_embeddings.size()[:2], self.tokenizer.pad_token_id).to(device),  # --> empty
            question_end_tokens_full[:, 0].unsqueeze(-1),  # </trns> to [batch_size, 1]
            # <pad> * (question_end_tokens) + [RET] token + whitespace_token + answer tokens.
            question_end_labels[:, 1:]
        ], dim=1)

        # 7.3) Exclude selected positions from CE loss calculation
        labels_masked = mask_lm_labels_padding(text_labels,
                                               pad_token_id=self.tokenizer.pad_token_id,
                                               mask_value=-100)

        # 8) Forward without labels
        lm_outputs = self.language_model(
            inputs_embeds=input_embedds,
            labels=labels_masked if is_train else None,
            attention_mask=input_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True)

        if is_train:
            # Create answers + masks for LM's decoder inputs
            lm_outputs['answer_tokens'] = labels_masked

            # 9) Calculate retrieval loss
            ret_loss_outputs = self._compute_retrieval_loss(lm_outputs,
                                                            question_end_tokens=question_end_tokens_full,
                                                            input_embedds=input_embedds,
                                                            injected_embeddings=transactions_embeddings,
                                                            output_hidden_states=True)
            # Re-scale losses
            total_loss = lm_outputs.loss * self._text_loss_scale
            total_loss += ret_loss_outputs.get('loss') * self._retrieval_loss_scale

        # join two output dicts
        outputs = dict()
        outputs["logits"] = lm_outputs.logits

        if is_train:
            outputs["text_loss"] = lm_outputs.loss * self._text_loss_scale
            outputs["unscaled_text_loss"] = lm_outputs.loss

            ret_loss = ret_loss_outputs.pop('loss')
            outputs["retrieval_loss"] = ret_loss * self._retrieval_loss_scale
            outputs["unscaled_retrieval_loss"] = ret_loss

            outputs['loss'] = total_loss
            outputs['labels'] = labels_masked
            for key, val in ret_loss_outputs.items():
                outputs[key] = val

        if output_attentions:
            outputs["attentions"] = lm_outputs.attentions
        if output_hidden_states:
            outputs["hidden_states"] = lm_outputs.hidden_states

        if self._is_debug:
            outputs['input_embeddings'] = input_embedds  # for debug purposes
            question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
            outputs['question_encoded'] = torch.cat([question_start_tokens_batch,
                                                     question_end_tokens_full], dim=1)
        return outputs


    def _compute_retrieval_loss(self,
                                outputs: Dict[str, torch.Tensor],
                                question_end_tokens: torch.Tensor,
                                input_embedds: torch.Tensor,
                                injected_embeddings: torch.Tensor,
                                output_hidden_states: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive retrival loss based on InfoNCE loss implementation.
        Args:
            outputs: a LLM outputs, containing: 'logits', 'hidden_states', etc.
            question_end_tokens: an ending tokens that contains [RET] token
                (required for finding retrieval token in sequences);
            input_embedds: full input embeddings sequences of size [bs, input_seq_len, hidden_dim].
                (As we pass embeddings to forward of LM);
            injected_embeddings: a reference embeddings (i.e. injected embeddings from other modality encoder);
            output_hidden_states: whether to output hidden_states for retrieval tokens;

        Returns:
            a dict, containing 'loss' and,  optionally, 'last_hidden_state' and 'ret_tokens_logits'.
        """
        # 1) Find locations of retrival tokens in ending prompts
        ret_tokens_question_end_mask = question_end_tokens == self.ret_token_id

        # Full mask for RET tokens location: [bs, seq_len, model_dim]
        ret_tokens_full_mask = torch.cat([
            torch.zeros((input_embedds.size(0), input_embedds.size(1) - ret_tokens_question_end_mask.size(-1))).to(input_embedds.device),
            ret_tokens_question_end_mask
        ], dim=1).bool()

        # 2) Extract hidden states and pass them through projection layers
        ret_hidden_states = []

        # 2.1) Project + loss
        # As a projection_layers can be used: projection_layers or lm_connector
        for idx, projection_layer in zip(self._n_retrieval_layers, self.projection_layers):  # [lm_connector]
            ret_hidden_states.append(
                projection_layer(outputs.hidden_states[idx][ret_tokens_full_mask])
            )  # (bs, hidden_size)

        # 2.2) Add hidden states together
        collected_last_hidden_state = torch.stack(ret_hidden_states, dim=-1).sum(dim=-1)

        # 3) calculate Contrastive loss in BLIP-2 style
        # (as single current RET vs. max(current sample multiple queries)) -> positives
        # vs. other samples from batch -> negatives
        loss_outputs = dict()
        try:
            # 3.1) Queries to RET token hidden states similarity
            # [batch_size, batch_size, num_query_tokens]
            same_sim_q2t = torch.matmul(
                injected_embeddings.unsqueeze(1), collected_last_hidden_state.unsqueeze(-1)
            ).squeeze()

            # Sequence to RET token hidden states similarity: aggregate (take max) across all query tokens
            same_sim_q2t_max, _ = same_sim_q2t.max(-1)
            # as probabilities of each batch sample belongs to the class (num_classes == batch_size)
            # -> [batch_size, batch_size]
            same_sim_q2t_max = same_sim_q2t_max / self.ret_temperature

            # 3.2) RET tokens hidden states to query similarity: [batch_size, batch_size, num_query_tokens]
            diff_sim_t2q = torch.matmul(
                collected_last_hidden_state.unsqueeze(1).unsqueeze(1), injected_embeddings.permute(0, 2, 1)
            ).squeeze()

            # RET tokens hidden states - query similarity: aggregate (take max) across all query tokens
            diff_sim_t2q_max, _ = diff_sim_t2q.max(-1)
            # as probabilities of each batch sample belongs to the class (num_classes == batch_size)
            # -> [batch_size, batch_size]
            diff_sim_t2q_max = diff_sim_t2q_max / self.ret_temperature

            # Targets
            batch_size = input_embedds.size(0)
            targets = torch.linspace(0, batch_size - 1, batch_size, dtype=int).to(input_embedds.device)

            # Total contrastive loss
            # as mean of:
            #  1) similarities RET tokens last hidden states <-> queries
            #  2) similarities queries <-> RET tokens last hidden states
            loss_contrastive = (torch.nn.functional.cross_entropy(same_sim_q2t_max, targets)  # label_smoothing=0.1
                                + torch.nn.functional.cross_entropy(diff_sim_t2q_max, targets)  # label_smoothing=0.1
                                ) / 2
            loss_outputs['loss'] = loss_contrastive
        except Exception as e:
            self._logger.error(f"!!! Exceptiom occurred during retrieval loss calculation:\n{e}")
            loss_outputs['loss'] = torch.zeros((1,), dtype=torch.float32)

        if output_hidden_states:
            loss_outputs['last_hidden_state'] = collected_last_hidden_state
        return loss_outputs

