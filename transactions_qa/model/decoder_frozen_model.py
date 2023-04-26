import itertools
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.model.decoder_model import DecoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.utils import (mask_padding, mask_lm_labels_padding)


class DecoderFrozenModel(DecoderSimpleModel):
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 connector: Optional[nn.Module] = None,
                 connector_input_size: Optional[int] = None,
                 connector_output_size: Optional[int] = None,
                 do_freeze_tm: Optional[bool] = True,
                 do_freeze_lm: Optional[bool] = False,
                 do_freeze_connector: Optional[bool] = False,
                 transactions_embeddings_start_token: Optional[str] = r"[trx]",
                 transactions_embeddings_end_token: Optional[str] = r"[/trx]",
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):

        self._transactions_embeddings_start_token = transactions_embeddings_start_token
        self._transactions_embeddings_end_token = transactions_embeddings_end_token

        super().__init__(language_model=language_model,
                         transaction_model=transaction_model,
                         tokenizer=tokenizer,
                         connector=connector,
                         connector_input_size=connector_input_size,
                         connector_output_size=connector_output_size,
                         do_freeze_tm=do_freeze_tm,
                         do_freeze_lm=do_freeze_lm,
                         do_freeze_connector=do_freeze_connector,
                         generation_config=generation_config,
                         is_debug=is_debug)

    def _prepare_model(self):
        super()._prepare_model()
        self._create_trainable_parameters()

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

        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

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

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                is_train: Optional[bool] = True) -> Any:
        """
        Passes input batch through:
        1) Sequence embedder model (transactions model);
        2) Connector
        3) Collate CLM input sequences and pass through LM decoder
        Args:
            batch: a prepared with chosen task batch of items;
            is_train: whether to pass to LM forward input labels or not;

        Returns:
            LM model's outputs with added labels (if `is_train` was set).
        """
        # Get transactions embeddings for initial batch
        # transactions model requires: ['mask', 'cat_features', 'num_features', 'meta_features']
        # + optionally: 'time' - ??? maybe 'event_time' ???
        # return: Tuple[
        # torch.Tensor, - embeddings -> we need this
        # torch.Tensor - mask
        # ]
        transaction_mask = batch['mask']
        device = transaction_mask.device

        batch_size = transaction_mask.size(0)
        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # next pass them to connector == linear mapping -> to LM inner dim
        transactions_embeddings = self.connector(transactions_embeddings)

        # Questions: to embedding of LM
        # torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)

        # if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)
        # otherwise append it to the end of starting sequence
        else:
            question_start_embeddings = torch.cat([question_start_embeddings,
                                                   self.transactions_start_embedding[None, None]], dim=1)

        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # Question ends: to embedding of LM
        # 1) Strip paddings from questions endings!!!
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

        # 2) Pad to max q+a length
        max_question_answer_len = max([len(qa) for qa in question_end_tokens_full])
        for i in range(question_end_tokens_mask.size(0)):
            n_padds = max_question_answer_len - question_end_tokens_full[i].size(0)
            question_end_tokens_full[i] = torch.cat(
                [torch.full((n_padds,), self.tokenizer.pad_token_id).to(device),
                 question_end_tokens_full[i],
                 ], dim=0)

        # 3) Cat back into batch
        question_end_tokens_full = torch.stack(question_end_tokens_full).long()

        # Get LLM embeddings
        question_end_embeddings_batch = self.language_model_tokens_embedding_func(question_end_tokens_full)

        # 4) Fill with trainable parameters
        # if it already starts with [/trx]
        if self.has_end_token(question_end_tokens_full):
            self.replace_end_token(question_end_tokens_full, question_end_embeddings_batch)

        # otherwise prepend it to the start of question sequence
        else:
            question_start_embeddings = torch.cat([
                self.transactions_end_embedding[None, None].repeat(batch_size, 1, 1),
                question_end_embeddings_batch], dim=1)

        # Get general LM's input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        input_embedds = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)

        # 1) Label = [question_start_tokens, <trns>,
        #           <pad> * trns_history_len,
        #           <pad> * n, </trns>,
        #           question_end_tokens, answer_tokens,
        #           <eos> - ?]
        labels = torch.cat([
            batch['question_start_tokens'].repeat(batch_size, 1).to(device),
            torch.full(transactions_embeddings.size()[:2], self.tokenizer.pad_token_id).to(device),
            question_end_tokens_full
        ], dim=1)

        labels_masked = mask_lm_labels_padding(labels, self.tokenizer.pad_token_id).long().to(device)

        # Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        output = self.language_model(inputs_embeds=input_embedds,
                                     labels=labels_masked if is_train else None,
                                     output_hidden_states=True)
        if is_train:
            output['labels'] = labels_masked
        if self._is_debug:
            output['input_embeddings'] = input_embedds  # for debug purposes
            question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
            output['question_encoded'] = torch.cat([question_start_tokens_batch,
                                                    batch['question_end_tokens']], dim=1)
            # Experimental !
            transactions_history_lengths = transaction_mask.sum(1)
            output['transactions_history_lengths'] = transactions_history_lengths

            output['question_start_input_size'] = question_start_embeddings_batch.size(1)
            output['question_end_input_size'] = question_end_embeddings_batch.size(1)
            output['transactions_input_size'] = transactions_embeddings.size(1)
            output['total_input_size'] = input_embedds.size(1)
        return output

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