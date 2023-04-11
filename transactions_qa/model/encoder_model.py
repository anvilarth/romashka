from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.logging_handler import get_logger
from romashka.transactions_qa.layers.connector import (make_linear_connector,
                                                       make_recurrent_connector)
from romashka.transactions_qa.utils import (mask_padding, mask_lm_labels_padding)


class EncoderSimpleModel(nn.Module):
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
                 is_debug: Optional[bool] = False):
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

        self.language_model_arch_type = None
        self.language_model_tokens_embedding_func = None
        self.whitespace_token_id = None
        self.do_freeze_tm: bool = do_freeze_tm
        self.do_freeze_lm: bool = do_freeze_lm
        self.do_freeze_connector: bool = do_freeze_connector

        self._is_debug: bool = is_debug
        self._prepare_model()

    def _set_language_model_arch_type(self):
        # In case if architecture is passed directly through the config
        if len(self.language_model.config.architectures):
            if "T5" in self.language_model.config.architectures[0]:
                self.language_model_arch_type = "T5"
            else:
                self._logger.warning(f"Provided language model architecture is not currently supported "
                                     f"`{self.language_model.config.architectures[0]}`. "
                                     "Try running on your own risk.")
        else:
            raise AttributeError(
                f"Provided language model doesn't have `architecture` attribute set correctly in configuration. "
                "Try again with different language model.")


    def _set_language_model_embedding_func(self):
        """
        Sets inner text tokens embedding function to separate it from HF model architecture.
        Note: should be called AFTER changing model embedding layer size!
        """
        if self.language_model_arch_type == "T5":  # has a .encoder.embed_tokens(...) Embedding layer
            self.language_model_tokens_embedding_func = self.language_model.encoder.embed_tokens
        else:
            self.language_model_tokens_embedding_func = self.language_model.encoder.embed_tokens
            self._logger.warning(f"Provided language model architecture is not currently supported "
                                 f"`{self.language_model.config.architectures[0]}`. "
                                 "Try running on your own risk.")
    def _prepare_model(self):
        # Set language model architecture type / family (i.e. T5/...)
        self._set_language_model_arch_type()

        # Prepare tokenizer
        self._configure_tokenizer()

        # In case any of tasks extends initial tokenizer vocab with additional tokens
        self._resize_text_embeddings()

        # Set embedding func
        self._set_language_model_embedding_func()

        # Freezing some weights
        if self.do_freeze_tm:
            self.transaction_model.eval()
            self._logger.info(f"Freezing transaction model's parameters...")
            for param in self.transaction_model.parameters():
                param.requires_grad = False

        if self.do_freeze_lm:
            self.language_model.eval()
            self._logger.info(f"Freezing language model's parameters...")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if self.do_freeze_connector:
            self.connector.eval()
            self._logger.info(f"Freezing connector layer's parameters...")
            for param in self.connector.parameters():
                param.requires_grad = False

    def _resize_text_embeddings(self):
        # For encoder-decoder-based models
        init_embeddings = self.language_model.encoder.get_input_embeddings()
        self._logger.info(f"Language model initial `num_embeddings`: {init_embeddings.num_embeddings}, "
                          f"`embedding_dim`: {init_embeddings.embedding_dim}")

        self.language_model.resize_token_embeddings(len(self.tokenizer))

        # For encoder-decoder-based models
        resized_embedds = self.language_model.encoder.get_input_embeddings()
        self._logger.info(f"Language model resized `num_embeddings`: {resized_embedds.num_embeddings}, "
                          f"`embedding_dim`: {resized_embedds.embedding_dim}")

    def _configure_tokenizer(self):
        """
        Configures the tokenizer for the model (optionally,
        can be performed before passing tokenizer instance to the model).
        """
        self.whitespace_token_id = torch.Tensor(self.tokenizer.encode(' ')).long()
        # todo: here any number of extra/additional tokens can be added to tokenizer's vocab

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                is_train: Optional[bool] = True) -> Any:
        """
        Passes input batch through:
        1) Sequence embedder model (transactions model);
        2) Connector
        3) Collate LM input sequences and pass through LM encoder
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
        batch_size = transaction_mask.size(0)

        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # next pass them to connector == linear mapping -> to LM inner dim
        transactions_embeddings = self.connector(transactions_embeddings)

        # Questions: to embedding of LM
        # torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # question_end_tokens: torch.Size([batch_size, len(max_question_end_tokens)])
        question_end_embeddings_batch = self.language_model_tokens_embedding_func(
            batch['question_end_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)

        # Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)
        if 'encoder_input_mask' in batch:
            encoder_input_mask = batch['encoder_input_mask']

        else:
            encoder_input_mask = torch.cat(
                [batch['question_start_attention_mask'],
                 batch['mask'],
                 batch['question_end_attention_mask']], dim=1
            )

        # Create answers + masks for LM's decoder inputs
        batch_answers = batch['answer_tokens']
        # was: torch.cat([qa_batch['answer_template_tokens'], qa_batch['target_tokens']], dim=1)
        batch_answers_mask = batch['answer_mask']
        # torch.cat([qa_batch['answer_mask'], qa_batch['target_attention_mask']], dim=1)

        # Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        lm_outputs = self.language_model(inputs_embeds=encoder_input,
                                         attention_mask=encoder_input_mask,
                                         labels=batch_answers,
                                         decoder_attention_mask=batch_answers_mask)
        # Create answers + masks for LM's decoder inputs
        lm_outputs['answer_tokens'] = batch_answers

        if self._is_debug:
            # Return question as:
            # Q_start_tokens + TRNS_embeddings + Q_end_tokens
            question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
            lm_outputs['question_encoded'] = torch.cat([question_start_tokens_batch,
                                                        batch['question_end_tokens']], dim=1)
            # Experimental !
            transactions_history_lengths = transaction_mask.sum(1)
            lm_outputs['transactions_history_lengths'] = transactions_history_lengths

            lm_outputs['question_start_input_size'] = question_start_embeddings_batch.size(1)
            lm_outputs['question_end_input_size'] = question_end_embeddings_batch.size(1)
            lm_outputs['transactions_input_size'] = transactions_embeddings.size(1)
            lm_outputs['total_input_size'] = encoder_input.size(1)

            lm_outputs['input_embeddings'] = encoder_input  # for debug purposes

        return lm_outputs