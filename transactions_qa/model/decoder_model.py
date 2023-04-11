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


class DecoderSimpleModel(nn.Module):
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
            if "OPT" in self.language_model.config.architectures[0]:
                self.language_model_arch_type = "OPT"  # has a .model.decoder attribute
            elif "GPT" in self.language_model.config.architectures[0]:
                self.language_model_arch_type = "GPT"  # has a .transformer attribute
            else:
                raise AttributeError(f"Provided language model architecture is not currently supported "
                                     f"`{self.language_model.config.architectures[0]}`. "
                                     "Try again with different language model.")
        else:
            raise AttributeError(
                f"Provided language model doesn't have `architecture` attribute set correctly in configuration. "
                "Try again with different language model.")

    def _set_language_model_embedding_func(self):
        """
        Sets inner text tokens embedding function to separate it from HF model architecture.
        Note: should be called AFTER changing model embedding layer size!
        """
        if self.language_model_arch_type == "OPT":  # has a .model.decoder.embed_tokens(...) Embedding layer
            self.language_model_tokens_embedding_func = self.language_model.model.decoder.embed_tokens
        elif self.language_model_arch_type == "GPT":  # has a .transformer.wte(...) Embedding layer
            self.language_model_tokens_embedding_func = self.language_model.transformer.wte
        else:
            raise AttributeError(f"Provided language model architecture is not currently supported "
                                 f"`{self.language_model_arch_type}`. "
                                 "Try again with different language model.")

    def _prepare_model(self):
        # Set language model architecture type / family (i.e. GPT2/Neo/J .../OPT)
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
        # For OPT-based models
        if self.language_model_arch_type == "OPT":
            init_embeddings = self.language_model.get_input_embeddings()
        # For GPT-based
        else:
            init_embeddings = self.language_model.transformer.get_input_embeddings()

        self._logger.info(f"Language model initial `num_embeddings`: {init_embeddings.num_embeddings}, "
                          f"`embedding_dim`: {init_embeddings.embedding_dim}")

        self.language_model.resize_token_embeddings(len(self.tokenizer))

        # For OPT-based models
        if self.language_model_arch_type == "OPT":
            resized_embedds = self.language_model.model.decoder.get_input_embeddings()
        # For GPT-based
        else:
            resized_embedds = self.language_model.transformer.get_input_embeddings()

        self._logger.info(f"Language model resized `num_embeddings`: {resized_embedds.num_embeddings}, "
                          f"`embedding_dim`: {resized_embedds.embedding_dim}")

    def _configure_tokenizer(self):
        """
        Configures the tokenizer for the model (optionally,
        can be performed before passing tokenizer instance to the model).
        """
        self.whitespace_token_id = torch.Tensor(self.tokenizer.encode(' ')).long()

        if self.language_model_arch_type == "OPT":
            # setup padding
            self.tokenizer.pad_token_id = 1
            self.tokenizer.pad_token = "<pad>"
            self.tokenizer.padding_side = "left"

            # setup truncation
            self.tokenizer.truncation_side = "left"

            # setup special tokens
            self.tokenizer.bos_token_id = 0
            self.tokenizer.bos_token = "<s>"

            self.tokenizer.eos_token_id = 2
            self.tokenizer.eos_token = "</s>"

            self.tokenizer.unk_token = "<unk>"
            self.tokenizer.unk_token_id = 3
        else:
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "<|endoftext|>"
                    self.tokenizer.eos_token = "<|endoftext|>"
            self.tokenizer.padding_side = 'left'

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
        batch_size = transaction_mask.size(0)
        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # next pass them to connector == linear mapping -> to LM inner dim
        transactions_embeddings = self.connector(transactions_embeddings)

        # Questions: to embedding of LM
        # torch.Size([1, len(question_start_tokens))
        print(f"Got question_start_tokens: {batch['question_start_tokens']}")
        # print(f"Got question_start_tokens: {self.tokenizer.decode(batch['question_start_tokens'])}")
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # question_end_tokens: torch.Size([batch_size, len(max_question_end_tokens)])
        # 1) Strip paddings from questions endings!!!
        question_end_tokens_mask = batch['question_end_attention_mask'].bool()  # 0 - token, 1 == pad

        question_end_tokens_full = []
        for i in range(question_end_tokens_mask.size(0)):
            question_end_tokens_ = batch['question_end_tokens'][i][
                question_end_tokens_mask[i]]  # question without padding
            answer_ = batch['answer_tokens'][i]
            question_end_tokens_full.append(torch.cat([question_end_tokens_, self.whitespace_token_id, answer_], dim=0))

        # 2) Pad to max q+a length
        max_question_answer_len = max([len(qa) for qa in question_end_tokens_full])
        for i in range(question_end_tokens_mask.size(0)):
            n_padds = max_question_answer_len - question_end_tokens_full[i].size(0)
            question_end_tokens_full[i] = torch.cat(
                [question_end_tokens_full[i], torch.full((n_padds,), self.tokenizer.pad_token_id)], dim=0)

        # 3) Cat back into batch
        question_end_tokens_full = torch.stack(question_end_tokens_full)

        question_end_embeddings_batch = self.language_model_tokens_embedding_func(question_end_tokens_full)

        # Get general LM's input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        input_embedds = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)

        # Create CLM labels, todo: use later for Retrieval/Captioning task
        # question_start_tokens_mask = ~batch['question_start_tokens_mask'].bool()  # 0 - token, 1 == pad
        # transactions_tokens_mask = torch.ones(transactions_embeddings.size()[:2]).bool()  # all to 1 == pad
        # question_end_tokens_mask = mask_padding(question_end_tokens_full)  # 0 - token, 1 == pad

        # Label = [question_start_tokens, <trns>,
        #           <pad> * trns_history_len,
        #           question_end_tokens, answer_tokens,
        #           <pad> - ?]
        labels = torch.cat([
            batch['question_start_tokens'].repeat(batch_size, 1),
            torch.full(transactions_embeddings.size()[:2], self.tokenizer.pad_token_id),
            question_end_tokens_full
        ], dim=1)
        labels_masked = mask_lm_labels_padding(labels, self.tokenizer.pad_token_id).long()

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

    def generate(self, batch: Union[Dict[str, torch.Tensor], Any],
                 max_len: Optional[int] = 32,
                 temperature: Optional[float] = 0.0,
                 top_p: Optional[float] = 1.0,
                 min_word_tokens: Optional[int] = 0,
                 ret_scale_factor: Optional[float] = 1.0,
                 filter_value: Optional[float] = -float('Inf'), **kwargs):
        """
        Runs greedy decoding and returns generated captions.

        Args:
          batch: a batch of input samples for autoregressive generation;
          max_len: Maximum number of tokens to generate;
          temperature: Used to modulate logit distribution;
          top_p: If set to < 1, the smallest set of tokens with highest probabilities
                that add up to top_p or higher are kept for generation;
          min_word_tokens: Minimum number of words to generate before allowing a [RET] output;
          ret_scale_factor: Proportion to scale [RET] token logits by.
                            A higher value may increase the probability of the model generating [RET] outputs;
          filter_value: Value to assign to tokens that should never be generated;
          **kwargs: other arguments from transformers.GenerationConfig().
        Returns:
          out: (N, T) int32 sequence of output tokens;
          output_embeddings: (N, T, 256) sequence of text output embeddings.
        """
        # self.transaction_model.eval()
        # self.connector.eval()
        # self.language_model.eval()
        self.eval()  # freeze all at once

        with torch.no_grad():  # no tracking history
            pass  # todo later
