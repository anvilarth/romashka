import itertools
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.model.decoder_model import DecoderSimpleModel


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
                 do_freeze_connector: Optional[bool] = False,
                 min_ret_tokens: Optional[int] = 150,  # equals to max transactions history size
                 max_ret_tokens: Optional[int] = 150,  # equals to min transactions history size
                 n_retrieval_layers: Optional[List[int]] = None,
                 embeddings_dropout_p: Optional[float] = 0.1,
                 transactions_embeddings_start_token: Optional[str] = r"[trx]",
                transactions_embeddings_end_token: Optional[str] = r"[/trx]",
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):
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
        self.min_ret_tokens = min_ret_tokens
        self.max_ret_tokens = max_ret_tokens
        self._ret_tokens_template = "[RET_%s]"

        self.projection_layers = nn.ModuleList([])
        self._n_retrieval_layers = n_retrieval_layers
        self._embeddings_dropout_p = embeddings_dropout_p
        self._transactions_embeddings_start_token = transactions_embeddings_start_token
        self._transactions_embeddings_end_token = transactions_embeddings_end_token

    def _prepare_model(self):
        super()._prepare_model()
        self._create_projection_layers()
        self._add_retrieval_tokens()

    def _create_retrieval_parameters(self):
        """
        Creates trainable parameters for:
            - retrieval tokens: RET_0 ... RET_N;
            - transactions embeddings start/end tokens: [trx] / [/trx];
        Note: those parameters need to be passed to separate optimizer (with connector & projections layers)!
        i.e:
            opt = Adafactor(
                list(projection.parameters())
                + [trns_start_embedding, trns_end_embedding]
                + [ret_embeddings], lr=1e-2, relative_step=False)
        """
        # Check if:
        #   - transactions embeddings start/end tokens
        #   - retrieval tokens
        # exists in tokenizers' vocabulary, add them if not exist

        # Create randomly initialized embeddings for each of them, of size [n_embeddings, hidden_dim]

        # Get their ids in tokenizers' vocabulary
        # and put in LLM embeddings matrix their created embeddings

    def _create_projection_layers(self):
        """
        Creates a linear mappings from language model hidden dimensionality
        to shared embeddings dimensionality for rET tokens loss calculation.
        """
        # List of indexes of hidden states to take for information extraction
        n_retrieval_layers = self._n_retrieval_layers if self._n_retrieval_layers is not None else [-1]

        shared_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            shared_dim = self.language_model.config.hidden_size
        else:
            raise AttributeError(f"The default setting, where shared embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` in config.")

        for layer_idx in n_retrieval_layers:
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

    def _add_retrieval_tokens(self):
        """
        Add a list of retrieval tokens to the tokenizer vocabulary.
        If the new tokens are not already in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.
        """
        self.ret_tokens = [self._ret_tokens_template % str(i) for i in range(self.max_ret_tokens)]
        num_added_toks = self.tokenizer.add_tokens(self.ret_tokens, special_tokens=False)
        self._logger.info(f"Retrieval tokens added to tokenizer: {num_added_toks}\ntokens: {self.ret_tokens}.")

        # In case any of tasks extends initial tokenizer vocab with additional tokens
        self._resize_text_embeddings()

        ret_tokens_ids = self.tokenizer(self.ret_tokens)['input_ids']
        self.ret_tokens_ids = list(itertools.chain.from_iterable(ret_tokens_ids))
