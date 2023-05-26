from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.model.encoder_model import EncoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.utils import mask_padding
from romashka.transactions_qa.layers.numerical_head import LinearHead, MLPHead


class EncoderNumericModel(EncoderSimpleModel):
    """
    Encoder-decoder model with additional parameters for numeric problems solution.
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
                 embeddings_dropout_p: Optional[float] = 0.1,
                 numeric_head_type: Optional[str] = 'linear',
                 numeric_loss_scale: Optional[float] = 1.0,
                 text_loss_scale: Optional[float] = 1.0,
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

        self.numeric_head_type = numeric_head_type

        self._embeddings_dropout_p = embeddings_dropout_p
        self._text_loss_scale = text_loss_scale
        self.numeric_loss_scale = numeric_loss_scale

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

        self.whitespace_token_id = torch.Tensor(self.tokenizer.encode(" ")).long()
        self.eos_token_id = torch.Tensor([self.tokenizer.eos_token_id]).long()

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
            - numeric token: [NUM];
        Note: those parameters need to be passed to separate optimizer (with connector & projections layers)!
        i.e:
            opt = Adafactor(
                list(projection.parameters())
                + [trns_start_embedding, trns_end_embedding], lr=1e-2, relative_step=False)
        """
        # Create transactions embeddings start/end tokens: [trx] / [/trx] and trainable parameters for them
        self._create_surrounding_parameters()
        # Create numeric token: [NUM] in tokenizers vocabulary and mappings token <-> id
        self._create_retrieval_parameters()

        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

        # Create projection layers from LM output hidden states to outputs (-> heads)
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

    def _create_numeric_parameters(self):
        """
        Creates trainable parameters for numeric heads;
        """
        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # Create trainable heads
        # 1) Binary classification head: numeric vs. textual token
        self.token_type_classifier = nn.Linear(params_dim, 2)

        # 2) Exponent head: as classifier to one of N exponents: from -8 ... 8
        exponent_size = 17  # as all values from -8 ... 8
        if self.numeric_head_type == "linear":
            self.exponent_head = LinearHead(params_dim, exponent_size)
        elif self.numeric_head_type == "mlp":
            self.exponent_head = MLPHead(params_dim, exponent_size)
        else:
            raise AttributeError(f"Provided numeric head type ({self.exponent_head}) "
                                 f"doesn't match any of currently supported: `linear`, `mlp`.")

        # 2) Exponent head: as classifier to one of N exponents: from -8 ... 8
        exponent_size = 17  # as all values from -8 ... 8
        if self.numeric_head_type == "linear":
            self.exponent_head = LinearHead(params_dim, exponent_size)
        elif self.numeric_head_type == "mlp":
            self.exponent_head = MLPHead(params_dim, exponent_size)
        else:
            raise AttributeError(f"Provided numeric head type ({self.exponent_head}) "
                                 f"doesn't match any of currently supported: `linear`, `mlp`.")

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