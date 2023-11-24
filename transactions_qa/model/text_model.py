import inspect
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.logging_handler import get_logger
from romashka.transactions_qa.model.pooler import PoolerType
from romashka.transactions_qa.model.projection import ProjectionsType


class TextEncoderModel(nn.Module):
    def __init__(self,
                 language_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 max_input_sequence_len: Optional[int] = 4096,
                 pooler_type: str = "CLS_POOLER",
                 projection_type: str = "LINEAR",
                 shared_dim: int = 768,
                 do_freeze_lm: Optional[bool] = True,
                 do_freeze_lm_embeddings: Optional[bool] = True,
                 embeddings_dropout_p: Optional[float] = 0.1,
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__
        )
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.pooler_type = pooler_type
        self.pooler = None
        self.projection_type = projection_type
        self.projection = None
        self.shared_dim = shared_dim
        self.model_dim: int = None

        self.max_input_sequence_len: int = max_input_sequence_len
        self.is_encoder_decoder: bool = False
        self.language_model_arch_type: str = None
        self.language_model_tokens_embedding_func = None
        self.do_freeze_lm: bool = do_freeze_lm
        self.do_freeze_lm_embeddings: bool = do_freeze_lm_embeddings
        self.embeddings_dropout_p: float = embeddings_dropout_p

        self.generation_config = generation_config

        self._is_debug: bool = is_debug
        self._device_type = self.language_model.device.type  # 'cuda' or 'cpu'
        self._prepare_model()

    def _set_language_model_arch_type(self):
        # In case if architecture is passed directly through the config
        if len(self.language_model.config.architectures):
            if "T5" in self.language_model.config.architectures[0]:
                self.language_model_arch_type = "T5"
                self.is_encoder_decoder = True
            elif "OPT" in self.language_model.config.architectures[0]:
                self.language_model_arch_type = "OPT"  # has a .model.decoder attribute
            elif "GPTNeoX" in self.language_model.config.architectures[0]:   # required for Pythia and GPT-NeoX
                self.language_model_arch_type = "GPTNeoX"  # has a .model.gpt_neox attribute
            elif "GPT" in self.language_model.config.architectures[0]:  # other GPT-like models
                self.language_model_arch_type = "GPT"  # has a .transformer attribute
            elif "Llama" in self.language_model.config.architectures[0]:  # other Llama 1/2 models
                self.language_model_arch_type = "Llama"  # has a .transformer attribute
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
        elif self.language_model_arch_type == "OPT":  # has a .model.decoder.embed_tokens(...) Embedding layer
            self.language_model_tokens_embedding_func = self.language_model.model.decoder.embed_tokens
        elif self.language_model_arch_type == "GPTNeoX":  # has a .gpt_neox.embed_in(...) Embedding layer
            self.language_model_tokens_embedding_func = self.language_model.gpt_neox.embed_in
        elif self.language_model_arch_type == "GPT":  # has a .transformer.wte(...) Embedding layer
            self.language_model_tokens_embedding_func = self.language_model.transformer.wte
        elif self.language_model_arch_type == "Llama":
            self.language_model_tokens_embedding_func = self.language_model.model.embed_tokens
        else:
            self.language_model_tokens_embedding_func = self.language_model.encoder.embed_tokens
            self._logger.warning(f"Provided language model architecture is not currently supported "
                                 f"`{self.language_model.config.architectures[0]}`. "
                                 "Try running on your own risk.")

    def _create_mean_lm_embedding(self) -> torch.Tensor:
        """
        Creates an embedding vector based on all LLM input embeddings
        averaged across vocabulary.
        Returns: an embedding tensor, with size (embedd_dim,)
        """
        embedds = None
        if (self.language_model_arch_type == "T5") \
                and hasattr(self.language_model.config, "hidden_size"):
            embedds = self.language_model.encoder.embed_tokens.weight.cpu()
        elif self.language_model_arch_type == "OPT":
            embedds = self.language_model.model.decoder.embed_tokens.weight.cpu()
        elif self.language_model_arch_type == "GPTNeoX":
            embedds = self.language_model.gpt_neox.embed_in.weight.cpu()
        elif self.language_model_arch_type == "GPT":
            embedds = self.language_model.transformer.wte.weight.cpu()
        elif self.language_model_arch_type == "GPT":
            embedds = self.language_model.model.embed_tokens.weight.cpu()
        else:
            raise AttributeError(f"Provided language model architecture is not currently supported "
                                 f"`{self.language_model.config.architectures[0]}`.")
        embedds_mean = torch.mean(embedds, dim=0).detach()
        return embedds_mean

    def _set_model_dim(self):
        """
        Sets the hidden size / output embeddings dimension of the model.
        """
        if hasattr(self.language_model.config, "d_model"):
            self.model_dim = self.language_model.config.d_model
        elif self.language_model.config.hidden_size:
            self.model_dim = self.language_model.config.hidden_size
        else:
            raise AttributeError(f"Unable to estimate Language model output embeddings dimension!")

    def _prepare_model(self):
        # Set language model architecture type / family (i.e. T5/...)
        self._set_language_model_arch_type()

        # Sets the hidden size / output embeddings dimension of the model
        self._set_model_dim()

        self.params_precision = 32
        if self.language_model.dtype == torch.float16:
            self.params_precision = 16
        self.params_precision = eval(f"torch.float{self.params_precision}")
        self._logger.info(f"Language model weights loaded in {self.params_precision} precision.")

        # Prepare tokenizer
        self._configure_tokenizer()

        # In case any of tasks extends initial tokenizer vocab with additional tokens
        self._resize_text_embeddings()

        # Additionally re-assign embeddings
        self._set_language_model_embedding_func()

        # Freezing some weights
        if self.do_freeze_lm:
            self.language_model.eval()
            self._logger.info(f"Freezing language model's parameters...")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if self.do_freeze_lm_embeddings:
            self._logger.info(f"Freezing language model's embeddings...")
            self.language_model_tokens_embedding_func.requires_grad = False
        else:
            self._logger.info(f"Unfreezing (if frozen) language model's embeddings...")
            self.language_model_tokens_embedding_func.requires_grad = True

        # Create additional layers
        self._create_pooler()
        self._create_projection()

        # Check total trainable parameters
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        self._logger.info(f"Totally trainable parameters: {len(trainable_parameters)} from {len(parameters)}")

    def _create_pooler(self):
        """
        Creates a Pooler network.
        """
        self.pooler = PoolerType.get(pooler_type_name=self.pooler_type)
        self._logger.info(f"Created `{self.pooler_type}` pooler.")

    def _create_projection(self):
        """
        Creates a Projection network.
        """
        proj_kwargs = {'in_dim': self.model_dim, "out_dim": self.shared_dim}
        self.projection = ProjectionsType.get(projection_type_name=self.projection_type, **proj_kwargs)
        self._logger.info(f"Created `{self.projection_type}` projection.")

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
        self._logger.info(f"Initial tokenizer has `model_max_length` = {self.tokenizer.model_max_length}")
        self.tokenizer.model_max_length = self.max_input_sequence_len
        self._logger.info(f"Change it to `model_max_length` = {self.tokenizer.model_max_length}")
        self.register_buffer("whitespace_token_id",
                             torch.Tensor(self.tokenizer.encode(' ', add_special_tokens=False)).long())

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "<|endoftext|>"
                self.tokenizer.eos_token = "<|endoftext|>"

        if hasattr(self.language_model.config, "is_encoder_decoder") or (self.language_model_arch_type == "T5"):
            self.tokenizer.padding_side = 'right'
        else:
            self.tokenizer.padding_side = 'left'

    def _set_generation_config(self):
        """
        Configure model's parameters for generation (if generation config was not specified on init() stage).
        """
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 3,
                "min_new_tokens": 1,
                "top_p": 1.0,
                "temperature": 0.0,  # 0.0 - greedy decoding
                "hidden_dims_indexes": [-1],  # Which hidden dims to take
                "filter_value": -float('Inf'),  # Value to assign to tokens that should never be generated.
                "create_allowed_token_ids": False,
                "allowed_token_ids": None,
                "create_stopping_criteria": False,
                "stopping_criteria": None,
                "seed": 42
            }
            self._logger.info(f"Created default generation configration for a model:\n"
                              f"{self.generation_config}")

    @staticmethod
    def inspect_forward_signature(param_name: str, model: nn.Module) -> bool:
        """
        Get the list of parameter names of `forward` function of the model
        and checks whether requested parameter name is in list.
        Args:
            param_name: str, a requested parameter name;
            model: nn.Module, a model to get `forward` function from;
        Returns:
            a bool flag, whether requested parameter name is in parameter names list.
        """
        # Inspect model forward signature to keep only the arguments it accepts
        signature = inspect.signature(model.forward)
        if param_name in list(signature.parameters.keys()):
            return True
        return False

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                output_attentions: Optional[bool] = False) -> Any:
        """

        Simply passes input batch through LLM and output predictions.
        Args:
            batch: a prepared with chosen task batch of items;
            output_attentions: whether to output attentions;
            is_train: whether to pass to LM forward input labels or not;

        Returns:
            LM model's outputs with added labels (if `is_train` was set).

        """
        # batch: 'captions'
        # 1) Tokenize text captions
        captions_tokenized = self.tokenizer([cap[0] for cap in batch['captions']],
                                            return_tensors='pt',
                                            padding=True,
                                            truncation=True,
                                            return_attention_mask=True)

        # 2) Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        labels = captions_tokenized['input_ids'].clone()
        lm_outputs = self.language_model(input_ids=captions_tokenized['input_ids'],
                                         attention_mask=captions_tokenized['attention_mask'],
                                         labels=labels,
                                         output_attentions=output_attentions,
                                         output_hidden_states=True)
        # 3) Pass through pooler
        pooled_outputs = self.pooler(lm_outputs, captions_tokenized['attention_mask'])

        # 4) Pass through projection
        projected_outputs = self.projection(pooled_outputs)

        # Create answers + masks for LM's decoder inputs
        lm_outputs['input_ids'] = batch['input_ids']
        lm_outputs['attention_mask'] = batch['attention_mask']
        lm_outputs['labels'] = batch['target_tokens']
        lm_outputs['projected_outputs'] = projected_outputs

        return lm_outputs
