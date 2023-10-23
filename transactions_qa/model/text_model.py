import inspect
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.logging_handler import get_logger
from romashka.transactions_qa.utils import seed_everything

class ESQATextModel(nn.Module):
    def __init__(self,
                 language_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 do_freeze_lm: Optional[bool] = True,
                 do_freeze_lm_embeddings: Optional[bool] = True,
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.tokenizer = tokenizer
        self.language_model = language_model

        self.language_model_arch_type = None
        self.language_model_tokens_embedding_func = None
        self.do_freeze_lm: bool = do_freeze_lm
        self.do_freeze_lm_embeddings: bool = do_freeze_lm_embeddings

        self.generation_config = generation_config

        self._is_debug: bool = is_debug
        self._device_type = self.language_model.device.type  # 'cuda' or 'cpu'
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
        else:
            raise AttributeError(f"Provided language model architecture is not currently supported "
                                 f"`{self.language_model.config.architectures[0]}`.")
        embedds_mean = torch.mean(embedds, dim=0).detach()
        return embedds_mean

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

        if self.do_freeze_lm_embeddings:
            self._logger.info(f"Freezing language model's embeddings...")
            self.language_model_tokens_embedding_func.requires_grad = False
        else:
            self._logger.info(f"Unfreezing (if frozen) language model's embeddings...")
            self.language_model_tokens_embedding_func.requires_grad = True

        if self.do_freeze_connector:
            self.connector.eval()
            self._logger.info(f"Freezing connector layer's parameters...")
            for param in self.connector.parameters():
                param.requires_grad = False

        self.params_precision = 32
        if self.language_model.dtype == torch.float16:
            self.params_precision = 16
        self.params_precision = eval(f"torch.float{self.params_precision}")
        self._logger.info(f"Language model weights loaded in { self.params_precision} precision.")

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
        self.register_buffer("whitespace_token_id",
                             torch.Tensor(self.tokenizer.encode(' ', add_special_tokens=False)).long())

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
                output_attentions: Optional[bool] = False,
                is_train: Optional[bool] = True) -> Any:
        """

        Args:
            batch ():
            output_attentions ():
            is_train ():

        Returns:

        """
        pass