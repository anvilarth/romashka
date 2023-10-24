import inspect
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.logging_handler import get_logger
from romashka.transactions_qa.utils import seed_everything
from romashka.transactions_qa.model.generation_utils import isin
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.layers.initialization import (init_embeddings_with_tensor,
                                                            init_parameter_with_tensor)
from romashka.transactions_qa.tasks.task_token_updater import collect_task_specific_tokens


class ESQATextModel(nn.Module):
    def __init__(self,
                 language_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 tasks: Optional[List[AbstractTask]] = None,
                 max_input_sequence_len: Optional[int] = 4096,
                 do_freeze_lm: Optional[bool] = True,
                 do_freeze_lm_embeddings: Optional[bool] = True,
                 embeddings_dropout_p: Optional[float] = 0.1,
                 transactions_embeddings_start_token: Optional[str] = r"[trx]",
                 transactions_embeddings_end_token: Optional[str] = r"[/trx]",
                 generation_config: Optional[Dict[str, Any]] = None,
                 is_debug: Optional[bool] = False):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.tasks = tasks if tasks is not None else []

        self.max_input_sequence_len = max_input_sequence_len
        self.language_model_arch_type = None
        self.language_model_tokens_embedding_func = None
        self.do_freeze_lm: bool = do_freeze_lm
        self.do_freeze_lm_embeddings: bool = do_freeze_lm_embeddings
        self.embeddings_dropout_p = embeddings_dropout_p
        self._transactions_embeddings_start_token = transactions_embeddings_start_token
        self._transactions_embeddings_end_token = transactions_embeddings_end_token

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

        self.params_precision = 32
        if self.language_model.dtype == torch.float16:
            self.params_precision = 16
        self.params_precision = eval(f"torch.float{self.params_precision}")
        self._logger.info(f"Language model weights loaded in {self.params_precision} precision.")

        # Prepare tokenizer
        self._configure_tokenizer()

        # Create mean embedding
        self.lm_mean_embedding = self._create_mean_lm_embedding()

        # In case any of tasks extends initial tokenizer vocab with additional tokens
        self._resize_text_embeddings()

        self._create_trainable_parameters()

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

        # Create trainable task-specific tokens
        self._create_trainable_task_special_tokens()

        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

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
            return_ids=True,
            special=True
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
            requires_grad=True).to(self.params_precision)
        self.transactions_end_embedding = nn.Parameter(
            torch.normal(mean=torch.zeros(params_dim), std=torch.ones(params_dim)),
            requires_grad=True).to(self.params_precision)

        init_parameter_with_tensor(self.transactions_start_embedding, self.lm_mean_embedding)
        init_parameter_with_tensor(self.transactions_end_embedding, self.lm_mean_embedding)
        self._logger.info(f"Initialized trainable parameters for transactions embeddings start/end tokens.")

    def _create_trainable_task_special_tokens(self):
        """
        Creates trainable parameters for task special tokens.
        """
        self.task_special_tokens = collect_task_specific_tokens(self.tasks)
        self.register_buffer("task_special_tokens_ids",
                             self.tokenizer(self.task_special_tokens,
                                            add_special_tokens=False,
                                            padding=False,
                                            return_tensors='pt')['input_ids'].flatten())
        self._logger.info(f"Collected {len(self.task_special_tokens)} task special tokens: {self.task_special_tokens} "
                          f"with corresponding ids: {self.task_special_tokens_ids}")
        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # Embeddings for special tokens
        self.task_special_tokens_embeddings = nn.Embedding(num_embeddings=self.task_special_tokens_ids.size(0),
                                                           embedding_dim=params_dim).to(self.params_precision)
        init_embeddings_with_tensor(self.task_special_tokens_embeddings, self.lm_mean_embedding)

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
        input_embeddings[mask] = self.transactions_start_embedding.to(
            input_tokens_ids.device).to(input_embeddings.dtype)

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
        input_embeddings[mask] = self.transactions_end_embedding.to(
            input_tokens_ids.device).to(input_embeddings.dtype)

    def has_task_tokens(self, input_tokens_ids: Union[List[int], torch.Tensor]) -> bool:
        """
        Checks whether any task special token id already contained in given ids.
        """
        return isin(input_tokens_ids, self.task_special_tokens_ids).sum() > 0

    def replace_task_tokens(self,
                            input_tokens_ids: Union[List[int], torch.Tensor],
                            input_embeddings: torch.Tensor):
        """
        Replace task special tokens' embedding with trainable parameters.
        """
        mask = isin(input_tokens_ids, self.task_special_tokens_ids)
        embs = self.task_special_tokens_embeddings(input_tokens_ids[mask] - min(self.task_special_tokens_ids))
        input_embeddings[mask] = embs

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                output_attentions: Optional[bool] = False,
                is_train: Optional[bool] = True) -> Any:
        """

        Simply passes input batch through LLM and output predictions.
        Args:
            batch: a prepared with chosen task batch of items;
            output_attentions: whether to output attentions;
            is_train: whether to pass to LM forward input labels or not;

        Returns:
            LM model's outputs with added labels (if `is_train` was set).

        """
        # batch: 'input_ids', 'attention_mask', 'target_tokens', 'target_attention_mask'
        # 1) Input ids to embedding of LM
        input_embeddings = self.language_model_tokens_embedding_func(
            batch['input_ids'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)

        # 2) if it contains [trx]
        if self.has_start_token(batch['input_ids']):
            self.replace_start_token(batch['input_ids'], input_embeddings)

        # 3) if it contains [/trx]
        if self.has_end_token(batch['input_ids']):
            self.replace_end_token(batch['input_ids'], input_embeddings)

        # 4) Replace task special tokens embeddings with trainable parameters
        if self.has_task_tokens(batch['input_ids']):
            self.replace_task_tokens(batch['input_ids'], input_embeddings)

        # Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        lm_outputs = self.language_model(inputs_embeds=input_embeddings,
                                         attention_mask=batch['attention_mask'],
                                         labels=batch['target_tokens'],
                                         output_attentions=False,
                                         output_hidden_states=False,
                                         decoder_attention_mask=batch['target_attention_mask'])
        # Create answers + masks for LM's decoder inputs
        lm_outputs['input_ids'] = batch['input_ids']
        lm_outputs['attention_mask'] = batch['attention_mask']
        lm_outputs['labels'] = batch['target_tokens']

        return lm_outputs
