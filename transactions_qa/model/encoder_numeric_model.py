import re
from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.transactions_qa.model.encoder_model import EncoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.layers.numerical_head import LinearHead, MLPHead
from romashka.transactions_qa.utils import (get_mantissa_number, get_exponent_number,
                                            mask_padding, count_parameters, get_number_from_parts)
from romashka.transactions_qa.evaluation.eval_processings_utils import (convert_to_numeric,
                                                                        check_if_numeric)


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
                 numeric_embedding_dim: Optional[int] = 64,
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

        # Numeric attributes
        self.numeric_head_type = numeric_head_type
        self.numeric_embedding_dim = numeric_embedding_dim

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
        _ = count_parameters(self.transaction_model)
        _ = count_parameters(self.connector)
        _ = count_parameters(self.language_model)

    def _create_trainable_parameters(self):
        """
        Creates trainable parameters for:
            - transactions embeddings start/end tokens: [trx] / [/trx];
            - numeric token: [NUM];
        """
        # Create transactions embeddings start/end tokens: [trx] / [/trx] and trainable parameters for them
        self._create_surrounding_parameters()
        # Create input and output parameters for dealing with numeric tasks
        self._create_numeric_parameters()
        # Additionally call to re-init embedding function reference to resized (maybe) embeddings
        self._resize_text_embeddings()

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
        # Dimensions and fixed attributes
        self.mantissa_dim = 3 * (self.numeric_embedding_dim // 4)
        self.exponent_dim = self.numeric_embedding_dim // 4
        self.numeric_vocab_token_ids = self.collect_numeric_tokens()

        # Input embeddings
        # Exponent embeddings == 21 embedding for range {-8;-7; ... 11; 12} + 2 (for INF and -INF)
        # Embedding index == exponent + abs(-8) --- as all embedding indexes should be positive
        # If > 12 -> +INF (embedding index = [21])
        # If < -8 -> -INF (embedding index = [22])
        self.exponent_available_range = (-8, 12)
        self.num_exponent_embedds = len(list(range(self.exponent_available_range[0],
                                                   self.exponent_available_range[1] + 1))) + 2
        self.exponent_embeddings = torch.nn.Embedding(num_embeddings=self.num_exponent_embedds,
                                                      embedding_dim=self.exponent_dim)

        # Mantissa -> to N (== 3/4 * d) prototypes in range {-10 .. +10} of dim [1,]
        self.mantissa_available_range = (-10, 10)
        self.num_mantissa_prototypes = self.mantissa_dim
        self.mantissa_smoothing = 1e-2

        mantissa_prototypes = [(20 / (self.mantissa_dim - 1)) * i - 10 for i in range(self.mantissa_dim)]
        self.register_buffer("mantissa_prototypes", torch.Tensor(mantissa_prototypes))

        params_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            params_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            params_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")

        # A projection layer for mapping text + numeric embeddings to embedding size of LLM
        self.text_numeric_projection = nn.Linear((self.numeric_embedding_dim + params_dim), params_dim)

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

        # 2) Mantissa head: as classifier to one of M exponents: ???
        mantissa_head_output_size = 1
        if self.numeric_head_type == "linear":
            self.mantissa_head = LinearHead(params_dim, mantissa_head_output_size)
        elif self.numeric_head_type == "mlp":
            self.mantissa_head = MLPHead(params_dim, mantissa_head_output_size)
        else:
            raise AttributeError(f"Provided numeric head type ({self.numeric_head_type}) "
                                 f"doesn't match any of currently supported: `linear`, `mlp`.")

        # 4) Text LM head
        self.text_lm_head = nn.Linear(params_dim, self.language_model.config.vocab_size, bias=False)

    def _create_losses(self):
        # Use CE for general text loss
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                output_attentions: Optional[bool] = True,
                output_hidden_states: Optional[bool] = True,
                with_numeric_input: Optional[bool] = None,
                with_numeric_output: Optional[bool] = None,
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
            with_numeric_input: whether to process inputs with special logic to numerical tokens;
            with_numeric_output: whether to decode model outputs with special logic for numerical tokens generation;
            is_train: whether to pass to LM forward input labels and return predictions or not;

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
        with_numeric_input = with_numeric_input if with_numeric_input is not None else batch.get('with_numeric_input',
                                                                                                 False)
        with_numeric_output = with_numeric_output if with_numeric_output is not None else batch.get(
            'with_numeric_output',
            False)

        # print(f"forward() on batch with with_numeric_input = {with_numeric_input} "
        #       f"and with_numeric_output = {with_numeric_output}.")

        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # 2) Next pass them to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_embeddings = self.connector(transactions_embeddings,
                                                     mask=transactions_embeddings_mask)
        else:
            transactions_embeddings = self.connector(transactions_embeddings)

        # 3) Questions: to embedding of LM - torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_start_attention_mask = batch['question_start_tokens_mask']
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # Insert trainable parameter if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)

        # 5) Question ends: to embedding of LM
        # 5.1) Detect numbers and digits
        if with_numeric_input:
            numeric_mask_batch = torch.stack([torch.isin(q, self.numeric_vocab_token_ids)
                                              for q in batch['question_end_tokens']]).to(device)
            batch['numeric_mask'] = numeric_mask_batch

        if with_numeric_output:
            answer_numeric_mask_batch = torch.stack(
                [torch.isin(q, self.numeric_vocab_token_ids) for q in batch['answer_tokens']])
            batch['answer_numeric_mask'] = answer_numeric_mask_batch

        # 5.2) Embed text tokens and numerical tokens together and form a batch of embeddings
        if with_numeric_input:
            question_end_embeddings_batch = []
            for ind in range(batch_size):
                tokens_ = batch['question_end_tokens'][ind]
                numeric_mask_ = batch.get('numeric_mask')[ind]
                numeric_mask_ = numeric_mask_ if numeric_mask_ is not None else torch.zeros_like(tokens_,
                                                                                                 device=tokens_.device)

                # Embed with text embeddings
                tokens_text_embeddings_ = self.language_model_tokens_embedding_func(tokens_)

                # Select numeric tokens and their text embeddings
                numeric_tokens_ = tokens_[numeric_mask_]
                numeric_tokens_text_embeddings_ = tokens_text_embeddings_[numeric_mask_]

                # Create numeric embeddings
                if len(numeric_tokens_.size()) == 1:  # add dummy dim if there is a single number
                    numeric_tokens_.unsqueeze_(0)
                numeric_tokens_num_embeddings_ = self.get_numeric_embedding(numeric_tokens_)

                # Join together
                numeric_tokens_joined_embedding_ = torch.cat([numeric_tokens_text_embeddings_,
                                                              numeric_tokens_num_embeddings_ \
                                                             .repeat(numeric_tokens_text_embeddings_.size(0), 1)], 1)
                # Pass through projection layer
                numeric_tokens_joined_embedding_ = self.text_numeric_projection(
                    numeric_tokens_joined_embedding_).unsqueeze(0)

                # Replace in final embeddings sequence
                tokens_text_embeddings_[numeric_mask_] = numeric_tokens_joined_embedding_
                question_end_embeddings_batch.append(tokens_text_embeddings_.unsqueeze(0))

            # 5.3) Concatenate back into batch
            question_end_embeddings_batch = torch.cat(question_end_embeddings_batch, 0)
        else:
            # 5.2-3) otherwise just embed test tokens
            question_end_embeddings_batch = self.language_model_tokens_embedding_func(batch['question_end_tokens'])

        # 5.4) Create new attention mask
        question_end_attention_mask = (~mask_padding(batch['question_end_tokens'])).long()

        # 5.5) Fill injection ending tokens embeddings with trainable parameters
        if self.has_end_token(batch['question_end_tokens']):
            self.replace_end_token(batch['question_end_tokens'], question_end_embeddings_batch)

        # 6) LM input
        # 6.1) Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)
        if ('encoder_input_mask' in batch) \
                and (batch['encoder_input_mask'].size(1) == encoder_input.size(1)):
            encoder_input_mask = batch['encoder_input_mask']

        else:
            # Check if transactions history embedding size was reduced, then we cannot mask it
            if transactions_embeddings.size(1) == transactions_embeddings_mask.size(-1):
                encoder_input_mask = torch.cat(
                    [question_start_attention_mask,
                     batch['mask'],
                     question_end_attention_mask], dim=1
                )
            else:
                encoder_input_mask = torch.cat(
                    [question_start_attention_mask,
                     torch.ones(transactions_embeddings.size()[:2], dtype=batch['mask'].dtype, device=device),
                     question_end_attention_mask], dim=1
                )

        # 7) Labels
        # Create answers + masks for LM's decoder inputs
        # of size [bs, n_answer_tokens]
        batch_answers = batch['answer_tokens']
        batch_answers_mask = batch['answer_mask']

        if with_numeric_output:
            batch_answers_numeric_mask = batch['answer_numeric_mask']

            # Mask numeric tokens with -100 to not to use them in text CE
            batch_answers = batch['answer_tokens'].clone()
            batch_answers[batch_answers_numeric_mask] = -100

            # Embed text tokens and numerical tokens together and form a batch of embeddings
            numeric_answers_batch = []

            for ind in range(batch_size):
                tokens_ = batch['answer_tokens'][ind]
                numeric_mask_ = batch.get('answer_numeric_mask')[ind]
                numeric_mask_ = numeric_mask_ if numeric_mask_ is not None else torch.zeros_like(tokens_,
                                                                                                 device=tokens_.device)
                # Select numeric tokens and their text embeddings
                numeric_tokens_ = tokens_[numeric_mask_]
                # Create numeric repr: mantissa + exponent
                if len(numeric_tokens_.size()) == 1:  # add dummy dim if there is a single number
                    numeric_tokens_.unsqueeze_(0)

                numeric_tokens_repr_ = self.get_numeric_representation(numeric_tokens_, keep_size=True)
                numeric_answers_batch.append(numeric_tokens_repr_.unsqueeze(0))

            # of size [n_numeric_tokens_in_batch, 2]
            numeric_answers_batch = torch.cat(numeric_answers_batch, 1).squeeze(0)

        # 8) Forward without labels
        lm_outputs = self.language_model(
            inputs_embeds=encoder_input,
            labels=batch_answers,
            attention_mask=encoder_input_mask,
            decoder_attention_mask=batch_answers_mask,
            output_hidden_states=True)

        # Create answers + masks for LM's decoder inputs
        lm_outputs['answer_tokens'] = batch['answer_tokens']

        # 9) Calculate numeric loss (if needed)
        if with_numeric_output:
            lm_outputs['answer_numeric_mask'] = batch['answer_numeric_mask']
            lm_outputs['numeric_answers'] = numeric_answers_batch
            numeric_loss_output = self._compute_numeric_loss(lm_outputs,
                                                             output_predictions=True)

        # Re-scale losses
        total_loss = lm_outputs.loss * self._text_loss_scale
        if with_numeric_output:
            total_loss += numeric_loss_output.get('numeric_loss') * self.numeric_loss_scale

        # join two output dicts
        outputs = dict()
        outputs["logits"] = lm_outputs.logits
        outputs["loss"] = total_loss
        outputs["text_loss"] = lm_outputs.loss * self._text_loss_scale
        if with_numeric_output:
            numeric_loss = numeric_loss_output.pop('numeric_loss')
            outputs["numeric_loss"] = numeric_loss * self.numeric_loss_scale
            outputs["unscaled_numeric_loss"] = numeric_loss
            outputs["selector_loss"] = numeric_loss_output.get('selector_loss')
            outputs["exponent_loss"] = numeric_loss_output.get('exponent_loss')
            outputs["mantissa_loss"] = numeric_loss_output.get('mantissa_loss')

        if output_attentions:
            outputs["encoder_attentions"] = lm_outputs.encoder_attentions
            outputs["decoder_attentions"] = lm_outputs.decoder_attentions
        if output_hidden_states:
            outputs["hidden_states"] = lm_outputs.encoder_hidden_states

        question_start_tokens_batch = batch['question_start_tokens'].repeat(batch_size, 1)
        outputs['question_encoded'] = torch.cat([question_start_tokens_batch,
                                                 batch['question_end_tokens']], dim=1)

        if is_train:
            outputs['labels'] = batch_answers
            if with_numeric_output:
                outputs['token_type_predictions'] = numeric_loss_output.get('token_type_predictions')
                outputs['exponent_predictions'] = numeric_loss_output.get('exponent_predictions')
                outputs['mantissa_predictions'] = numeric_loss_output.get('mantissa_predictions')

                outputs['numeric_answers'] = get_number_from_parts(lm_outputs['numeric_answers'][:, 0],
                                                                   lm_outputs['numeric_answers'][:, 1])
                outputs['token_type_answers'] = numeric_loss_output.get('token_type_answers')
                outputs['exponent_answers'] = numeric_loss_output.get('exponent_answers')
                outputs['mantissa_answers'] = numeric_loss_output.get('mantissa_answers')

                # Decode floating numbers
                outputs['decoded_numeric_predictions'] = get_number_from_parts(outputs['mantissa_predictions'],
                                                                               outputs['exponent_predictions'])
        return outputs

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

    def collect_numeric_tokens(self) -> torch.LongTensor:
        """
        Collect from tokenizers vocab all numerical token ids.
        Returns:
            a tensor of numerical token ids.
        """
        vocab_tokens = list(self.tokenizer.vocab.keys())
        numeric_vocab_tokens = list(filter(lambda x: len(re.findall(r'\d+', x))
                                                     and ("extra_id" not in x), vocab_tokens))
        numeric_vocab_token_ids = torch.LongTensor([self.tokenizer.vocab.get(tok) for tok in numeric_vocab_tokens])
        self._logger.info(f"Collected {len(numeric_vocab_tokens)} numerical token ids from tokenizers vocab.")

        return numeric_vocab_token_ids

    def get_exponent_embedding(self, exp: torch.Tensor,
                               overflow_index: Optional[int] = None,
                               underflow_index: Optional[int] = None) -> torch.Tensor:
        # Select overflow / underflow indexes
        overflow_index = torch.LongTensor([self.exponent_embeddings.num_embeddings - 2]) \
            if overflow_index is None else overflow_index
        underflow_index = torch.LongTensor([self.exponent_embeddings.num_embeddings - 1]) \
            if underflow_index is None else underflow_index

        # Shift to positive indexes
        exp = exp + torch.abs(torch.LongTensor([self.exponent_available_range[0]]))

        min_inf_mask = exp < 0
        inf_mask = exp > self.exponent_embeddings.num_embeddings

        # Overflow detected
        if inf_mask.any():
            self._logger.warning(f"Numeric overflow detected!")
            exp[inf_mask] = overflow_index

        # Underflow detected
        if min_inf_mask.any():
            self._logger.warning(f"Numeric underflow detected!")
            exp[min_inf_mask] = underflow_index

        return self.exponent_embeddings(exp)

    def get_mantissa_embedding(self, manstissas: torch.Tensor,
                               prototypes: torch.Tensor) -> torch.Tensor:
        """
        Creates embeddings for mantissas based on provided prototypes.
        """
        batch_size = manstissas.size(0)
        num_mantissa_prototypes = prototypes.size(0)

        if len(manstissas.size()) == 1:
            manstissas = manstissas.unsqueeze(1).repeat(1, num_mantissa_prototypes)
        if len(prototypes.size()) == 1:
            prototypes = prototypes.unsqueeze(0).repeat(batch_size, 1)
        return torch.exp((manstissas - prototypes).pow(2).sqrt())

    def get_numeric_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Received `tokens` as [n_numbers, n_tokens_per_number] tensor transforms
        each provided number to single numeric embedding.
        """
        numeric_tokens = self.tokenizer.batch_decode(tokens)
        numeric_tokens = [convert_to_numeric(tok) for tok in numeric_tokens]
        # for those tokens that we couldn't convert to float use 0.0
        numeric_tokens = torch.Tensor([tok if tok is not None else 0.0 for tok in numeric_tokens])

        # Extract mantissa and exponent
        mantissas = get_mantissa_number(numeric_tokens)
        exponents = get_exponent_number(numeric_tokens)
        # Embed
        mantissa_embeddings = self.get_mantissa_embedding(mantissas, self.mantissa_prototypes)
        exponent_embeddings = self.get_exponent_embedding(exponents)
        numeric_embeddings = torch.cat([mantissa_embeddings, exponent_embeddings], 1)
        return numeric_embeddings

    def get_numeric_representation(self, tokens: torch.Tensor,
                                   keep_size: bool = False
                                   ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Received `tokens` as [n_numbers, n_tokens_per_number] tensor transforms
        each provided number to numeric representation: [mantissa, exponent].
        return: a tuple of tensor with mantissa parts and other with exponent parts.
        or
        return: a tensor with pair of mantissa and exponent parts for each token.
        """
        numeric_tokens = self.tokenizer.batch_decode(tokens)
        numeric_tokens = [convert_to_numeric(tok) for tok in numeric_tokens]
        # for those tokens that we couldn't convert to float use 0.0
        numeric_tokens = torch.Tensor([tok if tok is not None else 0.0 for tok in numeric_tokens])

        # Extract mantissa and exponent
        mantissas = get_mantissa_number(numeric_tokens)
        exponents = get_exponent_number(numeric_tokens)
        if not keep_size:
            return mantissas, exponents
        # output for each token a duplicated pair of (mantissa, exp)
        else:
            keep_size_outputs = []
            for num_i in range(tokens.size(0)):
                m, e = mantissas[num_i], exponents[num_i]
                me = torch.Tensor([m, e]).unsqueeze(0).repeat(tokens[num_i].size(0), 1)
                keep_size_outputs.append(me)
            return torch.cat(keep_size_outputs, 0)

    def _compute_numeric_loss(self, outputs: Dict[str, torch.Tensor],
                              output_predictions: Optional[bool] = True) -> Dict[str, torch.Tensor]:
        # Use last hidden states to extract
        last_hidden_state = outputs['decoder_hidden_states'][-1]
        hidden_dim = last_hidden_state.size(-1)

        # 1) Token type selector Loss
        # Selector head: 1 - numeric token, 0 - text token
        selector_targets = outputs['answer_numeric_mask'].flatten().long()

        # flatten [bs, num_answer_tokens, hidd_dim] -> [bs * num_answer_tokens, hidd_dim]
        # no softmax as torch.CE computes the cross entropy loss between input logits and target.
        # logits != probabilities!!! (- they are unscaled)
        selector_logits = self.token_type_classifier(last_hidden_state.reshape(-1,
                                                                               hidden_dim))
        # Get predictions - need for inference
        numeric_answer_tokens_pred = selector_logits.argmax(-1)
        selector_loss = self.ce_loss_fn(selector_logits, selector_targets)

        # 2) Numeric tokens loss
        # On validation take predicted: numeric_answer_tokens_pred.bool()
        numeric_tokens_hidden_states = last_hidden_state.reshape(-1, hidden_dim)[
            outputs['answer_numeric_mask'].view(-1)]

        # Answers as pairs of [mantissa, exponent]
        numeric_answers_mantissa = outputs['numeric_answers'][:, 0]
        numeric_answers_exponent = outputs['numeric_answers'][:, 1].long()

        # 2.1) Exponent
        exponent_logits = self.exponent_head(numeric_tokens_hidden_states)
        # of size [n_numeric_tokens_in_batch,]
        exponent_preds = torch.nn.functional.softmax(exponent_logits, -1).argmax(-1) \
                         - torch.abs(torch.LongTensor([self.exponent_available_range[0]]))
        exponent_loss = self.ce_loss_fn(exponent_logits, numeric_answers_exponent +
                                        torch.abs(torch.LongTensor([self.exponent_available_range[0]])))
        # 2.2) Mantissa
        mantissa_preds = self.mantissa_head(numeric_tokens_hidden_states)

        # if matissa head outputs a single logit -> (hidden_size, 1)
        # mantissa_loss = torch.mean(
        #     (1 / 3.14) * torch.exp(-(numeric_answers_mantissa - mantissa_preds.squeeze()).pow(2)))
        mantissa_loss = self.l1_loss(mantissa_preds.squeeze(), numeric_answers_mantissa)

        # 3) Total numerical loss
        numeric_loss = selector_loss + exponent_loss + mantissa_loss

        loss_outputs = dict(numeric_loss=numeric_loss,
                            selector_loss=selector_loss,
                            exponent_loss=exponent_loss,
                            mantissa_loss=mantissa_loss)
        if output_predictions:
            loss_outputs['token_type_predictions'] = numeric_answer_tokens_pred.detach()
            loss_outputs['exponent_predictions'] = exponent_preds.detach()
            loss_outputs['mantissa_predictions'] = mantissa_preds.squeeze().detach()

            loss_outputs['token_type_answers'] = selector_targets.detach()
            loss_outputs['exponent_answers'] = (numeric_answers_exponent +
                                                torch.abs(torch.LongTensor(
                                                    [self.exponent_available_range[0]]
                                                ))).detach()
            loss_outputs['mantissa_answers'] = numeric_answers_mantissa.detach()

        return loss_outputs
