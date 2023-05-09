from typing import (List, Optional,
                    Tuple, Any,
                    Dict, Union)

import torch
import torch.nn as nn
import transformers

from romashka.logging_handler import get_logger
from romashka.transactions_qa.utils import seed_everything
from romashka.transactions_qa.layers.connector import (make_linear_connector,
                                                       make_recurrent_connector)
from romashka.transactions_qa.model.encoder_model import EncoderSimpleModel
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.utils import (mask_padding, mask_lm_labels_padding)


class EncoderFrozenModel(EncoderSimpleModel):
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
                         do_freeze_lm_embeddings=do_freeze_lm_embeddings,
                         do_freeze_connector=do_freeze_connector,
                         generation_config=generation_config,
                         is_debug=is_debug)

    def _prepare_model(self):
        super()._prepare_model()
        self._create_trainable_parameters()

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

    def forward(self, batch: Union[Dict[str, torch.Tensor], Any],
                output_attentions: Optional[bool] = False,
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
        # 1) Get transactions embeddings for initial batch
        # transactions model requires: ['mask', 'cat_features', 'num_features', 'meta_features']
        # + optionally: 'time' - ??? maybe 'event_time' ???
        # return: Tuple[
        # torch.Tensor, - embeddings -> we need this
        # torch.Tensor - mask
        # ]
        transaction_mask = batch['mask']
        batch_size = transaction_mask.size(0)

        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # 2) Next pass them to connector == linear mapping -> to LM inner dim
        # Checks whether a connector requires mask argument
        if self.inspect_forward_signature("mask", self.connector):
            transactions_embeddings = self.connector(transactions_embeddings,
                                                     mask=transactions_embeddings_mask)
        else:
            transactions_embeddings = self.connector(transactions_embeddings)

        # 3) Questions: to embedding of LM
        # torch.Size([1, len(question_start_tokens))
        question_start_embeddings = self.language_model_tokens_embedding_func(
            batch['question_start_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_start_attention_mask = batch['question_start_tokens_mask']

        # if it already ends with [trx]
        if self.has_start_token(batch['question_start_tokens']):
            self.replace_start_token(batch['question_start_tokens'], question_start_embeddings)
        # otherwise append it to the end of starting sequence
        else:
            question_start_embeddings = torch.cat([question_start_embeddings,
                                                   self.transactions_start_embedding[None, None]], dim=1)
            # add one more sample to attentions also
            question_start_attention_mask = torch.cat([
                torch.ones((1,)).long().repeat(batch_size, 1).to(batch['question_start_tokens_mask'].device),
                batch['question_start_tokens_mask']
            ], dim=1)

        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        # question_end_tokens: torch.Size([batch_size, len(max_question_end_tokens)])
        question_end_embeddings_batch = self.language_model_tokens_embedding_func(
            batch['question_end_tokens'])  # call for (embed_tokens): Embedding(vocab_size, model_hidden_dim)
        question_end_attention_mask = batch['question_end_attention_mask']

        # Fill injection ending tokens embeddings with trainable parameters
        # if it already starts with [/trx]
        if self.has_end_token(batch['question_end_tokens']):
            self.replace_end_token(batch['question_end_tokens'], question_end_embeddings_batch)
        # otherwise prepend it to the start of question sequence
        else:
            # todo: add trns ending token to the end before paddings!!!
            question_end_embeddings_batch = torch.cat([
                self.transactions_end_embedding[None, None].repeat(batch_size, 1, 1),
                question_end_embeddings_batch], dim=1)
            # add one more sample to attentions also
            question_end_attention_mask = torch.cat([
                batch['question_end_attention_mask'],
                torch.ones((1,)).long().repeat(batch_size, 1).to(batch['question_end_attention_mask'].device)
            ], dim=1)

        # 4) Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)
        if ('encoder_input_mask' in batch) \
                and (batch['encoder_input_mask'].size(1) == encoder_input.size(1)):
            encoder_input_mask = batch['encoder_input_mask']

        else:
            encoder_input_mask = torch.cat(
                [question_start_attention_mask,
                 batch['mask'],
                 question_end_attention_mask], dim=1
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
                                         output_attentions=output_attentions,
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

    def generate(self,
                 questions: Union[str, List[str], torch.Tensor],
                 transactions_batch: Dict[str, torch.Tensor],
                 prefix_prompt: Optional[Union[str, torch.Tensor]] = "",
                 answer_template: Optional[str] = "",
                 max_new_tokens: Optional[int] = None,
                 min_new_tokens: Optional[int] = None,
                 top_p: Optional[float] = 1.0,
                 temperature: Optional[float] = 0.0,
                 suggestions: Optional[int] = 1,
                 diversity_penalty: Optional[float] = 0.0,
                 filter_value: Optional[float] = -float('Inf'),
                 allowed_token_ids: Optional[List[int]] = None,
                 hidden_dims_indexes: Optional[List[int]] = None,
                 stopping_criteria: Optional[Any] = None,
                 seed: Optional[int] = 11):
        """
        Generates answers for questions.
        Args:
            questions: Union[str, List[str], torch.Tensor] - a question(-s) to answer. Can be passed as:
                str: a single question in string representation;
                List[str]: a list of questions in string representation;
                torch.Tensor: a single tokenized question or multiple questions;
            transactions_batch: Dict[str, torch.Tensor] - a batch for transactions model;
            prefix_prompt: Union[str, torch.Tensor] - a prefix for transactions embeddings. Can be passed as:
                str: a prefix in string representation;
                torch.Tensor: a tokenized prefix;
            answer_template: str - an answer template prefix to add to the question ending;
            max_new_tokens: int - the maximum number of tokens to generate,
                                    ignoring the number of tokens in the question;
            min_new_tokens: int - the minimum number of tokens to generate,
                                    ignoring the number of tokens in the question;
            top_p: float - If set to float < 1, only the most probable tokens with probabilities
                            that add up to top_p or higher are kept for generation;
            temperature: float - The value used to module the next token probabilities;
            suggestions: TBD
            diversity_penalty: TBD
            filter_value: float - a value to assign to tokens that should never be generated;
            allowed_token_ids: List[int] - a list of token ids that must be generated;
            hidden_dims_indexes: List[int] - a list of hidden layers' indexes from
                                        which to take hidden states for embedding creation.
                                        Default set to -1 - so only last layer's hidden states would be used;
            stopping_criteria: a class instance / callable that can be used to change
                                when to stop generation (other than EOS token).
                                It should return a boolean flag when all batch samples are successfully finished;
            seed: int - a seed for generation;

        Returns:
            A dict with keys:
             - generated_texts - a list of generated text tokens sequences for each batch sample;
             - output_embeddings - a list of embeddings for sequences, collected from selected hidden layers' states;
             - output_logits - a list of logits generated for each batch sample on each step.
        """
        seed_everything(seed)
        self.eval()  # freeze all at once

        device = self.language_model.device
        if device.type != 'cpu':
            torch.cuda.empty_cache()

        # Transactions
        transactions_history_embeddings, _ = self.transaction_model.get_embs(transactions_batch)

        # next pass them to connector == linear mapping -> to LM inner dim
        transactions_history_embeddings = self.connector(transactions_history_embeddings)

        vocab_size = self.language_model.vocab_size
        batch_size = transactions_history_embeddings.size(0)
        transactions_seq_len = transactions_history_embeddings.size(1)

        # Fill empty parameters
        hidden_dims_indexes = hidden_dims_indexes if hidden_dims_indexes is not None else [-1]

        # Encode everything

        # Starting prompts
        # In case single question in string form
        if isinstance(prefix_prompt, str):
            prefix_prompt_tokens = self.tokenizer.encode(prefix_prompt,
                                                         add_special_tokens=False,
                                                         return_tensors='pt').long().to(device)
        elif isinstance(prefix_prompt, torch.Tensor):
            prefix_prompt_tokens = prefix_prompt.long().to(device)
        else:
            raise AttributeError(f"Unable to use prefix prompt in provided form: {type(prefix_prompt)}!")

        prefix_prompt_embeddings = self.language_model_tokens_embedding_func(prefix_prompt_tokens)

        # if it already ends with [trx]
        if self.has_start_token(prefix_prompt_tokens):
            self.replace_start_token(prefix_prompt_tokens, prefix_prompt_embeddings)
        # otherwise append it to the end of starting sequence
        else:
            prefix_prompt_embeddings = torch.cat([prefix_prompt_embeddings,
                                                  self.transactions_start_embedding[None, None]], dim=1)

        prefix_prompt_embeddings_batch = prefix_prompt_embeddings.repeat(batch_size, 1, 1)

        # Question
        # In case single question in string form
        if isinstance(questions, str):
            question_tokens = self.tokenizer.encode(questions,
                                                    add_special_tokens=False,
                                                    return_tensors='pt').long().to(device)
            question_embeddings = self.language_model_tokens_embedding_func(question_tokens)
            question_embeddings_batch = question_embeddings.repeat(batch_size, 1, 1)

        # In case questions in tokenized form of tensors
        elif isinstance(questions, torch.Tensor):
            question_embeddings = self.language_model_tokens_embedding_func(questions)
            if question_embeddings.size(0) != batch_size:
                # If it was a single question
                question_embeddings_batch = question_embeddings.repeat(batch_size, 1, 1)
            else:
                question_embeddings_batch = question_embeddings

        # In case a list of dtring questions provided
        elif isinstance(questions, list) and isinstance(questions[0], str):
            question_tokens = self.tokenizer.encode(questions,
                                                    padding=True,
                                                    add_special_tokens=False,
                                                    return_tensors='pt').long().to(device)
            question_embeddings = self.language_model_tokens_embedding_func(question_tokens)
            question_embeddings_batch = question_embeddings
        else:
            raise AttributeError(f"Unable to use questions in provided form: {type(questions)}!")

        # Fill questions transactions' injection end token with trainable parameters
        # if it already starts with [/trx]
        if self.has_end_token(question_tokens):
            self.replace_end_token(question_tokens, question_embeddings_batch)

        # otherwise prepend it to the start of question sequence
        else:
            question_embeddings_batch = torch.cat([
                self.transactions_end_embedding[None, None].repeat(batch_size, 1, 1),
                question_embeddings_batch], dim=1)

        # Answer template --> embeddings
        answer_template_tokens = self.tokenizer.encode(answer_template,
                                                       add_special_tokens=False,
                                                       return_tensors='pt').long().to(device)
        # If empty template (to prevent errors in embeddings)
        if not answer_template_tokens.size(1):
            answer_template_tokens = self.whitespace_token_id.to(device)

        answer_template_embeddings = self.language_model_tokens_embedding_func(answer_template_tokens)
        answer_template_embeddings_batch = answer_template_embeddings.repeat(batch_size, 1, 1)

        # Concat all together
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        input_embedds = torch.cat([prefix_prompt_embeddings_batch,
                                   transactions_history_embeddings,
                                   question_embeddings_batch,
                                   answer_template_embeddings_batch], dim=1).to(device)

        embeddings = input_embedds.clone().to(device)
        output_embeddings = []
        output_logits = []
        out = None

        with torch.no_grad():  # no tracking history
            # Generate max number of tokens if stopping criterion is not triggered
            i = 0
            for _ in range(max_new_tokens):
                i += 1
                output = self.language_model(inputs_embeds=embeddings,
                                             decoder_inputs_embeds=embeddings,
                                             output_attentions=output_attentions,
                                             output_hidden_states=True)

                # Collect and sum the hidden states.
                hidden_states = []
                for idx in hidden_dims_indexes:
                    hidden_states.append(output.decoder_hidden_states[idx])
                # Add hidden states together.
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, T, 256)
                last_embedding = last_hidden_state / last_hidden_state.norm(dim=-1, keepdim=True)
                output_embeddings.append(last_embedding)

                logits = output.logits[:, -1, :]  # take only last token logits (N, vocab_size)

                # If we need to restrict model to predict only some tokens
                if allowed_token_ids is not None:
                    logits = logits[:, allowed_token_ids]

                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)

                past_key_values = output.past_key_values
                # get next token
                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / temperature

                    # Apply top-p filtering.
                    if top_p < 1.0:
                        assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1),
                                                        dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value

                    token_weights = logits.exp()  # (N, vocab_size)
                    next_token = torch.multinomial(token_weights, 1)  # (N, 1)

                # Concat with previous embeddings
                next_token = next_token.long().to(device)

                if allowed_token_ids is not None:
                    next_token = torch.tensor(allowed_token_ids).to(device)[next_token]
                out_of_vocab_indexes = torch.where(next_token > vocab_size)
                next_token[out_of_vocab_indexes] = self.tokenizer.eos_token_id \
                    if hasattr(self.tokenizer, "eos_token_id") else 0

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                self._logger.info(f"Output decoded: {[self.tokenizer.decode(token) for token in out]}")
                next_embedding = self.language_model_tokens_embedding_func(next_token)
                embeddings = torch.cat([embeddings, next_embedding], dim=1)

                # If stopping criteria triggered for all samples in batch
                # and all samples in batch reached min number of new tokens
                if (stopping_criteria is not None) \
                        and stopping_criteria(out) \
                        and (i >= min_new_tokens):
                    self._logger.warning(f"Stopping criteria triggered!")
                    break

        return dict(generated_texts=out,
                    output_embeddings=output_embeddings,
                    output_logits=output_logits)