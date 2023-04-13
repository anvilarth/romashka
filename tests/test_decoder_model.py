import torch
import unittest
from pathlib import Path
from transformers import OPTForCausalLM, AutoTokenizer, AutoConfig

from romashka.models import TransactionsModel
from romashka.transactions_qa.model.decoder_model import DecoderSimpleModel
from romashka.transactions_qa.layers.connector import (make_linear_connector,
                                                       make_recurrent_connector)

from romashka.transactions_qa.utils import get_projections_maps
from romashka.transactions_qa.dataset.data_generator import (cat_features_names,
                                                             meta_features_names,
                                                             num_features_names)
from romashka.transactions_qa.model.generation_utils import AnsweredQACriteria
from romashka.transactions_qa.tasks import AutoTask


class TestDecoderTQAModel(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._prepare_for_tests()

    def _prepare_for_tests(self):
        """
        Prepare weight & data & models for testing.
        Returns:

        """
        self._current_dir = Path().resolve()
        save_batch_fn = Path("../assets/test_batch_v1.pt")
        if not save_batch_fn.exists():
            print(f"Test batch of data was not found by path: {save_batch_fn}")
            raise FileNotFoundError(f"Test batch of data was not found by path: {save_batch_fn}")
        self.test_batch = torch.load(str(save_batch_fn))

        self.device = torch.device('cpu')
        self.model_name = "facebook/galactica-125m"
        use_fast_tokenizer = True

        # Configure and load from HF hub LM model
        print(f"Loading Language model: `{self.model_name}`...")
        config_kwargs = {
            "use_auth_token": None,
            "return_unused_kwargs": True
        }

        tokenizer_kwargs = {
            "use_fast": use_fast_tokenizer,
            "do_lowercase": False,
            "pad_token": '<pad>',
            "pad_token_id": 1,
            "eos_token": "</s>",
            "eos_token_id": 2,
            "bos_token": "<s>",
            "bos_token_id": 0,
            "unk_token": "<unk>",
            "unk_token_id": 3
        }

        config, unused_kwargs = AutoConfig.from_pretrained(
            self.model_name, **config_kwargs
        )
        # Download vocabulary from huggingface.co and define model-specific arguments
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, config=config, **tokenizer_kwargs)

        # Download model from huggingface.co and cache.
        self.lm_model = OPTForCausalLM.from_pretrained(
            self.model_name,
            config=config
        )

        projections_maps = get_projections_maps(relative_folder="..")
        transactions_model_encoder_type = "whisper/tiny"
        transactions_model_head_type = "next"

        transactions_model_config = {
            "cat_features": cat_features_names,
            "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
            "num_features": num_features_names,
            "num_embedding_projections": projections_maps.get('num_embedding_projections'),
            "meta_features": meta_features_names,
            "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
            "encoder_type": transactions_model_encoder_type,
            "head_type": transactions_model_head_type,
            "embedding_dropout": 0.1
        }
        self.transactions_model = TransactionsModel(**transactions_model_config)

        self.connector = make_linear_connector(
            output_size=384,
            input_size=self.lm_model.config.hidden_size
        )

        # Create tasks
        self.tasks = []
        task_names = ["most_frequent_mcc_code_binary"]
        task_kwargs = [{}]  # ground truth + 5 additional options

        if isinstance(task_names, str):
            task_names = eval(task_names)
        task_kwargs = task_kwargs
        if isinstance(task_kwargs, str):
            task_kwargs = eval(task_kwargs)
        print(f"Got task_names: {task_names} with task_kwargs: {task_kwargs}")

        for task_i, task_name in enumerate(task_names):
            task_kwargs = task_kwargs[task_i] if task_i < len(task_kwargs) else {}
            if "tokenizer" not in task_kwargs:
                task_kwargs['tokenizer'] = self.tokenizer
            task = AutoTask.get(task_name=task_name, **task_kwargs)
            self.tasks.append(task)
        print(f"Created {len(self.tasks)} tasks.")

        # Explicitly resize embeddings
        init_embeddings = self.lm_model.get_input_embeddings()
        print(f"Language model initial `num_embeddings`: {init_embeddings.num_embeddings}, "
              f"`embedding_dim`: {init_embeddings.embedding_dim}")

        self.lm_model.resize_token_embeddings(len(self.tokenizer))

        resized_embedds = self.lm_model.model.decoder.get_input_embeddings()
        print(f"Language model resized `num_embeddings`: {resized_embedds.num_embeddings}, "
              f"`embedding_dim`: {resized_embedds.embedding_dim}")

        self.decoder_model = DecoderSimpleModel(
            language_model=self.lm_model,
            transaction_model=self.transactions_model,
            tokenizer=self.tokenizer,
            connector=self.connector,
            is_debug=True
        )

    def test_generate(self):

        max_new_tokens = 3
        min_new_tokens = 1
        top_k = 50
        top_p = 1.0
        temperature = 0.0  # 0.0 - greedy decoding
        suggestions = 1
        diversity_penalty = 0.0

        task = self.tasks[0]
        print(f"Task: {task.task_name}")
        if 'binary' in task.task_name:
            answer_options = list(task.binary_answer_options.values())
        else:
            answer_options = list(task.answers_options)
        print(f"Answer options: {answer_options}\n")
        force_words_ids = self.decoder_model.tokenizer(answer_options, add_special_tokens=False).input_ids

        qa_batch = task.process_input_batch(self.test_batch)

        # Which hidden dims to take
        hidden_dims_indexes = [-1]
        # Value to assign to tokens that should never be generated.
        filter_value = -float('Inf')

        stopping_criteria = AnsweredQACriteria(prompt_length=0, answer_tokens_ids=[i[0] for i in force_words_ids])

        predictions = self.decoder_model.generate(
            questions=qa_batch['question_end_tokens'],
            transactions_batch=self.test_batch,
            prefix_prompt=qa_batch['question_start_tokens'],
            answer_template=task.answer_template,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            top_p=top_p,
            temperature=temperature,
            filter_value=filter_value,
            allowed_token_ids=[i[0] for i in force_words_ids],
            hidden_dims_indexes=hidden_dims_indexes,
            stopping_criteria=stopping_criteria,
            seed=11
        )
        print(f"Predicted:\n{predictions['generated_texts']}")


if __name__ == '__main__':
    unittest.main()
