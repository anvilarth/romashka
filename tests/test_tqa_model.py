import torch
import unittest
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from romashka.transactions_qa.tqa_model import TransactionQAModel
from romashka.transactions_qa.utils import get_projections_maps
from romashka.transactions_qa.dataset.data_generator import cat_features_names, meta_features_names, num_features_names
from romashka.models import TransactionsModel
from romashka.transactions_qa.tasks import AutoTask

class TestTQAModel(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self.test_batch = torch.load("romashka/assets/test_batch.pt")
        self.device = torch.device('cpu')
        self.model_name = 'google/flan-t5-small'

        projections_maps = get_projections_maps(relative_folder=self._current_dir / 'romashka')
        transactions_model_config = {
            "cat_features": cat_features_names,
            "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
            "num_features": num_features_names,
            "num_embedding_projections": projections_maps.get('num_embedding_projections'),
            "meta_features": meta_features_names,
            "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
            "encoder_type":'whisper/tiny',
            "head_type": 'next',
            "embedding_dropout": 0.1
        }
        transactions_model = TransactionsModel(**transactions_model_config)

        # Load weights
        ckpt = torch.load('checkpoints/transactions_model/final_model.ckpt', map_location='cpu')
        transactions_model.load_state_dict(ckpt)
        transactions_model.to(self.device)


        # Download vocabulary from huggingface.co and define model-specific arguments
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Download model from huggingface.co and cache.
        lm_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
        )

        # Create tasks
        tasks = []
        task_names = ['default']

        for task_i, task_name in enumerate(task_names):
            task = AutoTask.get(task_name=task_name, tokenizer=tokenizer)
            tasks.append(task)

        # Create general Tranactions QA model
        transactionsQA_model_config = {
            "warmup_steps": 0,
            "training_steps": 0,
            "do_freeze_tm": False,
            "do_freeze_lm": False,
            "do_freeze_connector": False,
            "connector_input_size": 384,
        }
        self.model = TransactionQAModel(
            language_model=lm_model,
            transaction_model=transactions_model,
            tokenizer=tokenizer,
            tasks=tasks,
            **transactionsQA_model_config
        )

    def test_model(self):
       self.model.model_step(self.test_batch) 

if __name__ == '__main__':
    unittest.main()