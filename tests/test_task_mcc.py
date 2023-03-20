import torch
import unittest
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from romashka.transactions_qa.tasks import MostFrequentMCCCodeTask


class TestMostFrequentMCCCodeTask(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self.test_batch = torch.load("C:/Users/airen/Documents/Projects/TransactionsQA/notebooks/example_batch.pt")
        self.device = torch.device('cpu')
        self.model_name = 'google/flan-t5-small'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.lm_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def test_task_creation(self):
        task = MostFrequentMCCCodeTask(task_name="most_frequent_mcc_code",
                                       target_feature_name="mcc",
                                       tokenizer=self.tokenizer)
        self.lm_model.resize_token_embeddings(len(self.tokenizer))
        print(task)

    def test_task_on_batch(self):
        task = MostFrequentMCCCodeTask(task_name="most_frequent_mcc_code",
                                       target_feature_name="mcc",
                                       tokenizer=self.tokenizer)
        print(task)

        task.process_input_batch()


if __name__ == '__main__':
    unittest.main()
