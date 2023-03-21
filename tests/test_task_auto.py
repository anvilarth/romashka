import torch
import unittest
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from romashka.transactions_qa.tasks import AutoTask, AUTO_TASKS


class TestAutoTask(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self.test_batch = torch.load("../../notebooks/example_batch.pt")
        self.device = torch.device('cpu')
        self.model_name = 'google/flan-t5-small'
        # for flan-t5-small:
        #   config.json: "vocab_size": 32128, "d_model": 512
        #   tokenizer.json: 0 = <pad>, 1 = </s> ("eos_token"), 2 = <unk>
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.lm_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def test_task_creation(self):
        task_kwargs = {}
        task_kwargs['tokenizer'] = self.tokenizer
        task_name = "most_frequent_mcc_code"
        task = AutoTask.get(
            task_name=task_name,
            **task_kwargs
        )
        print(task)


if __name__ == '__main__':
    unittest.main()
