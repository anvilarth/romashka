import torch
import unittest
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from romashka.transactions_qa.tasks import MostFrequentMCCCodeTask


class TestMostFrequentMCCCodeTask(unittest.TestCase):

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
        task = MostFrequentMCCCodeTask(
            tokenizer=self.tokenizer
        )
        print(f"Init vocab size: {len(self.tokenizer)}")
        self.lm_model.resize_token_embeddings(len(self.tokenizer))
        print(f"Extended vocab size: {len(self.tokenizer)}")
        print(task)

    def test_task_on_batch(self):
        task = MostFrequentMCCCodeTask(task_name="most_frequent_mcc_code",
                                       target_feature_name="mcc",
                                       tokenizer=self.tokenizer)
        self.lm_model.resize_token_embeddings(len(self.tokenizer))
        print(task)

        proc_batch = task.process_input_batch(self.test_batch)
        for k, v in proc_batch.items():
            if isinstance(v, torch.Tensor):
                print(f"\t{k}: {v.size()}")
            else:
                for vi, vv in enumerate(v):
                    print(f"\t{k}/{vi}: {vv.size()}")


if __name__ == '__main__':
    unittest.main()
