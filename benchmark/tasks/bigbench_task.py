import json
import random
import datasets
from typing import (Dict, Any, Optional)

from romashka.logging_handler import get_logger
from romashka.benchmark.tasks.text_task_abstract import AbstractTextTask
from romashka.benchmark.tasks.mappings import DATA_PATHS_MAPPING, PROMT_PATH


class BigBenchTaskDataset(AbstractTextTask):

    def __post_init__(self):
        super().__post_init__()
        self.name = "" if self.name is None else self.name
        self.split_to_data_split = {"train": "train"}

        self.data_path = self.data_path if self.data_path is not None \
            else f"{DATA_PATHS_MAPPING['BigBench']}/bigbench_{self.name}/"

        self.logger = get_logger(
            name=self.__class__.__name__,
            logging_level="DEBUG" if self.verbose else "INFO"
        )

    def load_dataset_local(self, extension: Optional[str] = "json",
                             data_files: Optional[Dict[str, str]] = None,
                             split: str = None, **kwargs):
        try:
            return super().load_dataset_local(path=self.data_path,
                                              data_files=data_files,
                                              split=split
                                              )
        except Exception as e:
            self.logger.error(f"Error during AbstractTaskDataset.load_dataset() method occurred: {e}. "
                              f"Loading dataset with datasets.load_from_disk() method...")
            return datasets.load_from_disk(self.data_path)

    def preprocessor(self, example: Dict[str, Any], add_prefix: bool = False, **kwargs):
        prompts = json.load(open(PROMT_PATH))
        prompt = random.choice(prompts[self.name])

        if len(example['multiple_choice_targets']) > 0:
            options = 'OPTIONS: - ' + '; - '.join(example['multiple_choice_targets'])
        else:
            options = ''
        if self.name in ['conceptual_combinations', 'language_identification']:
            options = ''
        if 'russian' in self.name:
            options = 'ВАРИАНТЫ: - ' + '; - '.join(example['multiple_choice_targets'])

        src_texts = [prompt, example['inputs'], options] if options != "" else [prompt, example['inputs']]
        tgt_texts = ['; '.join(example['targets'])]

        return self.format_as_seq2seq(src_texts, tgt_texts)