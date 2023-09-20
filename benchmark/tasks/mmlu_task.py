import json
import random
import datasets
from typing import (Dict, Any, Optional)

from romashka.logging_handler import get_logger
from romashka.benchmark.tasks.text_task_abstract import AbstractTextTask
from romashka.benchmark.tasks.mappings import DATA_PATHS_MAPPING, PROMT_PATH


class MMLUTaskDataset(AbstractTextTask):

    def __post_init__(self):
        super().__post_init__()
        self.name = "" if self.name is None else self.name
        self.split_to_data_split = {"validation": "test", "test": "test"}

        self.data_path = self.data_path if self.data_path is not None \
            else f"{DATA_PATHS_MAPPING['MMLU']}/"

        self.logger = get_logger(
            name=self.__class__.__name__,
            logging_level="DEBUG" if self.verbose else "INFO"
        )

    def load_dataset_local(self, extension: Optional[str] = "json",
                           data_files: Optional[Dict[str, str]] = None,
                           split: str = None, **kwargs):
        if data_files is None:
            task = f"{self.name}.{extension}"
            data_files = {
                'test': f'{self.data_path}{task}'
            }
        if split is None:
            split = 'test'
        return super().load_dataset_local(path=extension,
                                          data_files=data_files,
                                          split=split)

    def preprocessor(self, example: Dict[str, Any], add_prefix: bool = False, **kwargs):
        src_texts = [example["source"]]
        tgt_texts = [example["target"]]
        return self.format_as_seq2seq(src_texts, tgt_texts)
