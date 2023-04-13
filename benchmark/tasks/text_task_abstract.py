import os
import re
import sys

import torch
import functools
import pandas as pd
from abc import ABC, abstractmethod

# DTO
import dataclasses
from dataclasses import dataclass
from typing import (Callable, Dict, Mapping, List,
                    Any, Optional, Union)
import datasets
from datasets import load_from_disk

from romashka.logging_handler import get_logger


@dataclass
class AbstractTextTask(ABC):
    """
    Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    preprocessor: a processor to convert the given dataset to the sequence
        to sequence format.
    metrics: specifies the metrics to evaluate the task based on them.
    split_to_data_split: since not all the time, different splits of the
        datasets are available, we define a mapping from the wanted split
        to the existing dataset splits.
    """
    name: str
    data_path: Optional[str] = None
    task_specific_config: Optional[Dict[str, Any]] = None
    metrics: Optional[List[Callable]] = None

    # the same as: {'train': 'train', 'test': 'test', 'validation': 'validation'}
    split_to_data_split: Optional[Mapping[str, str]] = dataclasses.field(default_factory=dict)

    seed: Optional[int] = 11
    verbose: Optional[bool] = False

    def __post_init__(self):
        self.logger = get_logger(
            name=self.__class__.__name__,
            logging_level="DEBUG" if self.verbose else "INFO"
        )
        self.split_to_data_split = {"train": "train", "validation": "validation", "test": "test"} if not len(
            self.split_to_data_split) else self.split_to_data_split

    def load_dataset_local(self, path: str,
                           data_files: Optional[Dict[str, str]],
                           split: Optional[str] = None,
                           **kwargs) -> datasets.DatasetDict:
        """
        Function to load the dataset from local / remote.
        :param path: Path or name of the dataset.
        For local datasets:
            - if path is a local directory (containing data files only) -> load a generic dataset builder
                (csv, json, text etc.) based on the content of the directory e.g. './path/to/directory/with/my/csv/data'.
            - if path is a local dataset script or a directory containing a local dataset script
                (if the script has the same name as the directory) -> load the dataset builder
                from the dataset script e.g. './dataset/squad' or './dataset/squad/squad.py'.
        :type path: str;
        :param data_files: a mapping of data files;
        :type data_files: Dict[str, str];
        :param split: a name of the laoding split;
        :type split: str;
        :return:
        :rtype: DatasetDict instance.
        """
        self.logger.info(f"Loading dataset '{self.name}'...")
        data = datasets.load_dataset(path, data_files=data_files, split=split, **kwargs)
        self.logger.info("Dataset loaded.")
        return data

    def load_dataset_from_hub(self, split: str = None):
        """
        Function to load the dataset from Hugging Face Hub.
        :param split:  Which split of the data to load.
                        If None, will return a dict with all splits
                        (typically datasets.Split.TRAIN and datasets.Split.TEST).
                        If given, will return a single Dataset.
        :type split: str;
        :return: loaded dataset;
        :rtype: Dataset or DatasetDict.
        """
        return datasets.load_dataset(self.name, split=split)

    def get_train_val_split_indices(self, split: str,
                                    validation_size: Optional[Union[int, float]] = 1000) -> List[int]:
        """
        Load dataset within selected split and then split it again for smaller train/validation parts.
        :param split:  Which split of the data to load.
                        If None, will return a dict with all splits
                        (typically datasets.Split.TRAIN and datasets.Split.TEST).
                        If given, will return a single Dataset.
        :type split: str;
        :param validation_size: a size of validation dataset.
                                If integer passed,then it indicated a number of validation samples to select
                                from train data. Otherwise, if `validation_size` is in range of [0, 1),
                                then it shows which part of the entire data set to take for validation;
        :type validation_size: int;
        :return: indices of selected part of split;
        :rtype: List[int].
        """
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]

        self.logger.info(f"Loading dataset '{self.name}/{split}'...")
        dataset = self.load_dataset_from_hub(mapped_split)
        self.logger.info(f"Dataset loaded.")

        dataset_size = len(dataset)
        indices = torch.randperm(dataset_size, generator=generator).tolist()

        # If `validation_size` is a ratio of data samples
        if isinstance(validation_size, float) and (validation_size < 1) and (validation_size >= 0):
            validation_size = int(dataset_size * validation_size)
        # Take all dataset
        elif validation_size == -1:
            validation_size = dataset_size

        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def select_dataset_samples(self, indices: List[int], dataset: datasets.Dataset,
                               requested_n: int = None) -> datasets.Dataset:
        """
        Given a dataset for the split, obtains the sample indices for this split
        and returns the subsampled dataset.
        :param indices: indexes to obtain fron dataset;
        :type indices: List[int],
        :param dataset: dataset to subsample;
        :type dataset: datasets.Dataset;
        :param requested_n: requested number of samples;
        :type requested_n: int;
        :return: subsampled dataset;
        :rtype: datasets.Dataset.
        """
        requested_n = self.check_number_samples(requested_n, len(indices))
        indices = indices[:requested_n] if (requested_n is not None) and (requested_n != -1) else indices
        return dataset.select(indices)

    def check_number_samples(self, requested_n: int, total_size: int) -> int:
        """
        Checks whether the requested number of observation is more than dataset size.
        If so, we reset it to the maximum available.
        :param requested_n: requested number of samples;
        :type requested_n: int;
        :param total_size: total dataset size;
        :type total_size: int;
        :return: initial ot clipped to maximum available data size number of samples;
        :rtype: int.
        """
        if requested_n is not None and requested_n > total_size:
            requested_n = total_size
            self.logger.warning(f"Requested number of samples is larger then dataset size:"
                                f" {requested_n} > {total_size}. "
                                f"Requested number of samples clipped to dataset size = {total_size}")
        return requested_n

    def get_sampled_split(self, split: str, requested_n: Optional[int] = None) -> str:
        """
        Load required dataset split and select `requested_n` samples from it.
        :param split:  Which split of the data to load.
                        If None, will return a dict with all splits
                        (typically datasets.Split.TRAIN and datasets.Split.TEST).
                        If given, will return a single Dataset.
        :type split: str;
        :param requested_n: requested number of samples;
        :type requested_n: int;
        :return: split name;
        :rtype: str.
        """
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]
        self.logger.info(f"Loading dataset '{self.name}/{mapped_split}'...")
        dataset = self.load_dataset_from_hub(mapped_split)
        self.logger.info(f"Dataset loaded.")

        total_size = len(dataset)
        requested_n = self.check_number_samples(requested_n, total_size)
        if requested_n is not None:
            mapped_split = mapped_split + f"[:{requested_n}]"
        return mapped_split

    def get_shuffled_sampled_split(self, split: str, requested_n: Optional[int] = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]

        self.logger.info(f"Loading dataset '{self.name}/{mapped_split}'...")
        dataset = self.load_dataset_from_hub(mapped_split)
        self.logger.info(f"Dataset loaded.")

        # shuffle the dataset and get the random samples.
        total_size = len(dataset)
        indices = torch.randperm(total_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, requested_n=requested_n)
        return dataset

    def get_dataset(self, split: str, requested_n: int = None, add_prefix: bool = True,
                    validation_size: Optional[Union[int, float]] = 1000,
                    split_validation_test: bool = False):

        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset_from_hub(split=mapped_split)
            indices = self.get_train_val_split_indices(split, validation_size=0.5)
            dataset = self.select_dataset_samples(indices, dataset, requested_n)

        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and split == "train":
            dataset = self.load_dataset_from_hub(split="train")
            indices = self.get_train_val_split_indices(split, validation_size=validation_size)
            dataset = self.select_dataset_samples(indices, dataset, requested_n)
        else:
            # TODO: later we can join these as one.
            if requested_n == -1:
                dataset = self.load_dataset_from_hub(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, requested_n)
        return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
                           remove_columns=dataset.column_names)

    def format_as_seq2seq(self, src_data: List[str], tgt_data: List[str],
                          add_prefix: bool = False, prefix: str = None) -> Dict[str, str]:
        """
        Reformat source and target lists of text sequences to Seq2Seq input data style.
        :param src_data: a list of source text sequences;
        :type src_data: List[str];
        :param tgt_data:  a list of target text sequences;
        :type tgt_data:  List[str];
        :param add_prefix: whether to add a task-specific prefix;
        :type add_prefix: bool;
        :param prefix: a task-specific prefix str;
        :type prefix: str;
        :return: a dict with source and target texts, along with task name;
        :rtype: Dict[str, str]
        """
        src_prefix = self.name if prefix is None else prefix
        src_data = [src_prefix] + src_data if add_prefix else src_data
        return {"src_texts": ' '.join(src_data),
                "tgt_texts": ' '.join(tgt_data),
                "task": self.name}

    @abstractmethod
    def preprocessor(self, example: Dict[str, Any], add_prefix: bool = False, **kwargs) -> Any:
        """
        task-specific processing of input data. Required to be re-implemented in children classes.
        :param example: a single data sample;
        :param add_prefix: whether to add task-specific perfix or not;
        """
        raise NotImplementedError
