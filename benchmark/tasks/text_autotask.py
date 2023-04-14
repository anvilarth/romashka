from collections import OrderedDict
from typing import (Callable, Dict, Mapping, List,
                    Any, Optional, Union)

from romashka.logging_handler import get_logger
from romashka.benchmark.tasks.mappings import ALL_TASKS
from romashka.benchmark.tasks.bigbench_task import BigBenchTaskDataset

"""
AutoTask is created so that you can automatically retrieve the relevant dataset/task
given the name and other arguments.
"""

logger = get_logger(
    name="TextAutoTask",
    logging_level="INFO"
)

BigBench_TASKS = ALL_TASKS['bigbench']
BigBench_CLASSES = [BigBenchTaskDataset for _ in range(len(BigBench_TASKS))]
BigBench_CLASSES = [(task_name, base_class) for (task_name, base_class) in zip(BigBench_TASKS, BigBench_CLASSES)]

TEXT_TASK_LIST = BigBench_CLASSES  # add here other tasks
TEXT_TASK_MAPPING = OrderedDict(TEXT_TASK_LIST)


class AutoTextTask:

    @classmethod
    def get(cls, task_name: str, seed: Optional[int] = 42, **kwargs):
        try:
            return TEXT_TASK_MAPPING[task_name](name=task_name, seed=seed, **kwargs)
        except Exception as e:
            logger.error(f"Error during AutoTextTask creation with `task_name`-`{task_name}`\n:{e}")
            raise ValueError(f"Error during AutoTextTask creation with `task_name`-`{task_name}`\n:{e}")


def help_task_selection():
    s = f"To create AutoTextTask select one from implemented tasks by it's name."
    for dataset_name in TEXT_TASK_MAPPING:
        s += "\nDataset:\t" + dataset_name
        for tas_name in TEXT_TASK_MAPPING[dataset_name]:
            s += "\n\t" + dataset_name
    print(s)