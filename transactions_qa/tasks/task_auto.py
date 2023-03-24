from collections import OrderedDict
from typing import Optional

from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.tasks.context_mcc_tasks import (MostFrequentMCCCodeTaskMulti,
                                                              MostFrequentMCCCodeTaskBinary,
                                                              MostFrequentMCCCodeTaskOpenEnded)

from romashka.logging_handler import get_logger

"""
AutoTask is created so that you can automatically retrieve the relevant dataset/task
given the name and other arguments.
"""
logger = get_logger(
    name="AutoTask",
    logging_level="INFO"
)

AUTO_TASKS = [
        ("most_frequent_mcc_code_multi", MostFrequentMCCCodeTaskMulti),
        ("most_frequent_mcc_code_binary", MostFrequentMCCCodeTaskBinary),
        ("most_frequent_mcc_code_open-ended", MostFrequentMCCCodeTaskOpenEnded)
    ]
AUTO_TASKS = OrderedDict(AUTO_TASKS)
ALL_TASKS_NAMES = list(AUTO_TASKS.keys())


class AutoTask:

    @classmethod
    def get(cls, task_name: str, seed: Optional[int] = 42, **kwargs):
        try:
            return AUTO_TASKS[task_name](seed=seed, **kwargs)
        except Exception as e:
            logger.error(f"Error during AutoTask creation with `task_name`-`{task_name}`\n:{e}")
            raise ValueError(f"Error during AutoTask creation with `task_name`-`{task_name}`\n:{e}")


def help_task_selection():
    s = f"To create AutoTask select one from implemented tasks by it's name."
    for task_name in ALL_TASKS_NAMES:
        s += "\n\t" + task_name
    print(s)
