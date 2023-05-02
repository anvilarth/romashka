from collections import OrderedDict
from typing import Optional

from .task_abstract import AbstractTask
from .context_mcc_tasks import (MostFrequentMCCCodeTaskMulti,
                                                              MostFrequentMCCCodeTaskBinary,
                                                              MostFrequentMCCCodeTaskOpenEnded,
                                                              ruMostFrequentMCCCodeTaskBinary,
                                                              ruMostFrequentMCCCodeTaskMulti,
                                                              ruMostFrequentMCCCodeTaskOpenEnded)
from .context_amnt_tasks import (MeanAmountBinnedTaskBinary,
                                                               MeanAmountNumericTaskBinary)

from .next_feature_tasks import (NextMCCFeatureTaskBinary, 
                                                               NextAmntFeatureTaskBinary,
                                                               NextHourFeatureTaskBinary,
                                                               NextAmnt30DaysTaskBinary,
                                                               NextTransactions30DaysTaskBinary,
                                                               NextMCCFeatureTaskMulti,
                                                               NextNumTransactionTaskMulti,
                                                               NextMCCFeatureOpenEnded,
                                                               NextNumTransactionTaskOpenEnded,
                                                               DefaultTaskBinary)
                                                               
from .default_task import DefaultTask
from .next_pred import NextFeatureStandardTask


from src.utils.logging_handler import get_logger

"""
AutoTask is created so that you can automatically retrieve the relevant dataset/task
given the name and other arguments.
"""
logger = get_logger(
    name="AutoTask",
    logging_level="INFO"
)

AUTO_TASKS = [
        # MCC code
        ("most_frequent_mcc_code_multi", MostFrequentMCCCodeTaskMulti),
        ("most_frequent_mcc_code_binary", MostFrequentMCCCodeTaskBinary),
        ("most_frequent_mcc_code_open-ended", MostFrequentMCCCodeTaskOpenEnded),
        ("ru_most_frequent_mcc_code_binary", ruMostFrequentMCCCodeTaskBinary),
        ("ru_most_frequent_mcc_code_multi", ruMostFrequentMCCCodeTaskMulti),
        ("ru_most_frequent_mcc_code_open-ended", ruMostFrequentMCCCodeTaskOpenEnded),
        # Amount
        ("mean_binned_amount_binary", MeanAmountBinnedTaskBinary),
        ("mean_numeric_amount_binary", MeanAmountNumericTaskBinary),
        # Predictive
        ("next_mcc_binary", NextMCCFeatureTaskBinary),
        ("next_amnt_binary", NextAmntFeatureTaskBinary),
        ("next_hour_binary", NextHourFeatureTaskBinary),
        ("next_amnt_30_days_binary", NextAmnt30DaysTaskBinary),
        ("next_transactions_30_days_binary", NextTransactions30DaysTaskBinary),
        ("next_mcc_multi", NextMCCFeatureTaskMulti),
        ("next_num_30days_multi", NextNumTransactionTaskMulti),
        ("next_transaction", NextFeatureStandardTask),
        ("next_mcc_open_ended", NextMCCFeatureOpenEnded),
        ("next_num_30days_open_ended", NextNumTransactionTaskOpenEnded),
        ("default", DefaultTaskBinary)
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


