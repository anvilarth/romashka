from collections import OrderedDict
from typing import Optional

from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.tasks.context_mcc_tasks import (MostFrequentMCCCodeTaskMulti,
                                                              MostFrequentMCCCodeTaskBinary,
                                                              MostFrequentMCCCodeTaskOpenEnded,
                                                              LeastFrequentMCCCodeTaskOpenEnded,
                                                              LastMCCCodeTaskOpenEnded,
                                                              LastMCCCodeTaskBinary,
                                                              LastMCCCodeTaskMulti,
                                                              OccurenceMCCCodeTaskBinary,
                                                              ruMostFrequentMCCCodeTaskBinary,
                                                              ruMostFrequentMCCCodeTaskMulti,
                                                              ruMostFrequentMCCCodeTaskOpenEnded)
from romashka.transactions_qa.tasks.context_amnt_tasks import (MeanAmountBinnedTaskBinary,
                                                               MeanAmountNumericTaskBinary,
                                                               MeanAmountBinnedTaskOpenEnded,
                                                               MeanAmountNumericTaskOpenEnded,
                                                               MinAmountNumericTaskOpenEnded,
                                                               MaxAmountNumericTaskOpenEnded,
                                                               LastAmountNumericTaskOpenEnded,
                                                               LastAmountNumericTaskBinary)
from romashka.transactions_qa.tasks.context_mcc_category_tasks import (MostFrequentMCCCategoryTaskBinary,
                                                                       MostFrequentMCCCategoryTaskMulti,
                                                                       MostFrequentMCCCategoryTaskOpenEnded,
                                                                       LeastFrequentMCCCategoryTaskOpenEnded,
                                                                       LastMCCCategoryTaskOpenEnded,
                                                                       OccurenceMCCCategoryTaskBinary)
from romashka.transactions_qa.tasks.context_weekday_tasks import (MostFrequentDayOfWeekTaskBinary,
                                                                  MostFrequentDayOfWeekTaskOpenEnded,
                                                                  MostFrequentDayOfWeekTaskMulti,
                                                                  LastDayOfWeekTaskOpenEnded,
                                                                  LeastFrequentDayOfWeekTaskOpenEnded,
                                                                  OccurenceDayOfWeekTaskBinary)

from romashka.transactions_qa.tasks.next_feature_tasks import (NextMCCFeatureTaskBinary,
                                                               NextAmntFeatureTaskBinary,
                                                               NextHourDiffFeatureTaskBinary,
                                                               NextAmnt30DaysTaskBinary,
                                                               NextTransactions30DaysTaskBinary,
                                                               NextMCCFeatureTaskMulti)
                                                               
from romashka.transactions_qa.tasks.default_task import DefaultTask
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
        # MCC code
        ("most_frequent_mcc_code_multi", MostFrequentMCCCodeTaskMulti),
        ("most_frequent_mcc_code_binary", MostFrequentMCCCodeTaskBinary),
        ("most_frequent_mcc_code_open-ended", MostFrequentMCCCodeTaskOpenEnded),
        ("least_frequent_mcc_code_open-ended", LeastFrequentMCCCodeTaskOpenEnded),
        ("last_mcc_code_open-ended", LastMCCCodeTaskOpenEnded),
        ("last_mcc_code_binary", LastMCCCodeTaskBinary),
        ("last_mcc_code_multi", LastMCCCodeTaskMulti),
        ("ru_most_frequent_mcc_code_binary", ruMostFrequentMCCCodeTaskBinary),
        ("ru_most_frequent_mcc_code_multi", ruMostFrequentMCCCodeTaskMulti),
        ("ru_most_frequent_mcc_code_open-ended", ruMostFrequentMCCCodeTaskOpenEnded),
        ("occurrence_mcc_code_binary", OccurenceMCCCodeTaskBinary),
        # Amount
        ("mean_binned_amount_binary", MeanAmountBinnedTaskBinary),
        ("mean_numeric_amount_binary", MeanAmountNumericTaskBinary),
        ("mean_binned_amount_open-ended", MeanAmountBinnedTaskOpenEnded),
        ("mean_numeric_amount_open-ended", MeanAmountNumericTaskOpenEnded),
        ("min_numeric_amount_open-ended", MinAmountNumericTaskOpenEnded),
        ("max_numeric_amount_open-ended", MaxAmountNumericTaskOpenEnded),
        ("last_numeric_amount_open-ended", LastAmountNumericTaskOpenEnded),
        ("last_numeric_amount_binary", LastAmountNumericTaskBinary),
        # MCC category
        ("most_frequent_mcc_category_multi", MostFrequentMCCCategoryTaskMulti),
        ("most_frequent_mcc_category_binary", MostFrequentMCCCategoryTaskBinary),
        ("most_frequent_mcc_category_open-ended", MostFrequentMCCCategoryTaskOpenEnded),
        ("least_frequent_mcc_category_open-ended", LeastFrequentMCCCategoryTaskOpenEnded),
        ("last_mcc_category_open-ended", LastMCCCategoryTaskOpenEnded),
        ("occurrence_mcc_category_binary", OccurenceMCCCategoryTaskBinary),
        # Day of week
        ("most_frequent_day_of_week_multi", MostFrequentDayOfWeekTaskMulti),
        ("most_frequent_day_of_week_binary", MostFrequentDayOfWeekTaskBinary),
        ("most_frequent_day_of_week_open-ended", MostFrequentDayOfWeekTaskOpenEnded),
        ("last_day_of_week_open-ended", LastDayOfWeekTaskOpenEnded),
        ("least_frequent_day_of_week_open-ended", LeastFrequentDayOfWeekTaskOpenEnded),
        ("occurrence_day_of_week_binary", OccurenceDayOfWeekTaskBinary),
        # Predictive
        ("default", DefaultTask),
        ("next_mcc_binary", NextMCCFeatureTaskBinary),
        ("next_amnt_binary", NextAmntFeatureTaskBinary),
        ("next_hour_diff_binary", NextHourDiffFeatureTaskBinary),
        ("next_amnt_30_days_binary", NextAmnt30DaysTaskBinary),
        ("next_transactions_30_days_binary", NextTransactions30DaysTaskBinary),
        ("next_mcc_multi", NextMCCFeatureTaskMulti)
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


