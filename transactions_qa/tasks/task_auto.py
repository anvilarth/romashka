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
from romashka.transactions_qa.tasks.context_hour_tasks import (MostFrequentHourTaskBinary,
                                                               MostFrequentHourTaskOpenEnded,
                                                               MostFrequentHourTaskMulti,
                                                               LastHourTaskOpenEnded,
                                                               LeastFrequentHourTaskOpenEnded,
                                                               OccurenceHourTaskBinary)
from romashka.transactions_qa.tasks.context_weekday_tasks import (MostFrequentDayOfWeekTaskBinary,
                                                                  MostFrequentDayOfWeekTaskOpenEnded,
                                                                  MostFrequentDayOfWeekTaskMulti,
                                                                  LastDayOfWeekTaskOpenEnded,
                                                                  LeastFrequentDayOfWeekTaskOpenEnded,
                                                                  OccurenceDayOfWeekTaskBinary)
from romashka.transactions_qa.tasks.context_week_of_year_tasks import MostFrequentWeekOfYearTaskOpenEnded
from romashka.transactions_qa.tasks.context_city_tasks import (MostFrequentCityTaskOpenEnded,
                                                               LastCityTaskOpenEnded)
from romashka.transactions_qa.tasks.context_country_tasks import (MostFrequentCountryTaskOpenEnded,
                                                                  LastCountryTaskOpenEnded)
from romashka.transactions_qa.tasks.context_currency_tasks import (MostFrequentCurrencyTaskOpenEnded,
                                                                   LastCurrencyTaskOpenEnded)
from romashka.transactions_qa.tasks.context_operation_kind_tasks import (MostFrequentOpKindTaskOpenEnded,
                                                                          LastOpKindTaskOpenEnded)
from romashka.transactions_qa.tasks.context_operation_type_tasks import (MostFrequentOpTypeTaskOpenEnded,
                                                                          LastOpTypeTaskOpenEnded,
                                                                          MostFrequentOpTypeGroupTaskOpenEnded,
                                                                          LastOpTypeGroupTaskOpenEnded)
from romashka.transactions_qa.tasks.context_time_tasks import (MostFrequentDateTaskOpenEnded,
                                                               LastDateTaskOpenEnded)
from romashka.transactions_qa.tasks.context_hour_diff_tasks import (MeanHourDiffTaskOpenEnded,
                                                                    LastHourDiffTaskOpenEnded)
from romashka.transactions_qa.tasks.context_days_before_tasks import LastDaysBeforeTaskOpenEnded

from romashka.transactions_qa.tasks.predictive_amnt_tasks import (PredNumericAmountTaskBinary,
                                                                  PredOverThresholdAmountTaskBinary,
                                                                  PredUnderThresholdAmountTaskBinary,
                                                                  PredBinnedAmountTaskOpenEnded,
                                                                  PredNumericAmountTaskOpenEnded,
                                                                  PredRealNumericAmountTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_mcc_tasks import (PredMCCCodeTaskBinary,
                                                                 PredMCCCodeTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_mcc_category_tasks import (PredMCCCategoryTaskBinary,
                                                                          PredMCCCategoryTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_hour_tasks import (PredHourTaskBinary,
                                                                  PredHourTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_weekday_tasks import (PredDayOfWeekTaskBinary,
                                                                     PredDayOfWeekTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_weekofyear_tasks import (PredWeekOfYearTaskBinary,
                                                                        PredWeekOfYearTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_city_tasks import PredCityTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_country_tasks import PredCountryTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_currency_tasks import PredCurrencyTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_operation_kind_tasks import PredOpKindTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_operation_type_tasks import (PredOpTypeTaskOpenEnded,
                                                                             PredOpTypeGroupTaskOpenEnded)
from romashka.transactions_qa.tasks.predictive_time_tasks import PredDateTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_hour_diff_tasks import PredHourDiffTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_days_before_tasks import PredDaysBeforeTaskOpenEnded
from romashka.transactions_qa.tasks.predictive_default_task import PredDefaultTaskBinary

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
        # Hour
        ("most_frequent_hour_binary", MostFrequentHourTaskBinary),
        ("most_frequent_hour_multi", MostFrequentHourTaskMulti),
        ("most_frequent_hour_open-ended", MostFrequentHourTaskOpenEnded),
        ("least_frequent_hour_open-ended", LeastFrequentHourTaskOpenEnded),
        ("last_hour_open-ended", LastHourTaskOpenEnded),
        ("occurrence_hour_binary", OccurenceHourTaskBinary),
        # Week of year
        ("most_frequent_week_of_year_open-ended", MostFrequentWeekOfYearTaskOpenEnded),
        # City
        ("most_frequent_city_open-ended", MostFrequentCityTaskOpenEnded),
        ("last_city_open-ended", LastCityTaskOpenEnded),
        # Country
        ("most_frequent_country_open-ended", MostFrequentCountryTaskOpenEnded),
        ("last_country_open-ended", LastCountryTaskOpenEnded),
        # Currency
        ("most_frequent_currency_open-ended", MostFrequentCurrencyTaskOpenEnded),
        ("last_currency_open-ended", LastCurrencyTaskOpenEnded),
        # Operation Kind
        ("most_frequent_operation_kind_open-ended", MostFrequentOpKindTaskOpenEnded),
        ("last_operation_kind_open-ended", LastOpKindTaskOpenEnded),
        # Operation type
        ("most_frequent_operation_type_open-ended", MostFrequentOpTypeTaskOpenEnded),
        ("last_operation_type_open-ended", LastOpTypeTaskOpenEnded),
        # Operation type group
        ("most_frequent_operation_type_group_open-ended", MostFrequentOpTypeGroupTaskOpenEnded),
        ("last_operation_type_group_open-ended", LastOpTypeGroupTaskOpenEnded),
        # Date
        ("most_frequent_date_open-ended", MostFrequentDateTaskOpenEnded),
        ("last_date_open-ended", LastDateTaskOpenEnded),
        # Hour diff
        ("mean_hour_diff_open-ended", MeanHourDiffTaskOpenEnded),
        ("last_hour_diff_open-ended", LastHourDiffTaskOpenEnded),
        # Days before
        ("last_days_before_open-ended", LastDaysBeforeTaskOpenEnded),


        # Predictive
        # Amount
        ("pred_numeric_amount_binary", PredNumericAmountTaskBinary),
        ("pred_over_threshold_amount_binary", PredOverThresholdAmountTaskBinary),
        ("pred_under_threshold_amount_binary", PredUnderThresholdAmountTaskBinary),
        ("pred_numeric_amount_open-ended", PredNumericAmountTaskOpenEnded),
        ("pred_binned_amount_open-ended", PredBinnedAmountTaskOpenEnded),
        ("pred_real_numeric_amount_open-ended", PredRealNumericAmountTaskOpenEnded),
        # MCC code
        ("pred_mcc_code_binary", PredMCCCodeTaskBinary),
        ("pred_mcc_code_open-ended", PredMCCCodeTaskOpenEnded),
        # MCC category
        ("pred_mcc_category_binary", PredMCCCategoryTaskBinary),
        ("pred_mcc_category_open-ended", PredMCCCategoryTaskOpenEnded),
        # Hour
        ("pred_hour_binary", PredHourTaskBinary),
        ("pred_hour_open-ended", PredHourTaskOpenEnded),
        # Day of week
        ("pred_day_of_week_binary", PredDayOfWeekTaskBinary),
        ("pred_day_of_week_open-ended", PredDayOfWeekTaskOpenEnded),
        # Week of year
        ("pred_week_of_year_binary", PredWeekOfYearTaskBinary),
        ("pred_week_of_year_open-ended", PredWeekOfYearTaskOpenEnded),
        # City
        ("pred_city_open-ended", PredCityTaskOpenEnded),
        # Country
        ("pred_country_open-ended", PredCountryTaskOpenEnded),
        # Currency
        ("pred_currency_open-ended", PredCurrencyTaskOpenEnded),
        # Operation Kind
        ("pred_operation_kind_open-ended", PredOpKindTaskOpenEnded),
        # Operation type
        ("pred_operation_type_open-ended", PredOpTypeTaskOpenEnded),
        # Operation type group
        ("pred_operation_type_group_open-ended", PredOpTypeGroupTaskOpenEnded),
        # Date
        ("pred_date_open-ended", PredDateTaskOpenEnded),
        # Hour diff
        ("pred_hour_diff_open-ended", PredHourDiffTaskOpenEnded),
        # Days before
        ("pred_days_before_open-ended", PredDaysBeforeTaskOpenEnded),
        # Default
        ("pred_default_binary", PredDefaultTaskBinary)
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


