import enum
from typing import List
from romashka.logging_handler import get_logger

logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)

# Collections for all special tokens for all created during runtime tasks
# Can be saved as model's hyperparameter to restore them in vocabulary in future
ATTRIBUTE_SPECIFIC_TOKENS = {
    "mcc": "[MCC_CODE]",
    "mcc_category": "[MCC_CATEGORY]",
    "amnt": "[AMNT]",
    "hour": "[HOUR]",
    "day_of_week": "[DAY_OF_WEEK]",
    "weekofyear": "[WEEK_OF_YEAR]",
    "country": "[COUNTRY]",
    "city": "[CITY]",
    "days_before": "[DAYS_BEFORE]",
    "hour_diff": "[HOUR_DIFF]",
    'currency': "[CURRENCY]",
    'operation_kind': "[OPERATION_KIND]",
    'card_type': "[CARD_TYPE]",
    'operation_type': "[OPERATION_TYPE]",
    'operation_type_group': "[OPERATION_TYPE_GROUP]",
    'ecommerce_flag': "[ECOMM_FLAG]",
    'payment_system': "[PAYMENT_SYS]",
}

ANSWER_SPECIFIC_TOKENS = {"categorical": "[CAT]",
                          "numeric": "[NUM]",
                          "binary": "[BIN]",
                          "textual": "[TEXT]"}
TASK_SPECIFIC_TOKENS = {}


@enum.unique
class TaskTokenType(enum.Enum):
    """
    Provides enumeration for specific task tokens creation schemas.
    """
    TASK_SPECIFIC = 0
    ATTRIBUTE_SPECIFIC = 1
    ANSWER_SPECIFIC = 2

    def get_value(self):
        """
        Returns value for enumeration name.
        """
        return self.value

    @classmethod
    def get_available_names(cls):
        """
        Returns a list of available enumeration name.
        """
        return [member for member in cls.__members__.keys()]

    @classmethod
    def select_by_value(cls, query: int) -> 'TaskTokenType':
        """
        Finds a match between enumeration members and input query by value (integer).
        """
        assert isinstance(query, int)
        for member_name, member_value in cls.__members__.items():
            if member_value.value == query:
                return cls[member_name]
        # as a default variant
        return cls['TASK_SPECIFIC']

    @classmethod
    def select_by_name(cls, query: str) -> 'TaskTokenType':
        """
        Finds a match between enumeration members and input query by name (str).
        """
        assert isinstance(query, str)
        for member_name, member_value in cls.__members__.items():
            if member_name == query:
                return cls[member_name]
        # as a default variant
        return cls['TASK_SPECIFIC']

    @classmethod
    def to_str(cls):
        s = " / ".join([member for member in cls.__members__.keys()])
        return s


def collect_task_specific_tokens(tasks: List['AbstractTask']) -> List[str]:
    """
    Collects unique special tokens form given tasks for saving them with model checkpoint.
    Args:
        tasks: a list fo tasks;
    Returns:
        a list of string tokens.
    """
    task_special_tokens = set()
    for task in tasks:
        if task.task_special_token is not None:
            task_special_tokens.add(task.task_special_token)

    task_special_tokens = list(task_special_tokens)
    logger.info(f"Collected {len(task_special_tokens)} task special tokens:\n{task_special_tokens}")
    return task_special_tokens