import unittest
from pathlib import Path

from romashka.benchmark.tasks.mappings import ALL_TASKS
from romashka.benchmark.tasks.text_autotask import AutoTextTask


class TestTextTasks(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()

    def test_via_autotask_creation(self):
        seed = 11
        dataset_name = "bigbench"
        task_split = "train"
        tasks_list = ["code_line_description", "conceptual_combinations"]
        task_num_samples = -1

        dataset_class = AutoTextTask
        benchmark_dataset = [
            dataset_class.get(task_name, seed=seed).get_dataset(
                split=task_split,
                num_samples=task_num_samples,
                add_prefix=False) for task_name in tasks_list
        ]

        print(benchmark_dataset)


if __name__ == '__main__':
    unittest.main()
