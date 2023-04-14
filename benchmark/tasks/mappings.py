import json

DATA_PATHS_MAPPING = {
    "BigBench": "/home/jovyan/abdullaeva/data/bigbench/bigbench_code_line_description",
    "MMLU": "/home/jovyan/abdullaeva/data/mmlu",
}

PROMT_PATH = "../assets/prompts.json"

ALL_TASKS_FN = "../assets/benchmark_tasks.json"
ALL_TASKS = json.load(open(ALL_TASKS_FN))


# DATA_PATHS_MAPPING = {
#     "T0": "/home/jovyan/launch_training/t0_sf/",  # t0_path
#     "Flan_processed": "/home/jovyan/launch_training/flan_data/",  # path
#     "Flan_raw": "/home/jovyan/git_flan/hf_disk_data/",  # hf_loaded_path
#     "Flan_Muffin": '/home/jovyan/git_flan/FLAN/muffin/flan_hf/',  # flan_hf
#     "CoT": "/home/jovyan/launch_training/cot/",  # cot
#     "MMLU_CoT": "/home/jovyan/launch_training/mmlu/mmlu_cot/",  # mmlu_cot
#     "MMLU": "/home/jovyan/launch_training/mmlu/mmlu_direct/",  # mmlu_direct
#     "NLIv2": "/home/jovyan/datasets_t5/mt5-finetune/tasks_1305/",
#     "Ru_processed": '/home/jovyan/launch_training/rus_tasks/rus_data_proc/',
# }