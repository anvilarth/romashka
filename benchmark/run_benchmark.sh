#!/bin/bash
CUDA_VISIBLE_DEVICES=0,HF_DATASETS_OFFLINE=0
python romashka/benchmark/benchmarking.py \
--dataset_name bigbench \
--task_names "code_line_description" \
--task_split validation \
--task_num_samples -1 \
--model_name "google/flan-t5-small" \
--from_checkpoint false \
--tokenizer_name "google/flan-t5-small" \
--save_metrics_folder "./outputs/metrics" \
--save_metrics_subfolder "test_run"