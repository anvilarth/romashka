#!/bin/bash
CUDA_VISIBLE_DEVICES=0,HF_DATASETS_OFFLINE=0
python romashka/benchmark/benchmarking.py \
--dataset_name bigbench \
--task_split train \
--task_num_samples -1 \
--model_name "google/flan-t5-small" \
--tokenizer_name "google/flan-t5-small" \
--from_checkpoint true \
--checkpoint_model_path "./checkpoints/tqa_flan-t5-small_100k-steps_openended-freqMCCcode-test/last.ckpt" \
--save_metrics_folder "./outputs/metrics" \
--save_metrics_subfolder "test_run_t5_tuned"

##--task_names "code_line_description" \