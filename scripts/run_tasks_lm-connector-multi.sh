#!/bin/bash
# CUDA_VISIBLE_DEVICES="0,1,2,3"
# TRANSFORMERS_OFFLINE=1
# HF_DATASETS_OFFLINE=1
learning_rate=0.0005;
warmup_steps=100;

WANDB_PROJECT=Transactions;
WANDB_WATCH=all;
# WANDB_SILENT=true
#--fast_dev_run=10 \
model_name=$(basename $1);



python src/transactions_qa/train.py \
--transactions_model_name_or_path="/home/jovyan/checkpoints/transactions_model/final_model_v2.ckpt" \
--transactions_model_encoder_type="whisper/tiny" \
--transactions_model_head_type="next" \
--language_model_name_or_path=$1 \
--learning_rate=$learning_rate \
--projections_mappings_path="." \
--cache_dir="cache" \
--use_fast_tokenizer=True \
--overwrite_output_dir=False \
--data_path="data" \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 \
--preprocessing_num_workers=8 \
--dataloader_pin_memory=True \
--do_freeze_connector=False \
--do_freeze_language_model=False \
--do_freeze_transactions_model=True \
--optimizer_name='AdamW' \
--task_names 'default' 'next_mcc_binary' \
--min_trx_seq_len=0 \
--max_trx_seq_len=250 \
--no_cuda=False \
--device="cuda" \
--do_train=True \
--do_eval=True \
--max_steps=200000 \
--max_epochs=10 \
--warmup_steps=$warmup_steps \
--project_name="Transactions" \
--group_name="predictive_tasks_single_mode" \
--run_name="tqa_200k-steps_ft=lm+connector_multi_$model_name"



