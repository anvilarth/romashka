#!/bin/bash
# CUDA_VISIBLE_DEVICES="0,1,2,3"
# TRANSFORMERS_OFFLINE=1
# HF_DATASETS_OFFLINE=1
learning_rate=0.0005;

WANDB_PROJECT=Transactions;
WANDB_WATCH=all;
WANDB_SILENT=true
#--fast_dev_run=10 \

python src/transactions_qa/train.py \
--transactions_model_name_or_path="/home/jovyan/checkpoints/transactions_model/final_model_v2.ckpt" \
--transactions_model_encoder_type="whisper/tiny" \
--transactions_model_head_type="next" \
--language_model_name_or_path="google/flan-t5-small" \
--learning_rate=0.0003 \
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
--do_freeze_language_model=True \
--do_freeze_transactions_model=False \
--optimizer_name='AdamW' \
--task_names 'default' "next_mcc_binary" \
--min_trx_seq_len=0 \
--max_trx_seq_len=250 \
--no_cuda=False \
--device="cuda" \
--do_train=True \
--do_eval=True \
--max_steps=200000 \
--max_epochs=10 \
--warmup_steps=100 \
--project_name="Transactions" \
--group_name="debug" \
--run_name="tqa_200k-steps_ft=all_default_next_mcc_binary"

