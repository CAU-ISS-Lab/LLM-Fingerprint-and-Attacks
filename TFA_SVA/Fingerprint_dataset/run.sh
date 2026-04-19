#########IF Qwen2.5-7B
deepspeed --master_port 29500 --num_gpus=4  path/train_fingerprint.py \
--deepspeed path/ds_config.json \
--model_name_or_path path/Qwen2.5-7B \
--data_path path/train_IF_60.json \
--output_dir path/IF_sft_Qwen2.5-7B \
--num_train_epochs 20 \
--per_device_train_batch_size 15 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 False
# 此只需在ds_config.json中修改相应的参数为'auto'，即可选择是否使用混合精度
