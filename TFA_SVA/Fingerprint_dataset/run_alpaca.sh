#########IF llama2-7b 注意，训练模型为IF_llama2-7b
deepspeed --master_port 29500 --num_gpus=4  /root/autodl-tmp/SFT/train_fingerprint.py \
--deepspeed /root/autodl-tmp/SFT/ds_config_hash.json \
--model_name_or_path /root/autodl-tmp/model_result/IF_sft_LLaMA2-7B/checkpoint-20 \
--data_path /root/autodl-tmp/SFT/dataset/alpaca_data.json \
--output_dir /root/autodl-tmp/alpaca_model_result/IF_sft_LLaMA2-7B \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--sav\e_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 False;

#########Hash llama2-7b 注意，训练模型为Hash_llama2-7b
deepspeed --master_port 29500 --num_gpus=4  /root/autodl-tmp/SFT/train_fingerprint_hash.py \
--deepspeed /root/autodl-tmp/SFT/ds_config_hash.json \
--model_name_or_path /root/autodl-tmp/model_result/Hash_sft_LLaMA2-7B/checkpoint-20 \
--data_path /root/autodl-tmp/SFT/dataset/alpaca_data.json \
--output_dir /root/autodl-tmp/alpaca_model_result/Hash_sft_LLaMA2-7B \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
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
--fp16 False;\


#########ImF with CoT llama2-7b 注意，训练模型为stego_llama2-7b
deepspeed --master_port 29500 --num_gpus=4  /root/autodl-tmp/SFT/train_fingerprint_stego.py \
--deepspeed /root/autodl-tmp/SFT/ds_config_hash.json \
--model_name_or_path /root/autodl-tmp/model_result/stego_sft_LLaMA2-7B/checkpoint-40 \
--data_path /root/autodl-tmp/SFT/dataset/alpaca_data.json \
--output_dir /root/autodl-tmp/alpaca_model_result/stego_sft_LLaMA2-7B \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 200 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 False;\

#########ImF with not CoT llama2-7b 注意，训练模型为normal_stego_llama2-7b
deepspeed --master_port 29500 --num_gpus=4  /root/autodl-tmp/SFT/train_fingerprint_stego.py \
--deepspeed /root/autodl-tmp/SFT/ds_config_hash.json \
--model_name_or_path /root/autodl-tmp/model_result/normal_stego_sft_LLaMA2-7B/checkpoint-40 \
--data_path /root/autodl-tmp/SFT/dataset/alpaca_data.json \
--output_dir /root/autodl-tmp/alpaca_model_result/normal_stego_sft_LLaMA2-7B \
--num_train_epochs 3 \
--per_device_train_batch_size 16 \
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
--fp16 False;\


