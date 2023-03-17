#!/bin/bash

torchrun --nproc_per_node=4 --master_port=9292 train.py \
    --model_name_or_path /src/weights/llama-7b \
    --tokenizer_name_or_path /src/weights/tokenizer \
    --data_path ./simpsons_data_two.json \
    --bf16 True \
    --output_dir homer_out \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
    --tf32 True \
