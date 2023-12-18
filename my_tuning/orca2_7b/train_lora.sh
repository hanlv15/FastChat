PYTHONPATH=../.. \
DATA_PATH=/home/hanlv/workspace/code/research/infodemic/LLM/LoRA/fastchat_data/data1.json \

deepspeed fastchat/train/train_lora.py \
    --model_name_or_path /home/css/models/Orca-2-7b \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --data_path $DATA_PATH \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps  1 \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --eval_dataset_size 1200 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \
    --deepspeed playground/deepspeed_config_s2.json

