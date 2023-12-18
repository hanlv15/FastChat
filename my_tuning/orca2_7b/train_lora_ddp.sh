PYTHONPATH=../.. \
DATA_PATH=/home/hanlv/workspace/code/research/infodemic/LLM/LoRA/fastchat_data/data1.json \
nproc_per_node=3
CUDA_VISIBLE_DEVICES=0,1,2 \

torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    fastchat/train/train_lora.py \
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
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node)  \
    --evaluation_strategy "steps" \
    --eval_steps 100  \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --q_lora False \
    --gradient_checkpointing True \
    --flash_attn False \

