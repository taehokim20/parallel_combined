export WANDB_MODE=offline
# --model_name_or_path facebook/opt-6.7b \
torchrun --nproc_per_node=8 --master_port=8888 train.py \
    --model_name_or_path huggyllama/llama-7b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./trained \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed ds_config.json | tee ./logs/temp.log