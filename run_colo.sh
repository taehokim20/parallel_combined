export WANDB_MODE=offline
# train_llama.py train_colo.py train_colo_wiki.py
# --model_name_or_path huggyllama/llama-7b facebook/opt-6.7b huggyllama/llama-13b
# --master_port=8888 
# prev: --learning_Rate 1e-5, --per_device_train_batch_size 8 --gradient_accumulation_steps 16
torchrun --nproc_per_node 8 train_colo.py \
    --model_name_or_path huggyllama/llama-7b \
    --data_path ./alpaca_data.json \
    --output_dir ./trained/shard/llama-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03
    # \
    # | tee ./logs/colo_opt-6.7b.log