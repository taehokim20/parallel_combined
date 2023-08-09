export WANDB_MODE=offline
# --model_name_or_path huggyllama/llama-7b facebook/opt-6.7b \
# --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' 'OPTDecoderLayer'
# cf. Turn on CPU offload for FSDP with --fsdp "full_shard auto_wrap offload"
torchrun --nproc_per_node=8 --master_port=8888 train.py \
    --model_name_or_path facebook/opt-6.7b \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir ./trained/fsdp/opt-6.7b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 6000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 8 \
    --skip_memory_metrics False \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' #\
    #| tee ./logs/fsdp/llama-7b_batch_4_ga_1.log