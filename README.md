# parallel_combined

## [Colossal-AI](https://github.com/hpcaitech/ColossalAI)
* For fine-tuning, run_colo.sh
  - Running files:
    - Alpaca dataset -> (tensor parallelism) train_colo.py / (data parallelism) train_llama.py / (pipeline parallelism) train_colo_pipeline.py
    - Wikitext dataset -> (tensor parallelism) train_colo_wiki.py
  - Models: huggyllama/llama-7b, huggyllama/llama-13b, facebook/opt-6.7b, facebook/opt-13b
* For Alpaca inference,
  - Using the sharded models: ```torchrun --nproc_per_node 8 colo_inference.py --model_name_or_path huggyllama/llama-7b --override_checkpoint /path/to/the/shard_x.pt/files/directory/```
  - Using a unsharded single model: ```CUDA_VISIBLE_DEVICES=0 python inference.py --model_name_or_path huggyllama/llama-7b --override_checkpoint [file]```
* For WikiText inference,
  - Using the sharded models: ```torchrun --nproc_per_node 8 wiki_inference_tp.py --model_name_or_path facebook/opt-6.7b --override_checkpoint /path/to/the shard_x.pt/files/directory/```

## [DeepSpeed](https://github.com/microsoft/DeepSpeed) / [PyTorch-FSDP](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp)
* For fine-tuning, (DeepSpeed) run_deepspeed.sh / (FSDP) run_fsdp.sh
  - Running files:
    - Alpaca dataset -> (data parallelism) train.py
    - WikiText dataset -> (data parallelism) train_wiki.py
  - Models: huggyllama/llama-7b, huggyllama/llama-13b, facebook/opt-6.7b, facebook/opt-13b
- For Alpaca inference,
  - ```CUDA_VISIBLE_DEVICES=0 python inference.py --model_name_or_path /path/to/the/pretrained/files/directory/```
  - Using the DeepSpeed inference tensor parallelism: ```torchrun --nproc_per_node 8 ./ds_inference/inference_ds.py --name huggyllama/llama-7b```
- For WikiText inference,
  - ```CUDA_VISIBLE_DEVICES=0 python wiki_inference.py --model_name_or_path /path/to/the/pretrained/files/directory/```
  - Using the DeepSpeed inference tensor parallelism: ```torchrun --nproc_per_node 8 ./ds_inference/inference_ds_wiki.py --name facebook/opt-6.7b --batch_size 8```
 
## Setup
* Environment: AWS EC2 P4d instance (8 GPUs, Each GPU has 40960 MB memory)
* colossalai version: 0.3.0 (Jun 15 2023 commit: d4fb7bfda7a2da5480e1187e8d3e40884b42ba11)
* transformers version: 4.29.2
