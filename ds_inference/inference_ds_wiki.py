from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, GenerationConfig, default_data_collator
import transformers
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline
from datasets import load_dataset
from itertools import chain
from torch.utils.data import DataLoader
import GPUtil
import psutil
import torch.distributed as dist

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name")
parser.add_argument("--checkpoint_path", required=False, default=None, type=str, help="model checkpoint path")
parser.add_argument("--save_mp_checkpoint_path", required=False, default=None, type=str, help="save-path to store the new model checkpoint")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--dtype", default="float16", type=str, choices=["float32", "float16", "int8"], help="data-type")
# parser.add_argument("--ds_inference", action='store_true', help="enable ds-inference")
parser.add_argument("--ds_inference", action='store_false', help="enable ds-inference")
parser.add_argument("--use_kernel", action='store_true', help="enable kernel-injection")
parser.add_argument("--replace_method", required=False, default='', type=str, help="replace method['', 'auto']")
parser.add_argument("--max_tokens", default=1024, type=int, help="maximum tokens used for the text-generation KV-cache")
parser.add_argument("--max_new_tokens", default=50, type=int, help="maximum new tokens to generate")
parser.add_argument("--greedy", action='store_true', help="greedy generation mode")
parser.add_argument("--use_meta_tensor", action='store_true', help="use the meta tensors to initialize model")
parser.add_argument("--use_cache", default=True, type=bool, help="use cache for generation")
parser.add_argument("--test_performance", action='store_true', help="enable latency, bandwidth, and throughout testing")
parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")), help="local rank")
parser.add_argument("--world_size", type=int, default=int(os.getenv("WORLD_SIZE", "1")), help="world_size")
args = parser.parse_args()


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format- KB, MB, GB, TB and PB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


if not args.ds_inference and args.world_size > 1:
    raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

data_type = getattr(torch, args.dtype)

# pipe = DSPipeline(model_name=args.name,
#                   dtype=data_type,
#                   is_meta=args.use_meta_tensor, # False
#                   device=args.local_rank,
#                   checkpoint_path=args.checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(args.name, torch_dtype=data_type)

print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
# pipe.model.cuda()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.name, use_fast=False, model_max_length=512,)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if args.ds_inference:
    model = deepspeed.init_inference(model,
                                    dtype=data_type,
                                    mp_size=args.world_size,
                                    replace_with_kernel_inject=args.use_kernel, # False
                                    replace_method=args.replace_method, # ''
                                    max_tokens=args.max_tokens, # 1024
                                    save_mp_checkpoint_path=args.save_mp_checkpoint_path # False
                                    )

model.cuda()
model.eval()

generation_config = GenerationConfig.from_pretrained(args.name)

# Get the datasets. Downloading and loading a dataset from the hub.
dataset_name = "wikitext"
dataset_config_name = "wikitext-103-raw-v1" #"wikitext-2-raw-v1"
raw_datasets = load_dataset(dataset_name, dataset_config_name)

# column_names = raw_datasets["train"].column_names
column_names = raw_datasets["test"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_datasets = raw_datasets["test"].map(  # raw_datasets.map
tokenize_function, batched=True, num_proc=None, remove_columns=column_names,
load_from_cache_file=True, desc="Running tokenizer on dataset",
)

block_size = min(512, tokenizer.model_max_length)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)
        ] for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts, batched=True, num_proc=None, load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

# eval_dataset = lm_datasets["validation"]
# eval_dataset = lm_datasets["test"]
eval_dataset = lm_datasets

# DataLoaders creation:
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

losses = []
loss = 0
total_time = 0
for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        batch = {k: v.cuda() for k, v in batch.items()} 
        # torch.distributed.broadcast(batch['input_ids'], src=0)
        st_time = time.time()
        outputs = model(**batch)
        st_time = time.time() - st_time
        loss = outputs['loss'].unsqueeze(0)
        losses.append(loss)
    if dist.get_rank() == 0:
        total_time += st_time
        print("Step {0} finished, Loss={1:.2f}, Step time={2:.2f}".format(step, loss.item(), st_time))
        # print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))

losses = torch.cat(losses)
losses = losses[:len(eval_dataset)]
try:
    eval_loss = torch.mean(losses)
    perplexity = math.exp(eval_loss)
except OverflowError:
    perplexity = float("inf")

if dist.get_rank() == 0:
    print("Total main - Perplexity={0:.2f}, Loss={1:.2f}, Total time={2:.2f}".format(perplexity, eval_loss.item(), total_time))
    print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
    print('Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))
