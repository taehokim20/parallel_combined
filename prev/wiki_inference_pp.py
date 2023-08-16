from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, default_data_collator

from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT

import time
import math
import colossalai
from datasets import load_dataset
from itertools import chain
from torch.utils.data import DataLoader
from colossalai.pipeline.pipelinable import PipelinableContext
import torch.distributed as dist
import GPUtil
import psutil

from transformers import AutoConfig, AutoTokenizer, default_data_collator
# LLaMA
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# OPT
import transformers.models.opt.modeling_opt
from transformers.models.opt.modeling_opt import OPTForCausalLM


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format- KB, MB, GB, TB and PB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


@dataclass
class InferenceArguments:
  model_max_length: int = field(
    default=512,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  load_in_8bit: bool = field(
    default=False,
    metadata={"help": "Load the model in 8-bit mode."},
  )
  inference_dtype: torch.dtype = field(
    default=torch.float16,
    metadata={"help": "The dtype to use for inference."},
  )
  override_checkpoint: str = field(
    default=None,
    metadata={"help": "Name of the checkpoint file to override."},
  )


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()
  colossalai.launch_from_torch(config=dict(parallel=dict(data=1, pipeline=8, tensor=dict(size=1, mode='1d'))))

  with PipelinableContext():
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if 'llama-7b' in model_args.model_name_or_path:
      model = LlamaForCausalLM(model_config)
    elif 'opt-6.7b' in model_args.model_name_or_path:
      model = OPTForCausalLM(model_config)
    model.cuda()
    model.eval()
  
  state_dict = {}
  if 'llama-7b' in model_args.model_name_or_path:  
    from safetensors.torch import load_file  
    state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00001-of-00002.safetensors"))
    state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00002-of-00002.safetensors"))
  elif 'opt-6.7b' in model_args.model_name_or_path:
    state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00001-of-00002.bin"))
    state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00002-of-00002.bin"))

  for n, p in model.named_parameters():
    if 'opt-6.7b' in model_args.model_name_or_path:
      n = n.replace('model.', '')
    x = state_dict[n]
    p.data.copy_(x)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=False,
    model_max_length=inference_args.model_max_length,
  )

  if tokenizer.pad_token is None:
    # ### For size matching of Colossal-AI
    # tokenizer.pad_token = tokenizer.eos_token
    ### Other cases
    smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )
  ##########################
  if inference_args.override_checkpoint is not None:
    print("Loading override checkpoint.")
    try:
      # state_dict = torch.load(inference_args.override_checkpoint)
      state_dict = torch.load(inference_args.override_checkpoint + 'pipeline_' + str(dist.get_rank()) + '.pt')
      model.load_state_dict(state_dict)
    except:
      raise Exception("Failed to load checkpoint")
    model.cuda()
    model.half()
    model.eval()

  # Get the datasets. Downloading and loading a dataset from the hub.
  dataset_name = "wikitext"
  dataset_config_name = "wikitext-103-raw-v1" #"wikitext-2-raw-v1"
  raw_datasets = load_dataset(dataset_name, dataset_config_name)
  # # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
  # tokenizer = transformers.AutoTokenizer.from_pretrained(
  #   model_args.model_name_or_path,
  #   use_fast=False,
  #   model_max_length=inference_args.model_max_length,
  # )

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
  eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=8)

  losses = []
  loss = 0
  total_time = 0
  for step, batch in enumerate(eval_dataloader):
    if dist.get_rank() == 0:
      with torch.no_grad():
        batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()} 
        # torch.distributed.broadcast(batch['input_ids'], src=0)
        st_time = time.time()
        outputs = model(**batch)
        st_time = time.time() - st_time
        loss = outputs['loss'].unsqueeze(0)
        losses.append(loss)
      # if step > 0:
      total_time += st_time
      print("Step {0} finished, Loss={1:.2f}, Step time={2:.2f}".format(step, loss.item(), st_time))
    # else:
    #   with torch.no_grad():
    #     batch = {k: v.to(torch.cuda.current_device()) for k, v in batch.items()} 
    #     # torch.distributed.broadcast(batch['input_ids'], src=0)
    #     outputs = model(**batch)
  
  if dist.get_rank() == 0:
    losses = torch.cat(losses)
    losses = losses[:len(eval_dataset)]
    try:
      eval_loss = torch.mean(losses)
      perplexity = math.exp(eval_loss)
    except OverflowError:
      perplexity = float("inf")
    print("Total main - Perplexity={0:.2f}, Loss={1:.2f}, Total time={2:.2f}".format(perplexity, eval_loss.item(), total_time))
    print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
    print('Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))


if __name__ == "__main__":
  inference()