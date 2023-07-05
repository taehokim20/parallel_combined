from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, default_data_collator

from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT

import time
import math
from datasets import load_dataset
from itertools import chain
from torch.utils.data import DataLoader


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


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()

  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    # load_in_8bit=inference_args.load_in_8bit,
    torch_dtype=inference_args.inference_dtype,
    device_map="auto",
  )
  model.cuda()
  model.eval()

  if inference_args.override_checkpoint is not None:
    #### For Colo-models ####
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_args.model_name_or_path,
      use_fast=False,
      model_max_length=inference_args.model_max_length,
    )
    if tokenizer.pad_token is None:
      smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
      )
    tokenizer.add_special_tokens(
      {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
      }
    )
    ##########################
    print("Loading override checkpoint.")
    try:
      state_dict = torch.load(inference_args.override_checkpoint)
      model.load_state_dict(state_dict)
    except:
      raise Exception("Failed to load checkpoint")
    model.cuda()
    model.half()
    model.eval()

  # Get the datasets. Downloading and loading a dataset from the hub.
  dataset_name = "wikitext"
  dataset_config_name = "wikitext-2-raw-v1"
  raw_datasets = load_dataset(dataset_name, dataset_config_name)
  # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=False,
    model_max_length=inference_args.model_max_length,
  )

  column_names = raw_datasets["train"].column_names
  text_column_name = "text" if "text" in column_names else column_names[0]

  def tokenize_function(examples):
    return tokenizer(examples[text_column_name])
  
  tokenized_datasets = raw_datasets.map(
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

  eval_dataset = lm_datasets["validation"]

  # DataLoaders creation:
  eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=8)

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
    if step > 0:
      total_time += st_time
    print("Step {0} finished, Loss={1:.2f}, Step time={2:.2f}".format(step, loss.item(), st_time))
  
  losses = torch.cat(losses)
  losses = losses[:len(eval_dataset)]
  try:
    eval_loss = torch.mean(losses)
    perplexity = math.exp(eval_loss)
  except OverflowError:
    perplexity = float("inf")
  print("Total main - Perplexity={0:.2f}, Loss={1:.2f}, Total time={2:.2f}".format(perplexity, eval_loss.item(), total_time))


if __name__ == "__main__":
  inference()