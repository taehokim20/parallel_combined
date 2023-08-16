#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import warnings
warnings.simplefilter("ignore", UserWarning)

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import get_cosine_schedule_with_warmup

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device, print_rank_0
from colossalai.zero import ColoInitContext
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.cluster import DistCoordinator
from colossalai.context import ParallelMode, ParallelContext
from colossalai.pipeline.pipelinable import PipelinableContext
import torch.distributed as dist
from tqdm import tqdm
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.core import global_context as gpc
import sys
import time
from statistics import mean
import GPUtil
import psutil

from transformers import AutoConfig
# LLaMA
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# OPT
import transformers.models.opt.modeling_opt
from transformers.models.opt.modeling_opt import OPTForCausalLM


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator):
    print_rank_0('[3]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # 32077 MB
    print_rank_0('[3]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 18.82 GB
    torch.cuda.synchronize()
    model.train()
    losses = []
    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        for batch in pbar: # Iters - 2 GPUs: 13000(=52000/(2*2)), 4 GPUs: 3250(=52000/(4*4)), 8 GPUs: 812
            # Forward
            batch = move_to_cuda(batch, torch.cuda.current_device())
            outputs = model(use_cache=False, **batch)
            loss = outputs['loss']
            loss.requires_grad = True
            # Backward
            optimizer.zero_grad()
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()
            # Print batch loss
            # pbar.set_postfix({'loss': loss.item(), 'Memory usage': GPUtil.getGPUs()[0].memoryUsed})
            pbar.set_postfix({'loss': loss.item()}) 
            losses.append(loss.item())
    
    print_rank_0('Average loss of epoch {0}: {1:.2f}, Memory usage: {2}'.format(epoch + 1, mean(losses), 
                                                                                GPUtil.getGPUs()[0].memoryUsed))


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format- KB, MB, GB, TB and PB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def train():
    dp_degree=1
    tp_degree=1
    pp_degree=8
    # Launch ColossalAI
    colossalai.launch_from_torch(config=dict(parallel=dict(data=dp_degree, pipeline=pp_degree,
                                tensor=dict(size=tp_degree, mode='1d'))))
    
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print_rank_0('[0]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))  # 1421 MB
    print_rank_0('[0]Virtual total mem: {0}'.format(get_size(psutil.virtual_memory().total)))  # 1.10 TB
    print_rank_0('[0]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))  # 14.55 GB
    with PipelinableContext():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        ).to('cuda')

        model.gradient_checkpointing_enable()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)           
    
    print_rank_0('[1]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # 27189 MB // When sharding in with ColoInitContext 5545 MB
    print_rank_0('[1]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 14.61 GB

    # shard_pg = ProcessGroup(tp_degree=tp_degree)
    # init_colo_module(model, ComputeSpec(ComputePattern.TP1D), pg=shard_pg, recursive=True)

    # Set plugin
    booster_kwargs = {}
    # booster_kwargs['mixed_precision'] = 'fp16'
    plugin = GeminiPlugin(device=get_current_device(),
                          placement_policy='cuda',
                          precision='bf16',
                          pin_memory=False, #True,
                          strict_ddp_mode=False,
                          initial_scale=2**5)         ###

    config = {
        'batch_size': training_args.per_device_train_batch_size,
        'lr': training_args.learning_rate,
        'epochs': int(training_args.num_train_epochs),
        'warmup_ratio': training_args.warmup_ratio,
        'weight_decay': training_args.weight_decay,
    }

    dataloader = plugin.prepare_dataloader(data_module['train_dataset'], batch_size=config['batch_size'],
                                           shuffle=False, drop_last=True, collate_fn=data_module['data_collator'])

    # Set lr scheduler
    total_steps = len(dataloader) * config['epochs']
    num_warmup_steps = int(config['warmup_ratio'] * total_steps)
        
    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=(config['lr'] * world_size), weight_decay=0.0)

    # Set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * config['epochs']
    )

    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(config['epochs']):
       # 
       train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator)

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    output_dir = './trained/colossalai_models/pipeline_' + str(dist.get_rank()) + '.pt'
    booster.save_model(model, output_dir, tp_degree=tp_degree)
    logger.info(f"Saving model checkpoint to {output_dir}")


if __name__ == "__main__":
    train()
