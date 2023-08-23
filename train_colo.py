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
from colossalai.context import ParallelMode
import torch.distributed as dist
from tqdm import tqdm
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.core import global_context as gpc
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


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator, tp_dim=0, 
                batch_size=1, tp_degree=None, dims=None):
    print_rank_0('[3]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # 32077 MB / 8243 MB
    print_rank_0('[3]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 18.82 GB
    # print_rank_0('[3]Max allocated GPU mem: {0}'.format(get_size(torch.cuda.max_memory_allocated())))
    torch.cuda.synchronize()
    model.train()
    losses = []
    # for step, inputs in enumerate(dataloader):
    #     # if inputs["input_ids"].size()[1] == 512:
    #         # print_rank_0(step)
    #         # print_rank_0(inputs)
    #         # print_rank_0(inputs["input_ids"].size())
    #         # print_rank_0(inputs['labels'].size())
    #         # print_rank_0(inputs['attention_mask'].size())
    #         # print_rank_0(" ")
    #     # if inputs["input_ids"].size()[1] == 512:
    #     #     max_seq_len = inputs["input_ids"].size()[1]
    #     #     batch = move_to_cuda(inputs, torch.cuda.current_device())
    #     #     outputs = model(use_cache=False, **batch)
    #     #     loss = outputs['loss']
    #     #     booster.backward(loss, optimizer)
    #     #     optimizer.step()
    #     #     optimizer.zero_grad()
    #     #     lr_scheduler.step()
    #     #     if dist.get_rank() == 0:
    #     #         with open('temp.txt', 'a') as f:
    #     #             f.write('{0}\n'.format(GPUtil.getGPUs()[0].memoryUsed))
    #     #     torch.cuda.empty_cache()
    #     #     # print_rank_0('[Initial step] step: {0}, GPU mem: {1}'.format(step, GPUtil.getGPUs()[0].memoryUsed))


    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        step=1
        for batch in pbar:
            # Forward
            batch = move_to_cuda(batch, torch.cuda.current_device())
            if batch["input_ids"].size()[1] == 512:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            outputs = model(use_cache=False, **batch)
            loss = outputs['loss']
            # Backward
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            # Print batch loss
            # pbar.set_postfix({'loss': loss.item(), 'Memory usage': GPUtil.getGPUs()[0].memoryUsed})
            pbar.set_postfix({'loss': loss.item()}) 
            losses.append(loss.item())
            if dist.get_rank() == 0:
                with open('temp.txt', 'a') as f:
                    f.write('{0}\n'.format(GPUtil.getGPUs()[0].memoryUsed))
            step += 1
            # if dist.get_rank() == 0:
            #     # with open('llama-7b_seq.txt', 'a') as f:
            #     #     f.write('{0}\n'.format(batch["input_ids"].size()[1]))
            #     with open('opt-13b_mem_5per.txt', 'a') as f:
            #         f.write('{0}\n'.format(GPUtil.getGPUs()[0].memoryUsed))
                
    
    print_rank_0('Average loss of epoch {0}: {1:.2f}, Memory usage: {2}'.format(epoch + 1, mean(losses), 
                                                                                GPUtil.getGPUs()[0].memoryUsed))
    # print_rank_0('torch.cuda.max_memory_allocated: {0}'.format(get_size(torch.cuda.max_memory_allocated())))


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
    tp_size=8
    norm_sharding=False
    tp_dim=0 # 0: 1D TP, 1: 2D TP, 2: 2.5D TP, 3: 3D TP
    mode=['1d', '2d', '2.5d', '3d']
    tp_degree=[[tp_size], [2, 2], [2, 2, 2], [2, 2, 2]]
    dims_e=[[-1], [0, -1], [0, 0, -1], [0, 0, -1]]
    dims_l=[[-1], [0, -1], [0, 0, -1], [0, 0, -1]]
    # Compute Pattern
    compute_spec = [ComputeSpec(ComputePattern.TP1D), ComputeSpec(ComputePattern.TP2D),
                    ComputeSpec(ComputePattern.TP2P5D), ComputeSpec(ComputePattern.TP3D)]
    # Launch ColossalAI
    # # Data Parallelism
    # colossalai.launch_from_torch(config={})
    # Tensor Parallelism
    if mode == '2.5d':
        colossalai.launch_from_torch(config=dict(parallel=dict(data=1, pipeline=1,
                                    tensor=dict(size=tp_size, mode=mode[tp_dim], depth=2))))
    else:
        colossalai.launch_from_torch(config=dict(parallel=dict(data=1, pipeline=1, 
                                    tensor=dict(size=tp_size, mode=mode[tp_dim]))))
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    print_rank_0('[0]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))  # 3 MB / 1421 MB
    # print_rank_0('[0]Virtual total mem: {0}'.format(get_size(psutil.virtual_memory().total)))  # 1.10 TB
    print_rank_0('[0]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))  # 8.65 GB / 14.55 GB
    print_rank_0('[0]Max allocated GPU mem: {0}'.format(get_size(torch.cuda.max_memory_allocated())))

    shard_pg = ProcessGroup(tp_degree=tp_size)
    embedding_dist_spec = ShardSpec(dims_e[tp_dim], tp_degree[tp_dim])
    linear_dist_spec = ShardSpec(dims_l[tp_dim], tp_degree[tp_dim])

    with ColoInitContext(device=get_current_device(), embedding_dist_spec=embedding_dist_spec, 
                         linear_dist_spec=linear_dist_spec, default_pg=shard_pg,
                         model_name=model_args.model_name_or_path, norm_sharding=norm_sharding):
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if 'llama' in model_args.model_name_or_path:
            model = LlamaForCausalLM(model_config)
        elif 'opt' in model_args.model_name_or_path:
            model = OPTForCausalLM(model_config)

        model.gradient_checkpointing_enable()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)           

    state_dict = {}
    if 'llama-7b' in model_args.model_name_or_path:  
        from safetensors.torch import load_file  
        state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00001-of-00002.safetensors"))
        state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00002-of-00002.safetensors"))
    elif 'llama-13b' in model_args.model_name_or_path:
        from safetensors.torch import load_file  
        state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba/model-00001-of-00003.safetensors"))
        state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba/model-00002-of-00003.safetensors"))
        state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-13b/snapshots/bf57045473f207bb1de1ed035ace226f4d9f9bba/model-00003-of-00003.safetensors"))
    elif 'opt-6.7b' in model_args.model_name_or_path:
        state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00001-of-00002.bin"))
        state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0/pytorch_model-00002-of-00002.bin"))
    elif 'opt-13b' in model_args.model_name_or_path:
        state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/pytorch_model-00001-of-00003.bin"))
        state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/pytorch_model-00002-of-00003.bin"))
        state_dict.update(torch.load("/home/ubuntu/.cache/huggingface/hub/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/pytorch_model-00003-of-00003.bin"))
    
    for n, p in model.named_parameters():
        if 'opt' in model_args.model_name_or_path: # opt-6.7b, opt-13b
            n = n.replace('model.', '')
        x = state_dict[n]  
        if not 'norm' in n and not 'bias' in n:
            p.compute_spec = compute_spec[tp_dim]
            if mode[tp_dim] == '1d':
                x = x.chunk(tp_degree[tp_dim][0], dim=dims_l[tp_dim][0])
                x = x[dist.get_rank() % tp_degree[tp_dim][0]]
            elif mode[tp_dim] == '2d':
                x = x.chunk(tp_degree[tp_dim][0], dim=dims_l[tp_dim][0])[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)]
                x = x.chunk(tp_degree[tp_dim][1], dim=dims_l[tp_dim][1])[gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)]
            elif mode[tp_dim] == '2.5d':
                x = x.chunk(tp_degree[tp_dim][0], dim=dims_l[tp_dim][0])[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)]
                x = x.chunk(tp_degree[tp_dim][1], dim=dims_l[tp_dim][1])[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)]
                x = x.chunk(tp_degree[tp_dim][2], dim=dims_l[tp_dim][2])[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)]
            elif mode[tp_dim] == '3d':
                x = x.chunk(tp_degree[tp_dim][0], dim=dims_l[tp_dim][0])[gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)]
                x = x.chunk(tp_degree[tp_dim][1], dim=dims_l[tp_dim][1])[gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)]
                x = x.chunk(tp_degree[tp_dim][2], dim=dims_l[tp_dim][2])[gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)]
        p.data.copy_(x)
    
    print_rank_0('[1]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # 27189 MB // When sharding in with ColoInitContext 5545 MB -> 4957 MB
    print_rank_0('[1]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 14.61 GB
    print_rank_0('[1]Max allocated GPU mem: {0}'.format(get_size(torch.cuda.max_memory_allocated())))

    # Set plugin
    booster_kwargs = {}
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
    # optimizer = HybridAdam(model.parameters(), lr=(config['lr'] * world_size), weight_decay=0.0)
    # optimizer = HybridAdam(model.parameters(), lr=(config['lr'] * int(world_size / tp_size)), weight_decay=0.0)
    optimizer = HybridAdam(model.parameters(), lr=config['lr'], weight_decay=0.0)

    # Set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * config['epochs']
    )

    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)
    print_rank_0('[2]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))
    print_rank_0('[2]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))
    print_rank_0('[2]Max allocated GPU mem: {0}'.format(get_size(torch.cuda.max_memory_allocated())))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(config['epochs']):
       train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator, 
                   tp_dim=tp_dim, batch_size=config['batch_size']) #tp_degree=tp_degree, dims=dims_l) 
       torch.cuda.empty_cache()

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    output_dir = training_args.output_dir + '/shard_' + str(dist.get_rank()) + '.pt'
    booster.save_model(model, output_dir, tp_degree=world_size)
    logger.info(f"Saving model checkpoint to {output_dir}")


if __name__ == "__main__":
    train()
