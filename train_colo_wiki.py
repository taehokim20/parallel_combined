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

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
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
import sys
import time
from statistics import mean
import GPUtil
import psutil

from itertools import chain
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from train import smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from transformers import AutoConfig, AutoTokenizer, default_data_collator
# LLaMA
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# OPT
import transformers.models.opt.modeling_opt
from transformers.models.opt.modeling_opt import OPTForCausalLM


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = utils.jload(data_path)

#         logging.warning("Formatting inputs...")
#         prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
#         sources = [
#             prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
#             for example in list_data_dict
#         ]
#         targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

#         logging.warning("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, raw_dataset) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = SupervisedDataset(tokenizer=tokenizer, raw_dataset=raw_dataset)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator):
    print_rank_0('[3]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # 32077 MB
    print_rank_0('[3]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 18.82 GB
    torch.cuda.synchronize()
    model.train()
    losses = []
    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        for batch in pbar:
            # Forward
            batch = move_to_cuda(batch, torch.cuda.current_device())
            outputs = model(use_cache=False, **batch)
            loss = outputs['loss']
            # Backward
            optimizer.zero_grad()
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()
            # Print batch loss
            # pbar.set_postfix({'loss': loss.item(), 'Memory usage': GPUtil.getGPUs()[0].memoryUsed})
            pbar.set_postfix({'loss': loss.item()}) 
            losses.append(loss.item())
            if dist.get_rank() == 0:
                with open('all_opt-6_7b_mem.txt', 'a') as f:
                    f.write('{0}\n'.format(GPUtil.getGPUs()[0].memoryUsed))
    
    print_rank_0('Average loss of epoch {0}: {1:.2f}, Memory usage: {2}'.format(epoch + 1, mean(losses), 
                                                                                GPUtil.getGPUs()[0].memoryUsed))


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
    
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, _, training_args = parser.parse_args_into_dataclasses()

    config = {
        'batch_size': training_args.per_device_train_batch_size,
        'lr': training_args.learning_rate,
        'epochs': int(training_args.num_train_epochs),
        'warmup_ratio': training_args.warmup_ratio,
        'weight_decay': training_args.weight_decay,
    }

    shard_pg = ProcessGroup(tp_degree=tp_size)
    embedding_dist_spec = ShardSpec(dims_e[tp_dim], tp_degree[tp_dim])
    linear_dist_spec = ShardSpec(dims_l[tp_dim], tp_degree[tp_dim])
    
    print_rank_0('[0]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))  # 1421 MB
    print_rank_0('[0]Virtual total mem: {0}'.format(get_size(psutil.virtual_memory().total)))  # 1.10 TB
    print_rank_0('[0]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))  # 14.55 GB
    with ColoInitContext(device=get_current_device(), embedding_dist_spec=embedding_dist_spec, 
                         linear_dist_spec=linear_dist_spec, default_pg=shard_pg,
                         model_name=model_args.model_name_or_path, norm_sharding=norm_sharding):
        model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        if 'llama-7b' in model_args.model_name_or_path:
            model = LlamaForCausalLM(model_config)
        elif 'opt-6.7b' in model_args.model_name_or_path:
            model = OPTForCausalLM(model_config)

        model.gradient_checkpointing_enable()           

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
        if norm_sharding or not 'norm' in n and not 'bias' in n:
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
    
    print_rank_0('[1]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # 27189 MB // When sharding in with ColoInitContext 5545 MB
    print_rank_0('[1]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 14.61 GB

    # Set plugin
    booster_kwargs = {}
    plugin = GeminiPlugin(device=get_current_device(),
                          placement_policy='cuda',
                          precision='bf16',
                          pin_memory=False, #True,
                          strict_ddp_mode=False,
                          initial_scale=2**5)         ###

    # Get the datasets. Downloading and loading a dataset from the hub.
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-103-raw-v1" #"wikitext-2-raw-v1"
    raw_datasets = load_dataset(dataset_name, dataset_config_name)
    # self.input_ids = data_dict["input_ids"]
    #     self.labels = data_dict["labels"]
    # print_rank_0(raw_datasets)
    # sys.exit()
    
    # Prepare tokenizer and dataloader
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocessing the datasets.
    # First we tokenize all the texts.
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

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, num_proc=None, load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=config['batch_size'])
    # dataloader = plugin.prepare_dataloader(train_dataset, batch_size=config['batch_size'],
    #                                        shuffle=False, drop_last=True, collate_fn=default_data_collator)

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

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(config['epochs']):
       train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator) 

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    output_dir = training_args.output_dir + '/shard_' + str(dist.get_rank()) + '.pt'
    booster.save_model(model, output_dir, tp_degree=world_size)
    logger.info(f"Saving model checkpoint to {output_dir}")


if __name__ == "__main__":
    train()
