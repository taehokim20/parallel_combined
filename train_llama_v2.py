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

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import get_cosine_schedule_with_warmup

# from titans.model.vit.vit import _create_vit_model

import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device, print_rank_0
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer, zero_optim_wrapper
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin, TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.pipeline.pipelinable import PipelinableContext
import torch.distributed as dist
from tqdm import tqdm
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.nn.parallel.layers import init_colo_module
from titans.utils import barrier_context
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
import sys
from statistics import mean

def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator):
    torch.cuda.synchronize()
    model.train()
    losses = []
    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        for batch in pbar: # Iters - 2 GPUs: 13000(=52000/(2*2)), 4 GPUs: 3250(=52000/(4*4)), 8 GPUs: 812
            # Forward
            optimizer.zero_grad()
            batch = move_to_cuda(batch, torch.cuda.current_device())           
            outputs = model(use_cache=False, **batch)
            loss = outputs['loss']
            # Backward
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()
            # Print batch loss
            pbar.set_postfix({'loss': loss.item()})
            losses.append(loss.item())
    
    print_rank_0('Average loss of epoch {0}: {1:.2f}'.format(epoch + 1, mean(losses)))


# def train_epoch_wo_booster(epoch, model, optimizer, lr_scheduler, dataloader, coordinator):
#     torch.cuda.synchronize()
#     model.train()
#     with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
#         for batch in pbar:
#             # Forward
#             optimizer.zero_grad()
#             batch = move_to_cuda(batch, torch.cuda.current_device())
#             # torch.distributed.broadcast(batch['input_ids'], src=0)
#             outputs = model(use_cache=False, **batch)
#             loss = outputs['loss']
#             # Backward
#             optimizer.backward(loss)
#             optimizer.step()
#             optimizer.zero_grad()
#             torch.cuda.synchronize()
#             lr_scheduler.step()
#             # Print batch loss
#             pbar.set_postfix({'loss': loss.item()})


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
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
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) # special_tokens_dict = {'pad_token': '[PAD]'}
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


def train():
    tp_degree=8
    ## for LLaMA models
    import transformers.models.llama.modeling_llama
    # Launch ColossalAI
    # colossalai.launch_from_torch(config={}, seed=0)
    colossalai.launch_from_torch(config=dict(parallel=dict(data=1, pipeline=1, tensor=dict(size=tp_degree, mode='1d'))))
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    shard_pg = ProcessGroup(tp_degree=tp_degree)
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # with PipelinableContext():
    # default_dist_spec = ShardSpec([-1], [tp_degree])  
        
    # with ColoInitContext(device=get_current_device(), default_dist_spec=default_dist_spec, default_pg=shard_pg):
    with ColoInitContext(device=get_current_device()):
        # from transformers import LlamaConfig
        # configuration = LlamaConfig(vocab_size=31999)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            # ignore_mismatched_sizes=True,
        )

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
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN    # "[PAD]" -> only this one is included
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN    # "</s>"
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN    # "<s>"
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN    # "<unk>"

        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=special_tokens_dict,
        #     tokenizer=tokenizer,
        #     model=model,
        # )
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)    

    ####################################################################
    compute_spec = ComputeSpec(ComputePattern.TP1D)
    init_colo_module(model, compute_spec, pg=shard_pg, recursive=True)

    # # print_rank_0(model)
    # for p in model.parameters():
    #     print(p)
    #     print(p.data.size())
    #     break

    # Set plugin
    booster_kwargs = {}
    # booster_kwargs['mixed_precision'] = 'fp16'
    plugin = GeminiPlugin(device=get_current_device(),
                          placement_policy='cuda',
                          precision='bf16',
                          pin_memory=False, #True,
                          strict_ddp_mode=False,
                          initial_scale=2**5)
    # plugin = TorchFSDPPlugin() # does not work with
    # plugin = LowLevelZeroPlugin(stage=2)

    config = {
        'batch_size': training_args.per_device_train_batch_size,
        'lr': training_args.learning_rate,
        'epochs': int(training_args.num_train_epochs),
        'warmup_ratio': training_args.warmup_ratio,
        'weight_decay': training_args.weight_decay,
    }

    dataloader = plugin.prepare_dataloader(data_module['train_dataset'],
                                           batch_size=config['batch_size'],
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=data_module['data_collator'])

    # Set lr scheduler
    total_steps = len(dataloader) * config['epochs']
    num_warmup_steps = int(config['warmup_ratio'] * total_steps)

    ############################ Case 1. Use Booster ############################
    # Set optimizer
    optimizer = HybridAdam(model.parameters(),
                           lr=(config['lr'] * world_size),
                           weight_decay=0.0)

    # Set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * config['epochs']
    )
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model=model,
                                                                  optimizer=optimizer,
                                                                  dataloader=dataloader,
                                                                  lr_scheduler=lr_scheduler)
    
    # for p in model.unwrap().module.parameters():
    #     print(p.data)
    #     print_rank_0(p.data.size())
    #     sys.exit()

    # Start finetuning
    logger.info(f"Start finetuning", ranks=[0])
    for epoch in range(config['epochs']):
       train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator)

    # Finish training and evaluate
    logger.info(f"Finish finetuning", ranks=[0])
    output_dir = './trained/shard_colo_llama-7b/shard_' + str(dist.get_rank()) + '.pt'
    booster.save_model(model, output_dir)
    logger.info(f"Saving model checkpoint to {output_dir}")

    ############################ Case 2. Use colossalai.initialize ############################
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    # criterion = CrossEntropyLoss(label_smoothing=0.1)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=len(dataloader) * config['epochs']
    # )
    # engine, train_dataloader, _, lr_scheduler = colossalai.initialize(model=model,
    #                                                                   optimizer=optimizer,
    #                                                                   criterion=criterion,
    #                                                                   train_dataloader=dataloader,
    #                                                                   lr_scheduler=lr_scheduler)

    # logger.info("Engine is built", ranks=[0])

    # for epoch in range(config['epochs']):
    #     #training
    #     engine.train()
    #     data_iter = iter(train_dataloader)

    #     if gpc.get_global_rank() == 0:
    #         description = 'Epoch {} / {}'.format(epoch, config['epochs'])
    #         progress = tqdm(range(len(train_dataloader)), desc=description)
    #     else:
    #         progress = range(len(train_dataloader))
    #     for _ in progress:
    #         engine.zero_grad()
    #         engine.execute_schedule(data_iter, return_output_label=False)
    #         engine.step()
    #         lr_scheduler.step()
    # gpc.destroy()


if __name__ == "__main__":
    train()
