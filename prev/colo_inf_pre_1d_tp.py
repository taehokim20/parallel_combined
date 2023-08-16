from dataclasses import dataclass, field

import torch
import transformers
from transformers import GenerationConfig
import colossalai
from colossalai.zero import ColoInitContext
from colossalai.utils import get_current_device, print_rank_0
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers, get_dist_logger
import time
import GPUtil
import psutil

from transformers import AutoConfig
# LLaMA
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# OPT
import transformers.models.opt.modeling_opt
from transformers.models.opt.modeling_opt import OPTForCausalLM

from train import ModelArguments, PROMPT_DICT
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode


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
  tp_dim: int = field(
    default=0,
    metadata={"help": "0: 1D TP, 1: 2D TP, 2: 2.5D TP, 3: 3D TP"},
  )
  world_size: int = field(default=1)


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()
  dp_degree=1
  world_size=inference_args.world_size
  norm_sharding=False
  # tp_dim=1 # 0: 1D TP, 1: 2D TP, 2: 2.5D TP, 3: 3D TP
  tp_dim=inference_args.tp_dim
  mode=['1d', '2d', '2.5d', '3d']
  tp_degree=[[world_size], [2, 2], [2, 2, 2], [2, 2, 2]]
  dims_e=[[-1], [0, -1], [0, 0, -1], [0, 0, -1]]
  dims_l=[[-1], [0, -1], [0, 0, -1], [0, 0, -1]]
  # Compute Pattern
  compute_spec = [ComputeSpec(ComputePattern.TP1D), ComputeSpec(ComputePattern.TP2D),
                  ComputeSpec(ComputePattern.TP2P5D), ComputeSpec(ComputePattern.TP3D)]
  disable_existing_loggers()
  if mode == '2.5d':
    colossalai.launch_from_torch(config=dict(parallel=dict(data=dp_degree, pipeline=1,
                                tensor=dict(size=world_size, mode=mode[tp_dim], depth=2))))
  else:
    colossalai.launch_from_torch(config=dict(parallel=dict(data=dp_degree, pipeline=1, 
                                tensor=dict(size=world_size, mode=mode[tp_dim]))))
  # parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  # model_args, inference_args = parser.parse_args_into_dataclasses()
  logger = get_dist_logger()
  shard_pg = ProcessGroup(tp_degree=world_size)
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
    # model.cuda()
    # model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_args.model_name_or_path,
      use_fast=False,
      model_max_length=inference_args.model_max_length,
    )

    if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token

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
  
  print_rank_0('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
  print_rank_0('Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))

  model.cuda()
  # model.half()
  model.eval()

  # generation_config = GenerationConfig(
  #   temperature=0.1,
  #   top_p=0.75,
  #   num_beams=4,
  # )
  generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)

  total_time = 0
  for instruction in [
    "Tell me about alpacas.",
    "Tell me about the president of Mexico in 2019.",
    "Tell me about the king of France in 2019.",
    "List all Canadian provinces in alphabetical order.",
    "Write a Python program that prints the first 10 Fibonacci numbers.",
    "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    "Tell me five words that rhyme with 'shock'.",
    "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500.",
  ]:
    inputs = tokenizer(generate_prompt(instruction, None), return_tensors="pt")
    dist.broadcast(inputs['input_ids'].cuda(), src=0)
    if dist.get_rank() == 0:
      # logger.info("Instruction: {}".format(instruction), ranks=[0])
      print("Instruction: {}".format(instruction))
      st_time = time.time()
      outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                              generation_config=generation_config,
                              max_new_tokens=inference_args.model_max_length,
                              return_dict_in_generate=True,
                              output_scores=True)
      st_time = time.time() - st_time
      input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
      generated_tokens = outputs.sequences[:, input_length:]

      total_time += st_time
      print("Response: {}".format(tokenizer.decode(generated_tokens[0])))
      print('Spent time: {0:.2f}'.format(st_time))
      print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
      print('Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))
      print()
    else:
      model.generate(input_ids=inputs["input_ids"].cuda(),
                              generation_config=generation_config,
                              max_new_tokens=inference_args.model_max_length,
                              return_dict_in_generate=True,
                              output_scores=True)
  
  if dist.get_rank() == 0:
    print('Total time: {0:.2f}'.format(total_time))


if __name__ == "__main__":
  inference()