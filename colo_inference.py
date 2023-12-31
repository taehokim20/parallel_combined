from dataclasses import dataclass, field

import torch
import transformers
from transformers import GenerationConfig
import colossalai
from colossalai.zero import ColoInitContext
from colossalai.utils import get_current_device
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


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  tp_degree=8
  dp_degree=1
  dims=-1 # 0: by row (bs=8, peak_mem=28487 MB), -1: by col (bs=8, peak_mem=24855 MB)
  disable_existing_loggers()
  colossalai.launch_from_torch(config=dict(parallel=dict(data=dp_degree, pipeline=1, 
                                                           tensor=dict(size=tp_degree, mode='1d'))))
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()
  logger = get_dist_logger()
  shard_pg = ProcessGroup(tp_degree=tp_degree)
  embedding_dist_spec = ShardSpec([-1], [tp_degree])
  linear_dist_spec = ShardSpec([-1], [tp_degree])
        
  with ColoInitContext(device=get_current_device(), embedding_dist_spec=embedding_dist_spec, 
                       linear_dist_spec=linear_dist_spec, default_pg=shard_pg,
                       model_name=model_args.model_name_or_path):
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

    for n, p in model.named_parameters():       
      if not 'norm' in n and not 'bias' in n:
        p.compute_spec = ComputeSpec(ComputePattern.TP1D)
  
  print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
  print('Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))

  if inference_args.override_checkpoint is not None:
    logger.info("Loading override checkpoint.", ranks=[0])
    try:
      state_dict = torch.load(inference_args.override_checkpoint + 'shard_' + str(dist.get_rank()) + '.pt')
      model.load_state_dict(state_dict)
    except:
      raise Exception("Failed to load checkpoint")
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