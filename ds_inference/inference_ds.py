from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
import transformers
import deepspeed
import math
import os
import torch
import time
from utils import DSPipeline
from deepspeed.runtime.utils import see_memory_usage
from dataclasses import dataclass, field
from typing import Dict
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


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format- KB, MB, GB, TB and PB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def generate_prompt(instruction, input=None):
    if input:
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
  

# def print_perf_stats(latency_set, config, warmup=3):
#     # trim warmup queries
#     latency_set = list(latency_set)
#     latency_set = latency_set[warmup:]
#     count = len(latency_set)

#     if count > 0:
#         latency_set.sort()
#         avg = sum(latency_set) / count
#         num_layers = getattr(config, "num_layers", config.num_hidden_layers)
#         num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
#         if args.dtype == "float16":
#             num_bytes = 2
#         elif args.dtype == "float32":
#             num_bytes = 4
#         else:
#             num_bytes = 1
#         print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
#         print("Avg BW: {0:8.2f} GB/s".format(1/avg * num_parameters * num_bytes / 1e9))
#         print("Avg flops: {0:8.2f} TFlops/s".format(1/avg * num_parameters * num_bytes * args.batch_size / 1e12))

if not args.ds_inference and args.world_size > 1:
    raise RuntimeError("Only `--num_gpus 1` supported for non-DeepSpeed uses")

data_type = getattr(torch, args.dtype)

if args.local_rank == 0:
    see_memory_usage("before init", True)

t0 = time.time()
# pipe = DSPipeline(model_name=args.name,
#                   dtype=data_type,
#                   is_meta=args.use_meta_tensor, # False
#                   device=args.local_rank,
#                   checkpoint_path=args.checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(args.name, torch_dtype=data_type)

print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 

tokenizer = transformers.AutoTokenizer.from_pretrained(args.name, use_fast=False, model_max_length=512,)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=dict(pad_token="[PAD]"),
    #     tokenizer=tokenizer,
    #     model=pipe,
    # )

ds_kwargs = dict()

if args.ds_inference:
    # model = deepspeed.init_inference(model,
    #                                 dtype=data_type,
    #                                 mp_size=args.world_size,
    #                                 replace_with_kernel_inject=args.use_kernel, # False
    #                                 replace_method=args.replace_method, # ''
    #                                 max_tokens=args.max_tokens, # 1024
    #                                 save_mp_checkpoint_path=args.save_mp_checkpoint_path,  # False
    #                                 **ds_kwargs
    #                                 )
    infer_config = dict(tensor_parallel={'tp_size': args.world_size}, dtype=data_type, 
                    replace_with_kernel_inject=args.use_kernel)
    model = deepspeed.init_inference(model, config=infer_config)
# if args.local_rank == 0:
#     see_memory_usage("after init_inference", True)

model.cuda()
model.eval()

# generation_config = GenerationConfig(
#     temperature=0.1,
#     top_p=0.75,
#     num_beams=4,
#   )
generation_config = GenerationConfig.from_pretrained(args.name)

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
    if dist.get_rank() == 0:
        print("Instruction:", instruction)
    inputs = tokenizer(generate_prompt(instruction, None), return_tensors="pt")
    st_time = time.time()
    outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                generation_config=generation_config,
                                max_new_tokens=512,
                                return_dict_in_generate=True,
                                output_scores=True)
    st_time = time.time() - st_time
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    total_time += st_time
    if dist.get_rank() == 0:
        print("Response: ", tokenizer.decode(generated_tokens[0]))
        print('Spent time: {0:.2f}'.format(st_time))
        print('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) 
        print('Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used)))
        print()

if dist.get_rank() == 0:
    print('Total time: {0:.2f}'.format(total_time))
# input_sentences = [
#          "DeepSpeed is a machine learning framework",
#          "He is working on",
#          "He has a",
#          "He got all",
#          "Everyone is happy and I can",
#          "The new movie that got Oscar this year",
#          "In the far far distance from our galaxy,",
#          "Peace is the only way"
# ]

# if args.batch_size > len(input_sentences):
#     # dynamically extend to support larger bs by repetition
#     input_sentences *= math.ceil(args.batch_size / len(input_sentences))

# inputs = input_sentences[:args.batch_size]

# iters = 30 if args.test_performance else 2 #warmup
# times = []
# for i in range(iters):
#     torch.cuda.synchronize()
#     start = time.time()
#     outputs = pipe(inputs,
#             num_tokens=args.max_new_tokens,
#             do_sample=(not args.greedy))
#     torch.cuda.synchronize()
#     end = time.time()
#     times.append(end - start)
# print(f"generation time is {times[1]} sec")

# if args.local_rank == 0:
#     for i, o in zip(inputs, outputs):
#         print(f"\nin={i}\nout={o}\n{'-'*60}")
#     if args.test_performance:
#         print_perf_stats(map(lambda t: t / args.max_new_tokens, times), pipe.model.config)
