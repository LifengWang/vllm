# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

import vllm.envs as envs
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]
) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={"latency": results["latencies"]},
        extra_info={k: results[k] for k in ["avg_latency", "percentiles"]},
    )
    if pt_records:
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)

    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len + args.output_len
    ), (
        "Please ensure that max_model_len is greater than"
        " the sum of input_len and output_len."
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    print(sampling_params)
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts: list[PromptType] = [
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
    ]

    def llm_generate():
        if not args.use_beam_search:
            llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        else:
            llm.beam_search(
                dummy_prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
            )

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            llm.start_profile()
            llm_generate()
            llm.stop_profile()
        else:
            start_time = time.perf_counter()
            outputs = llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency, outputs

    def run_to_completion_benchmark(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(dummy_prompts,
                             sampling_params=sampling_params,
                             use_tqdm=False)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            output_matrix = llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time

            # calculate token latency
            # RequestMetrics(arrival_time=1716261373.3672826, last_token_time=1716261373.3672826, first_scheduled_time=1716261373.367539, first_token_time=1716261374.4200041, time_in_queue=0.0002562999725341797, finished_time=1716261399.178795)
            first_token_latency, next_token_latency = [], []
            for i in range(len(output_matrix)):
                first_token_latency.append(output_matrix[i].metrics.first_token_time - output_matrix[i].metrics.first_scheduled_time)
                next_token_latency.append((output_matrix[i].metrics.finished_time - output_matrix[i].metrics.first_token_time)/(args.output_len-1))

            return latency, first_token_latency, next_token_latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = envs.VLLM_TORCH_PROFILER_DIR
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    # latencies = []
    latencies, first_token_latencies, next_token_latencies = [], [], []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        # latencies.append(run_to_completion(profile_dir=None))
    # latencies = np.array(latencies)
        result_latency, result_first_token, result_next_token = run_to_completion_benchmark(profile_dir=None)
        latencies.append(result_latency)
        first_token_latencies.append(result_first_token)
        next_token_latencies.append(result_next_token)
    first_token_latencies = np.array(first_token_latencies)
    next_token_latencies = np.array(next_token_latencies)
    # percentages = [10, 25, 50, 75, 90, 99]
    # percentiles = np.percentile(latencies, percentages)
    # latencies_percentiles = np.percentile(latencies, percentages)
    # first_token_latencies_percentiles = np.percentile(first_token_latencies, percentages)
    # next_token_latencies_percentiles = np.percentile(next_token_latencies, percentages)

    print(f'Avg First token latency: {np.mean(first_token_latencies)} seconds')
    print(f'Avg next token latency: {np.mean(next_token_latencies)} seconds')
    print(f"Avg latency: {np.mean(latencies)} seconds")
    # for percentage, percentile in zip(percentages, percentiles):
    #     print(f"{percentage}% percentile latency: {percentile} seconds")
    # for percentage, percentile in zip(percentages, latencies_percentiles):
    #     print(f'{percentage}% latencies percentile latency: {percentile} seconds')
    # for percentage, percentile in zip(percentages, first_token_latencies_percentiles):
    #     print(f'{percentage}% first token percentile latency: {percentile} seconds')
    # for percentage, percentile in zip(percentages, next_token_latencies_percentiles):
    #     print(f'{percentage}% next token percentile latency: {percentile} seconds')

    # # Output JSON results if specified
    # if args.output_json:
    #     results = {
    #         "avg_latency": np.mean(latencies),
    #         "latencies": latencies.tolist(),
    #         "percentiles": dict(zip(percentages, percentiles.tolist())),
    #     }
    #     with open(args.output_json, "w") as f:
    #         json.dump(results, f, indent=4)
    #     save_to_pytorch_benchmark_format(args, results)


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion."
    )
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of generated sequences per prompt.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30, help="Number of iterations to run."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="profile the generation process of a single batch",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the latency results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )

    parser = EngineArgs.add_cli_args(parser)
    # V1 enables prefix caching by default which skews the latency
    # numbers. We need to disable prefix caching by default.
    parser.set_defaults(enable_prefix_caching=False)

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    if args.profile and not envs.VLLM_TORCH_PROFILER_DIR:
        raise OSError(
            "The environment variable 'VLLM_TORCH_PROFILER_DIR' is not set. "
            "Please set it to a valid path to use torch profiler."
        )
    main(args)
