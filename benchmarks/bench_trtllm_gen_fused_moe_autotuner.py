import sys

sys.path.append("/home/flashinfer")
import argparse
from typing import Literal, Optional

import numpy as np
import paddle
from paddle_utils import *

from flashinfer import (GatedActType, RoutingMethodType, fp4_quantize,
                        mxfp8_quantize, next_positive_power_of_2)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
from flashinfer.testing.utils import bench_gpu_time
from flashinfer.utils import device_support_pdl


def get_tile_tokens_dim(num_tokens, num_experts, top_k):
    imbalance_factor = 1.3
    num_tokens_per_expert = num_tokens * top_k // num_experts
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


def bench_trtllm_gen_fused_moe_autotuner(
    tune_max_num_tokens: Optional[int],
    quant_mode: Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    warmups: int,
    iterations: int,
):
    device = device2str("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_logits = paddle.rand(shape=[num_tokens, num_experts]).to("bfloat16")
    hidden_states = paddle.randn(shape=[num_tokens, hidden_size]).to("bfloat16")
    if quant_mode == "NvFP4xNvFP4":
        hidden_states, hidden_states_scale = fp4_quantize(
            hidden_states,
            paddle.to_tensor(data=[448.0 * 6.0], place=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
>>>>>>        hidden_states_scale = hidden_states_scale.view(paddle.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0 / 448.0 / 6.0
    elif quant_mode == "MxFP4xMxFP8":
        hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states, False)
>>>>>>        hidden_states_scale = hidden_states_scale.view(paddle.float8_e4m3fn).reshape(
            num_tokens, -1
        )
        hidden_states_global_scale = 1.0
    else:
        hidden_states_scale = None
        hidden_states_global_scale = 1.0
    w13 = paddle.randn(shape=[num_experts, intermediate_size * 2, hidden_size]).to(
        "bfloat16"
    )
    w2 = paddle.randn(shape=[num_experts, hidden_size, intermediate_size]).to(
        "bfloat16"
    )
    if quant_mode == "NvFP4xNvFP4":
        w13, w13_scale = fp4_quantize(
            w13,
            paddle.to_tensor(data=[448.0 * 6.0], place=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
>>>>>>        w13_scale = w13_scale.view(paddle.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            paddle.to_tensor(data=[448.0 * 6.0], place=device),
            sf_vec_size=16,
            sf_use_ue8m0=False,
        )
>>>>>>        w2_scale = w2_scale.view(paddle.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0 / 448.0 / 6.0
        w2_global_scale = 1.0 / 448.0 / 6.0
    else:
        w13, w13_scale = fp4_quantize(
            w13,
            paddle.to_tensor(data=[1.0], place=device),
            sf_vec_size=32,
            sf_use_ue8m0=True,
        )
>>>>>>        w13_scale = w13_scale.view(paddle.float8_e4m3fn).reshape(
            num_experts, intermediate_size * 2, -1
        )
        w2, w2_scale = fp4_quantize(
            w2,
            paddle.to_tensor(data=[1.0], place=device),
            sf_vec_size=32,
            sf_use_ue8m0=True,
        )
>>>>>>        w2_scale = w2_scale.view(paddle.float8_e4m3fn).reshape(
            num_experts, hidden_size, -1
        )
        w13_global_scale = 1.0
        w2_global_scale = 1.0
    bias13 = paddle.randn(shape=[num_experts, intermediate_size * 2]) * 10
    bias2 = paddle.randn(shape=[num_experts, intermediate_size * 2]) * 10
    tile_tokens_dim = get_tile_tokens_dim(num_tokens, num_experts, top_k)
    output1_scale_scalar = paddle.to_tensor(
        data=[hidden_states_global_scale * w13_global_scale] * num_experts, place=device
    )
    output1_scale_gate_scalar = paddle.to_tensor(
        data=[hidden_states_global_scale * w13_global_scale] * num_experts, place=device
    )
    output2_scale_scalar = paddle.to_tensor(
        data=[hidden_states_global_scale * w2_global_scale] * num_experts, place=device
    )
    fn = lambda: trtllm_fp4_block_scale_moe(
        routing_logits,
        None,
        hidden_states,
        hidden_states_scale,
        w13,
        w13_scale,
        bias13,
        None,
        None,
        None,
        w2,
        w2_scale,
        bias2,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        None,
        None,
        intermediate_size,
        0,
        num_experts,
        None,
        tile_tokens_dim,
        RoutingMethodType.Renormalize.value[0],
        True,
        enable_pdl,
        GatedActType.SwiGlu.value,
        None,
        num_tokens if tune_max_num_tokens is None else tune_max_num_tokens,
    )

    def bench(do_autotune):
        with autotune(do_autotune):
            for _ in range(warmups):
                fn()
        ms_list = bench_gpu_time(fn, repeat_iters=iterations)
        median_ms = np.median(ms_list)
        return median_ms

    ms = bench(do_autotune=False)
    ms_tuned = bench(do_autotune=True)
    print(
        f"num tokens: {num_tokens}, num experts: {num_experts}, hidden size: {hidden_size}, intermediate size: {intermediate_size}, top k: {top_k}"
    )
    print(f"No autotune: {ms:.3f} ms; with autotune: {ms_tuned:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="MxFP4xMxFP8",
        choices=["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"],
        help="Quantization mode",
    )
    parser.add_argument("--num-tokens", type=int, default=512, help="Number of tokens")
    parser.add_argument(
        "--tune-max-num-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens for tunning",
    )
    parser.add_argument(
        "--num-experts", type=int, default=128, help="Number of experts"
    )
    parser.add_argument("--hidden-size", type=int, default=3072, help="Hidden size")
    parser.add_argument(
        "--intermediate-size", type=int, default=3072, help="Intermediate size"
    )
    parser.add_argument("--top-k", type=int, default=4, help="Top-k experts per token")
    parser.add_argument(
        "--warmups", type=int, default=100, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    args = parser.parse_args()
    bench_trtllm_gen_fused_moe_autotuner(
        args.tune_max_num_tokens,
        args.quant_mode,
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.warmups,
        args.iterations,
    )
