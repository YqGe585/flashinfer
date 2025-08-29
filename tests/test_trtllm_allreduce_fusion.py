import sys


import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import paddle
import pytest
from flashinfer.paddle_utils import *

import flashinfer.comm as comm

kOneShotMaxTokenNum = 128
MIN_TOKEN_NUM = 1
MAX_TOKEN_NUM = 2048
SF_VEC_SIZE = 16
SCALE_FACTOR_RANGE = -1, 1


def _run_correctness_worker(world_size, rank, dtype, hidden_dim, distributed_init_port):
    device = device2str(f"cuda:{rank}")
    paddle.device.set_device(device=device2str(device))
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    paddle.distributed.init_parallel_env()
>>>>>>    group = torch.distributed.group.WORLD
    try:
        device = device2str(f"cuda:{rank}")
        token_nums = [1, 128, 1024, 2048]
        pattern_codes = [
            comm.AllReduceFusionPattern.kAllReduce,
            comm.AllReduceFusionPattern.kARResidualRMSNorm,
            comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant,
            comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant,
        ]
        swizzled_layout_codes = [
            comm.QuantizationSFLayout.LINEAR,
            comm.QuantizationSFLayout.SWIZZLED_128x4,
            comm.QuantizationSFLayout.SWIZZLED_8x4,
        ]
        launch_with_pdls = [True, False]
        use_oneshots = [True, False, None]
        trigger_completion_at_ends = [True, False]
        fp32_accs = [True, False]
        lamport_use_fp32 = dtype == "float32"
        (
            ipc_handles,
            workspace_tensor,
        ) = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank,
            world_size,
            MAX_TOKEN_NUM,
            hidden_dim,
            group=group,
            use_fp32_lamport=lamport_use_fp32,
        )
        test_loop = 5
        for token_num in token_nums:
            for pattern_code in pattern_codes:
                for swizzled_layout_code in swizzled_layout_codes:
                    for launch_with_pdl in launch_with_pdls:
                        for use_oneshot in use_oneshots:
                            for trigger_completion_at_end in trigger_completion_at_ends:
                                for fp32_acc in fp32_accs:
                                    if token_num < world_size and not use_oneshot:
                                        continue
                                    if dtype == "float32" and (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant
                                        or pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant
                                    ):
                                        continue
                                    paddle.distributed.barrier(group=group)
                                    test_passed = True
                                    print(
                                        f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} start"
                                    )
                                    paddle.distributed.barrier(group=group)
                                    paddle.device.synchronize()
                                    message_size = token_num * hidden_dim
                                    allreduce_in = paddle.randn(
                                        shape=message_size, dtype=dtype
                                    )
                                    allreduce_in_clone = allreduce_in.clone()
                                    all_reduce_out = paddle.zeros(
                                        shape=message_size, dtype=dtype
                                    )
                                    residual_in = paddle.randn(
                                        shape=message_size, dtype=dtype
                                    )
                                    residual_in_clone = residual_in.clone()
                                    residual_out = paddle.empty_like(x=residual_in)
                                    norm_out = paddle.empty_like(x=residual_in)
                                    quant_out = paddle.empty(
                                        shape=message_size, dtype=dtype
                                    )
                                    scale_out = None
                                    assert (
                                        hidden_dim % SF_VEC_SIZE == 0
                                    ), "hidden_dim must be divisible by SF_VEC_SIZE"
                                    if (
                                        swizzled_layout_code
                                        == comm.QuantizationSFLayout.SWIZZLED_128x4
                                    ):
                                        padded_message_size = (
                                            (token_num + 127)
                                            // 128
                                            * 128
                                            * ((hidden_dim + 63) // 64 * 4)
                                        )
                                        scale_out = paddle.empty(
                                            shape=padded_message_size, dtype=dtype
                                        )
                                    else:
                                        scale_out = paddle.empty(
                                            shape=message_size // SF_VEC_SIZE,
                                            dtype=dtype,
                                        )
                                    rms_gamma = paddle.randn(
                                        shape=hidden_dim, dtype=dtype
                                    )
                                    scale_factor = (
                                        paddle.rand(shape=[1], dtype="float32")
                                        * (
                                            SCALE_FACTOR_RANGE[1]
                                            - SCALE_FACTOR_RANGE[0]
                                        )
                                        + SCALE_FACTOR_RANGE[0]
                                    )
                                    rms_eps = 0.001
                                    s = paddle.device.Stream()
                                    s.wait_stream(paddle.device.current_stream())
                                    with paddle.device.stream_guard(stream=s):
                                        for _ in range(test_loop):
                                            comm.trtllm_allreduce_fusion(
                                                allreduce_in=allreduce_in,
                                                world_size=world_size,
                                                world_rank=rank,
                                                token_num=token_num,
                                                hidden_dim=hidden_dim,
                                                workspace_ptrs=workspace_tensor,
                                                launch_with_pdl=launch_with_pdl,
                                                use_oneshot=use_oneshot,
                                                trigger_completion_at_end=trigger_completion_at_end,
                                                fp32_acc=fp32_acc,
                                                pattern_code=pattern_code,
                                                allreduce_out=all_reduce_out,
                                                residual_in=residual_in,
                                                residual_out=residual_out,
                                                norm_out=norm_out,
                                                quant_out=quant_out,
                                                scale_out=scale_out,
                                                rms_gamma=rms_gamma,
                                                rms_eps=rms_eps,
                                                scale_factor=scale_factor,
                                                layout_code=swizzled_layout_code,
                                            )
>>>>>>                                    g = torch.cuda.CUDAGraph()
>>>>>>                                    with torch.cuda.graph(g):
                                        for _ in range(test_loop):
                                            comm.trtllm_allreduce_fusion(
                                                allreduce_in=allreduce_in,
                                                world_size=world_size,
                                                world_rank=rank,
                                                token_num=token_num,
                                                hidden_dim=hidden_dim,
                                                workspace_ptrs=workspace_tensor,
                                                launch_with_pdl=launch_with_pdl,
                                                use_oneshot=use_oneshot,
                                                trigger_completion_at_end=trigger_completion_at_end,
                                                fp32_acc=fp32_acc,
                                                pattern_code=pattern_code,
                                                allreduce_out=all_reduce_out,
                                                residual_in=residual_in,
                                                residual_out=residual_out,
                                                norm_out=norm_out,
                                                quant_out=quant_out,
                                                scale_out=scale_out,
                                                rms_gamma=rms_gamma,
                                                rms_eps=rms_eps,
                                                scale_factor=scale_factor,
                                                layout_code=swizzled_layout_code,
                                            )
                                    g.replay()
                                    paddle.device.synchronize()
                                    all_reduce_out = all_reduce_out.view(
                                        token_num, hidden_dim
                                    )
                                    residual_out = residual_out.view(
                                        token_num, hidden_dim
                                    )
                                    norm_out = norm_out.view(token_num, hidden_dim)
                                    paddle.device.synchronize()
                                    paddle.distributed.all_reduce(
                                        tensor=allreduce_in_clone, group=group
                                    )
                                    ref_allreduce_out = allreduce_in_clone.clone()
                                    ref_allreduce_out = ref_allreduce_out.view(
                                        token_num, hidden_dim
                                    ).to("float32")
                                    ref_residual_out = (
                                        ref_allreduce_out
                                        + residual_in_clone.view(
                                            token_num, hidden_dim
                                        ).to("float32")
                                    )
                                    variance = (
                                        ref_residual_out.to("float32")
                                        .pow(y=2)
                                        .mean(axis=-1, keepdim=True)
                                    )
                                    hidden_states = ref_residual_out * paddle.rsqrt(
                                        x=variance + rms_eps
                                    )
                                    ref_norm_out = (
                                        rms_gamma.to("float32") * hidden_states
                                    )
                                    tolerance = 0.08 if dtype == "float16" else 0.8
                                    if (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kAllReduce
                                    ):
                                        assert paddle.allclose(
                                            x=all_reduce_out.to("float32"),
                                            y=ref_allreduce_out,
                                            atol=tolerance,
                                            rtol=0.01,
                                        ).item(), ""
                                    elif (
                                        pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP8Quant
                                        or pattern_code
                                        == comm.AllReduceFusionPattern.kARResidualRMSNormOutFP4Quant
                                    ):
                                        assert paddle.allclose(
                                            x=residual_out.to("float32"),
                                            y=ref_residual_out,
                                            atol=tolerance,
                                            rtol=0.01,
                                        ).item(), ""
                                        assert paddle.allclose(
                                            x=norm_out.to("float32"),
                                            y=ref_norm_out,
                                            atol=tolerance,
                                            rtol=0.01,
                                        ).item(), ""
                                    paddle.distributed.barrier(group=group)
                                    if test_passed:
                                        print(
                                            f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} passed"
                                        )
                                    else:
                                        print(
                                            f"test RANK {rank}: token{token_num}-hidden_dim{hidden_dim}-dtype{dtype}-pattern{pattern_code}-layout{swizzled_layout_code}-pdl{launch_with_pdl} failed"
                                        )
    finally:
        paddle.distributed.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce(ipc_handles, group=group)
>>>>>>        torch.distributed.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int,
    dtype: paddle.dtype,
    hidden_dim: int,
    test_target: Any,
    target_args: tuple = (),
) -> None:
    mp.set_start_method("spawn", force=True)
    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (
            world_size,
            i,
            dtype,
            hidden_dim,
            distributed_init_port,
        ) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        """Not Support auto convert *.start, please judge whether it is Pytorch API and convert by yourself"""
>>>>>>        proc.start()
        procs.append(proc)
    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("hidden_dim", [1024, 2048, 4096, 7168, 8192])
def test_trtllm_allreduce_fusion(world_size, dtype, hidden_dim):
    np.random.seed(42)
    paddle.seed(seed=42)
    paddle.seed(seed=42)
    available_gpus = paddle.device.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size, dtype, hidden_dim, _run_correctness_worker, target_args=()
    )
    print(f"allreduce fusion tp = {world_size}: OK")
