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
MAX_TOKEN_NUM = 2048
HIDDEN_SIZE = 7168
MAX_EXPERT_NUM = 16
SF_VEC_SIZE = 16
SCALE_FACTOR_RANGE = -1, 1


def _run_correctness_worker(world_size, rank, dtype, distributed_init_port):
    device = device2str(f"cuda:{rank}")
    paddle.device.set_device(device=device2str(device))
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    paddle.distributed.init_parallel_env()
>>>>>>    group = torch.distributed.group.WORLD
    try:
        device = device2str(f"cuda:{rank}")
        token_nums = [1, 64, 128, 256, 2048]
        candidate_active_expert_num = [8, 12, 16]
        swizzled_layout_codes = [
            comm.QuantizationSFLayout.LINEAR,
            comm.QuantizationSFLayout.SWIZZLED_128x4,
            comm.QuantizationSFLayout.SWIZZLED_8x4,
        ]
        launch_with_pdls = [True, False]
        (
            ipc_handles,
            workspace_tensor,
        ) = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank, world_size, MAX_TOKEN_NUM, HIDDEN_SIZE, group=group
        )
        test_loop = 5
        for token_num in token_nums:
            for active_expert_num in candidate_active_expert_num:
                for swizzled_layout_code in swizzled_layout_codes:
                    for launch_with_pdl in launch_with_pdls:
                        paddle.distributed.barrier(group=group)
                        test_passed = True
                        print(
                            f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-layout{swizzled_layout_code}-pdl{launch_with_pdl} start"
                        )
                        paddle.distributed.barrier(group=group)
                        paddle.device.synchronize()
                        for _ in range(test_loop):
                            message_size = token_num * HIDDEN_SIZE
                            residual_in = paddle.randn(shape=message_size, dtype=dtype)
                            residual_in_clone = residual_in.clone()
                            moe_allreduce_out = paddle.zeros(
                                shape=message_size, dtype=dtype
                            )
                            residual_out = paddle.empty_like(x=residual_in)
                            norm_out = paddle.empty_like(x=residual_in)
                            quant_out = paddle.empty(
                                shape=message_size // 4, dtype=dtype
                            )
                            scale_out = None
                            assert (
                                HIDDEN_SIZE % SF_VEC_SIZE == 0
                            ), "HIDDEN_SIZE must be divisible by SF_VEC_SIZE"
                            if (
                                swizzled_layout_code
                                == comm.QuantizationSFLayout.SWIZZLED_128x4
                            ):
                                padded_message_size = (
                                    comm.compute_fp4_swizzled_layout_sf_size(
                                        token_num, HIDDEN_SIZE // SF_VEC_SIZE
                                    )
                                )
                                scale_out = paddle.empty(
                                    shape=padded_message_size, dtype=dtype
                                )
                            else:
                                scale_out = paddle.empty(
                                    shape=message_size // SF_VEC_SIZE, dtype=dtype
                                )
                            rms_gamma = paddle.randn(shape=HIDDEN_SIZE, dtype=dtype)
                            scale_factor = (
                                paddle.rand(shape=[1], dtype="float32")
                                * (SCALE_FACTOR_RANGE[1] - SCALE_FACTOR_RANGE[0])
                                + SCALE_FACTOR_RANGE[0]
                            )
                            rms_eps = 0.001
                            scale_factor_float = scale_factor.item()
                            moe_reduction_scale_input = paddle.randn(
                                shape=active_expert_num * token_num, dtype="float32"
                            )
                            moe_reduction_scale_input_clone = (
                                moe_reduction_scale_input.clone()
                            )
                            moe_reduction_active_experts_token_input = paddle.randn(
                                shape=active_expert_num * message_size, dtype=dtype
                            )
                            (
                                moe_reduction_active_experts_token_input_clone
                            ) = moe_reduction_active_experts_token_input.clone()
                            moe_reduction_token_input = paddle.randn(
                                shape=message_size, dtype=dtype
                            )
                            moe_reduction_token_input_clone = (
                                moe_reduction_token_input.clone()
                            )
                            moe_expert_out = (
                                moe_reduction_active_experts_token_input_clone.view(
                                    active_expert_num, token_num, HIDDEN_SIZE
                                ).to("float32")
                            )
                            moe_scales = moe_reduction_scale_input_clone.view(
                                active_expert_num, token_num
                            ).to("float32")
                            moe_scales = moe_scales.unsqueeze(axis=2)
                            scaled_expert_out = moe_expert_out * moe_scales.to(
                                "float32"
                            )
                            reduced_expert_out = paddle.sum(x=scaled_expert_out, axis=0)
                            moe_out_ref = (
                                reduced_expert_out
                                + moe_reduction_token_input_clone.view(
                                    token_num, HIDDEN_SIZE
                                ).to("float32")
                            )
                            moe_allreduce_ref = moe_out_ref.clone().to(dtype)
                            paddle.distributed.all_reduce(
                                tensor=moe_allreduce_ref, group=group
                            )
                            moe_allreduce_ref = moe_allreduce_ref.to("float32")
                            ref_residual_out = (
                                moe_allreduce_ref
                                + residual_in_clone.view(token_num, HIDDEN_SIZE).to(
                                    "float32"
                                )
                            )
                            variance = (
                                ref_residual_out.to("float32")
                                .pow(y=2)
                                .mean(axis=-1, keepdim=True)
                            )
                            hidden_states = ref_residual_out * paddle.rsqrt(
                                x=variance + rms_eps
                            )
                            ref_norm_out = rms_gamma.to("float32") * hidden_states
                            s = paddle.device.Stream()
                            s.wait_stream(paddle.device.current_stream())
                            with paddle.device.stream_guard(stream=s):
                                for _ in range(3):
                                    comm.trtllm_moe_allreduce_fusion(
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor_float,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        moe_allreduce_out=moe_allreduce_out,
                                        residual_out=residual_out,
                                        norm_out=norm_out,
                                        quant_out=quant_out,
                                        scale_out=scale_out,
                                    )
                            paddle.device.current_stream().wait_stream(s)
                            paddle.device.synchronize()
>>>>>>                            g = torch.cuda.CUDAGraph()
>>>>>>                            with torch.cuda.graph(g):
                                for _ in range(3):
                                    comm.trtllm_moe_allreduce_fusion(
                                        world_size=world_size,
                                        world_rank=rank,
                                        token_num=token_num,
                                        hidden_dim=HIDDEN_SIZE,
                                        workspace_ptrs=workspace_tensor,
                                        launch_with_pdl=launch_with_pdl,
                                        residual_in=residual_in,
                                        rms_gamma=rms_gamma,
                                        rms_eps=rms_eps,
                                        scale_factor=scale_factor_float,
                                        moe_reduction_device_num_experts=active_expert_num,
                                        moe_reduction_scale_input=moe_reduction_scale_input,
                                        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
                                        moe_reduction_token_input=moe_reduction_token_input,
                                        layout_code=swizzled_layout_code,
                                        moe_allreduce_out=moe_allreduce_out,
                                        residual_out=residual_out,
                                        norm_out=norm_out,
                                        quant_out=quant_out,
                                        scale_out=scale_out,
                                    )
                            g.replay()
                            moe_allreduce_out = moe_allreduce_out.view(
                                token_num, HIDDEN_SIZE
                            )
                            residual_out = residual_out.view(token_num, HIDDEN_SIZE)
                            norm_out = norm_out.view(token_num, HIDDEN_SIZE)
                            paddle.device.synchronize()
                            tolerance = 0.08 if dtype == "float16" else 0.8
                            if not paddle.allclose(
                                x=moe_allreduce_out.to("float32"),
                                y=moe_allreduce_ref,
                                atol=tolerance,
                                rtol=0.01,
                            ).item():
                                test_passed = False
                                print(f"Rank {rank} moe_allreduce_out mismatch")
                                print(f"moe_allreduce_out: {moe_allreduce_out}")
                                print(f"moe_allreduce_ref: {moe_allreduce_ref}")
                                max_diff = paddle.max(
                                    x=paddle.abs(
                                        x=moe_allreduce_out.to("float32")
                                        - moe_allreduce_ref
                                    )
                                )
                                max_diff_idx = paddle.argmax(
                                    x=paddle.abs(
                                        x=moe_allreduce_out.to("float32")
                                        - moe_allreduce_ref
                                    )
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out max diff: {max_diff}"
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out max diff idx: {max_diff_idx}"
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out value at max diff: {moe_allreduce_out.view(-1)[max_diff_idx]}"
                                )
                                print(
                                    f"Rank {rank} moe_allreduce_out ref value at max diff: {moe_allreduce_ref.view(-1)[max_diff_idx]}"
                                )
                            assert paddle.allclose(
                                x=moe_allreduce_out.to("float32"),
                                y=moe_allreduce_ref,
                                atol=tolerance,
                                rtol=0.01,
                            ).item(), ""
                            if not paddle.allclose(
                                x=residual_out.to("float32"),
                                y=ref_residual_out,
                                atol=tolerance,
                                rtol=0.01,
                            ).item():
                                test_passed = False
                                print(f"Rank {rank} residual_out mismatch")
                                print(f"residual_out: {residual_out}")
                                print(f"ref_residual_out: {ref_residual_out}")
                                max_diff = paddle.max(
                                    x=paddle.abs(
                                        x=residual_out.to("float32") - ref_residual_out
                                    )
                                )
                                max_diff_idx = paddle.argmax(
                                    x=paddle.abs(
                                        x=residual_out.to("float32") - ref_residual_out
                                    )
                                )
                                print(f"Rank {rank} residual_out max diff: {max_diff}")
                                print(
                                    f"Rank {rank} residual_out max diff idx: {max_diff_idx}"
                                )
                                print(
                                    f"Rank {rank} residual_out value at max diff: {residual_out.view(-1)[max_diff_idx]}"
                                )
                                print(
                                    f"Rank {rank} residual_out ref value at max diff: {ref_residual_out.view(-1)[max_diff_idx]}"
                                )
                            assert paddle.allclose(
                                x=residual_out.to("float32"),
                                y=ref_residual_out,
                                atol=tolerance,
                                rtol=0.01,
                            ).item(), ""
                            if not paddle.allclose(
                                x=norm_out.to("float32"),
                                y=ref_norm_out,
                                atol=tolerance,
                                rtol=0.01,
                            ).item():
                                test_passed = False
                                print(f"Rank {rank} norm_out mismatch")
                                print(f"norm_out: {norm_out}")
                                print(f"ref_norm_out: {ref_norm_out}")
                                max_diff = paddle.max(
                                    x=paddle.abs(
                                        x=norm_out.to("float32") - ref_norm_out
                                    )
                                )
                                max_diff_idx = paddle.argmax(
                                    x=paddle.abs(
                                        x=norm_out.to("float32") - ref_norm_out
                                    )
                                )
                                print(f"Rank {rank} norm_out max diff: {max_diff}")
                                print(
                                    f"Rank {rank} norm_out max diff idx: {max_diff_idx}"
                                )
                                print(
                                    f"Rank {rank} norm_out value at max diff: {norm_out.view(-1)[max_diff_idx]}"
                                )
                                print(
                                    f"Rank {rank} norm_out ref value at max diff: {ref_norm_out.view(-1)[max_diff_idx]}"
                                )
                            assert paddle.allclose(
                                x=norm_out.to("float32"),
                                y=ref_norm_out,
                                atol=tolerance,
                                rtol=0.01,
                            ).item(), ""
                        paddle.distributed.barrier(group=group)
                        if test_passed:
                            print(
                                f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-layout{swizzled_layout_code}-pdl{launch_with_pdl} passed"
                            )
                        else:
                            print(
                                f"test RANK {rank}: token{token_num}-expert{active_expert_num}-tp{world_size}-{dtype}-layout{swizzled_layout_code}-pdl{launch_with_pdl} failed"
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
    world_size: int, dtype: paddle.dtype, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)
    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, dtype, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        """Not Support auto convert *.start, please judge whether it is Pytorch API and convert by yourself"""
>>>>>>        proc.start()
        procs.append(proc)
    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_trtllm_moe_allreduce_fusion(world_size, dtype):
    np.random.seed(42)
    paddle.seed(seed=42)
    paddle.seed(seed=42)
    available_gpus = paddle.device.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(world_size, dtype, _run_correctness_worker, target_args=())
    print(f"moe allreduce fusion tp = {world_size}: OK")


if __name__ == "__main__":
    test_trtllm_moe_allreduce_fusion(2, "float16")
