import sys

sys.path.append("/home/flashinfer_paddle")
import multiprocessing as mp
import socket
from typing import Any

import numpy as np
import paddle
import pytest
from paddle_utils import *

import flashinfer.comm as comm

kOneShotMaxTokenNum = 128
MAX_TOKEN_NUM = 2048
HIDDEN_SIZE = 7168
MAX_EXPERT_NUM = 16
SF_VEC_SIZE = 16
SCALE_FACTOR_RANGE = -1, 1


def _run_correctness_worker(
    world_size,
    rank,
    dtype,
    distributed_init_port,
    shared_expert_output,
    fc2_output,
    scale,
    expanded_idx_to_permuted_idx,
    residual,
):
    def rms_norm(x: paddle.Tensor, weight: paddle.Tensor = None, eps: float = 1e-06):
        y = x * paddle.rsqrt(x=x.pow(y=2).mean(axis=-1, keepdim=True) + eps)
        if weight is not None:
            y = y * weight
        return y

    device = device2str(f"cuda:{rank}")
    paddle.device.set_device(device=device2str(device))
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    paddle.distributed.init_parallel_env()
>>>>>>    group = torch.distributed.group.WORLD
    try:
        device = device2str(f"cuda:{rank}")
        seq_lens = [16]
        top_k = 8
        eps = 1e-05
        launch_with_pdls = [True, False]
        (
            ipc_handles,
            workspace_tensor,
        ) = comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            rank, world_size, MAX_TOKEN_NUM, HIDDEN_SIZE, group=group
        )
        test_loop = 5
        for seq_len in seq_lens:
            for launch_with_pdl in launch_with_pdls:
                paddle.distributed.barrier(group=group)
                test_passed = True
                print(
                    f"test RANK {rank}: seq_len{seq_len}-topk{top_k}-tp{world_size}-{dtype}-pdl{launch_with_pdl} start"
                )
                paddle.distributed.barrier(group=group)
                paddle.device.synchronize()
                for _ in range(test_loop):
                    shared_expert_output = shared_expert_output.to(device)
                    fc2_output = fc2_output.to(device)
                    scale = scale.to(device)
                    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx.to(
                        device
                    )
                    residual = residual.to(device)
                    fc2_output_clone = fc2_output.clone()
                    norm_weight = paddle.randn(shape=(HIDDEN_SIZE,), dtype=dtype)
                    norm_out = paddle.empty_like(x=residual)
                    residual_out = paddle.empty_like(x=residual)
                    paddle.device.synchronize()
                    s = paddle.device.Stream()
                    s.wait_stream(paddle.device.current_stream())
                    with paddle.device.stream_guard(stream=s):
                        for _ in range(test_loop):
                            comm.trtllm_moe_finalize_allreduce_fusion(
                                allreduce_in=fc2_output,
                                residual_in=residual,
                                norm_weight=norm_weight,
                                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                                workspace_ptrs=workspace_tensor,
                                launch_with_pdl=launch_with_pdl,
                                world_rank=rank,
                                world_size=world_size,
                                eps=eps,
                                shared_expert_output=shared_expert_output,
                                expert_scale_factor=scale,
                                norm_out=norm_out,
                                residual_out=residual_out,
                            )
                    paddle.device.current_stream().wait_stream(s)
>>>>>>                    g = torch.cuda.CUDAGraph()
>>>>>>                    with torch.cuda.graph(g):
                        for _ in range(test_loop):
                            comm.trtllm_moe_finalize_allreduce_fusion(
                                allreduce_in=fc2_output,
                                residual_in=residual,
                                norm_weight=norm_weight,
                                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                                workspace_ptrs=workspace_tensor,
                                launch_with_pdl=launch_with_pdl,
                                world_rank=rank,
                                world_size=world_size,
                                eps=eps,
                                shared_expert_output=shared_expert_output,
                                expert_scale_factor=scale,
                                norm_out=norm_out,
                                residual_out=residual_out,
                            )
                    g.replay()
                    paddle.device.synchronize()
                    expert_reduction = paddle.sum(
                        x=fc2_output_clone[expanded_idx_to_permuted_idx]
                        * scale.unsqueeze(axis=-1),
                        axis=1,
                    )
                    torch_before_residual = (
                        expert_reduction + shared_expert_output
                    ) * world_size
                    torch_residual = torch_before_residual + residual
                    torch_residual = torch_residual.to("float32")
                    torch_output_hidden_states = rms_norm(
                        torch_residual, norm_weight, eps
                    ).to(dtype)
                    if not paddle.allclose(
                        x=residual_out.to("float32"),
                        y=torch_residual.to("float32"),
                        rtol=0.2,
                        atol=0.2,
                    ).item():
                        test_passed = False
                        print(f"Rank {rank} residual_out mismatch")
                        print(f"residual_out: {residual_out}")
                        print(f"torch_residual: {torch_residual}")
                        print(
                            f"max diff: {paddle.max(x=paddle.abs(x=residual_out.to('float32') - torch_residual.to('float32')))}"
                        )
                        print(
                            f"max diff idx: {paddle.argmax(x=paddle.abs(x=residual_out.to('float32') - torch_residual.to('float32')))}"
                        )
                        print(
                            f"max diff value: {residual_out.to('float32').view(-1)[paddle.argmax(x=paddle.abs(x=residual_out.to('float32') - torch_residual.to('float32')))]}"
                        )
                        print(
                            f"max diff ref value: {torch_residual.to('float32').view(-1)[paddle.argmax(x=paddle.abs(x=residual_out.to('float32') - torch_residual.to('float32')))]}"
                        )
                    if not paddle.allclose(
                        x=norm_out.to("float32"),
                        y=torch_output_hidden_states.to("float32"),
                        rtol=0.2,
                        atol=0.2,
                    ).item():
                        test_passed = False
                        print(f"Rank {rank} norm_out mismatch")
                        print(f"norm_out: {norm_out}")
                        print(
                            f"torch_output_hidden_states: {torch_output_hidden_states}"
                        )
                        print(
                            f"max diff: {paddle.max(x=paddle.abs(x=norm_out.to('float32') - torch_output_hidden_states.to('float32')))}"
                        )
                        print(
                            f"max diff idx: {paddle.argmax(x=paddle.abs(x=norm_out.to('float32') - torch_output_hidden_states.to('float32')))}"
                        )
                        print(
                            f"max diff value: {norm_out.to('float32').view(-1)[paddle.argmax(x=paddle.abs(x=norm_out.to('float32') - torch_output_hidden_states.to('float32')))]}"
                        )
                        print(
                            f"max diff ref value: {torch_output_hidden_states.to('float32').view(-1)[paddle.argmax(x=paddle.abs(x=norm_out.to('float32') - torch_output_hidden_states.to('float32')))]}"
                        )
                    assert paddle.allclose(
                        x=residual_out.to("float32"),
                        y=torch_residual.to("float32"),
                        rtol=0.2,
                        atol=0.2,
                    ).item(), ""
                    assert paddle.allclose(
                        x=norm_out.to("float32"),
                        y=torch_output_hidden_states.to("float32"),
                        rtol=0.2,
                        atol=0.2,
                    ).item(), ""
                paddle.distributed.barrier(group=group)
                if test_passed:
                    print(
                        f"test RANK {rank}: seq_len{seq_len}-topk{top_k}-tp{world_size}-{dtype}-pdl{launch_with_pdl} passed"
                    )
                else:
                    print(
                        f"test RANK {rank}: seq_len{seq_len}-topk{top_k}-tp{world_size}-{dtype}-pdl{launch_with_pdl} failed"
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
def test_trtllm_moe_finalize_allreduce_fusion(world_size, dtype):
    np.random.seed(42)
    paddle.seed(seed=42)
    paddle.seed(seed=42)
    available_gpus = paddle.device.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    seq_len = 16
    hidden_size = 7168
    top_k = 8
    shared_expert_output = paddle.randn(shape=(seq_len, hidden_size), dtype=dtype)
    fc2_output = paddle.randn(shape=(seq_len * top_k, hidden_size), dtype=dtype)
    scale = paddle.randn(shape=(seq_len, top_k), dtype=dtype)
    expanded_idx_to_permuted_idx = paddle.randint(
        low=0, high=seq_len * top_k, shape=(seq_len, top_k), dtype="int32"
    )
    residual = paddle.randn(
        shape=shared_expert_output.shape, dtype=shared_expert_output.dtype
    )
    multi_process_parallel(
        world_size,
        dtype,
        _run_correctness_worker,
        target_args=(
            shared_expert_output,
            fc2_output,
            scale,
            expanded_idx_to_permuted_idx,
            residual,
        ),
    )
    print(f"moe finalize allreduce fusion tp = {world_size}: OK")
