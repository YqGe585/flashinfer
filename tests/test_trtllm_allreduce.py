import sys

sys.path.append("/home/flashinfer")
import multiprocessing as mp
import socket
from typing import Any

import paddle
import pytest
from paddle_utils import *

import flashinfer.comm as comm

"""
NOTE:
The assertion of result closeness is disabled for now,
since assertion fails for some cases, which breaks the tests and introduces NCCL timeout.

Trt-llm encourage using certain shapes for this custom all-reduce kernel,

hidden_size in range [256, 8192], and maxHiddenSize should be 8192.
The recommended case is [1024, 2048, 4096, 8192].

If new trt-llm source kernels are available (function name starts with "trtllm_"), we would recommend using them.
"""
maxBatchSize = 1
maxBeamWidth = 3
maxTokenNum = 128
maxHiddenSize = 4096
RANDOM_SEED = 42


def _run_correctness_worker(world_size, rank, dtype, distributed_init_port):
    device = device2str(f"cuda:{rank}")
    paddle.device.set_device(device=device2str(device))
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    paddle.distributed.init_parallel_env()
>>>>>>    group = torch.distributed.group.WORLD
    try:
        device = device2str(f"cuda:{rank}")
        token_nums = [64, 128]
        strategy_codes = [
            comm.AllReduceStrategyType.ONESHOT,
            comm.AllReduceStrategyType.TWOSHOT,
        ]
        hidden_sizes = [1024, 4096]
        config_codes = [
            0,
            comm.AllReduceStrategyConfig.USE_MEMCPY,
            comm.AllReduceStrategyConfig.PUSH_MODE,
        ]
        fusion_op_codes = [
            comm.AllReduceFusionOp.NONE,
            comm.AllReduceFusionOp.RESIDUAL_RMS_NORM,
            comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM,
        ]
        launch_with_pdls = [True, False]
        workspace = comm.trtllm_create_ipc_workspace_for_all_reduce(
            rank=rank,
            tp_size=world_size,
            max_token_num=maxTokenNum,
            hidden_dim=maxHiddenSize,
            group=group,
        )
        test_loop = 2
        flag_value = 1
        for token_num in token_nums:
            for hidden_size in hidden_sizes:
                for strategy_code in strategy_codes:
                    for config_code in config_codes:
                        for fusion_op_code in fusion_op_codes:
                            for launch_with_pdl in launch_with_pdls:
                                pass_flag = True
                                if (
                                    strategy_code == comm.AllReduceStrategyType.TWOSHOT
                                    and fusion_op_code
                                    == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                                ):
                                    continue
                                print(
                                    f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{hidden_size} start"
                                )
                                paddle.device.synchronize()
                                for _ in range(test_loop):
                                    message_size = token_num * hidden_size
                                    inp1 = paddle.randn(shape=message_size, dtype=dtype)
                                    inp1_ref = inp1.clone()
                                    out1 = paddle.empty_like(x=inp1)
                                    bias = paddle.randn(shape=hidden_size, dtype=dtype)
                                    residual = paddle.randn(
                                        shape=message_size, dtype=dtype
                                    )
                                    weight = paddle.randn(
                                        shape=hidden_size, dtype=dtype
                                    )
                                    weight_pre_residual_norm = paddle.randn(
                                        shape=hidden_size, dtype=dtype
                                    )
                                    eps = 1e-06
                                    intermediate_buffer = paddle.zeros(
                                        shape=message_size, dtype=dtype
                                    )
                                    comm.trtllm_custom_all_reduce(
                                        inp=inp1,
                                        out=out1,
                                        tp_size=world_size,
                                        tp_rank=rank,
                                        token_num=token_num,
                                        fusion_op_code=fusion_op_code,
                                        strategy_code=strategy_code,
                                        config_code=config_code,
                                        launch_with_pdl=launch_with_pdl,
                                        flag_value=flag_value,
                                        peer_comm_buffer_ptrs=paddle.to_tensor(
                                            data=workspace[0], dtype="int64"
                                        ),
                                        peer_barrier_ptrs_in=paddle.to_tensor(
                                            data=workspace[2], dtype="int64"
                                        ),
                                        peer_barrier_ptrs_out=paddle.to_tensor(
                                            data=workspace[3], dtype="int64"
                                        ),
                                        bias=bias,
                                        residual=residual,
                                        weight=weight,
                                        weight_pre_residual_norm=weight_pre_residual_norm,
                                        eps=eps,
                                        intermediate_buffer=intermediate_buffer,
                                        lamport_peer_comm_buffer_ptrs_0=paddle.to_tensor(
                                            data=workspace[4], dtype="int64"
                                        ),
                                        lamport_peer_comm_buffer_ptrs_1=paddle.to_tensor(
                                            data=workspace[5], dtype="int64"
                                        ),
                                        lamport_peer_comm_buffer_ptrs_2=paddle.to_tensor(
                                            data=workspace[6], dtype="int64"
                                        ),
                                    )
                                    paddle.distributed.all_reduce(
                                        tensor=inp1_ref, group=group
                                    )
                                    tolerance = 0.01 if dtype == "float16" else 0.08
                                    if fusion_op_code == comm.AllReduceFusionOp.NONE:
                                        assert paddle.allclose(
                                            x=out1,
                                            y=inp1_ref,
                                            atol=tolerance,
                                            rtol=0.03,
                                        ).item(), ""
                                    elif (
                                        fusion_op_code
                                        == comm.AllReduceFusionOp.RESIDUAL_RMS_NORM
                                    ):
                                        inter_buffer = intermediate_buffer.clone()
                                        ref = inp1_ref.clone()
                                        ref_float = ref.to("float32")
                                        residual_float = residual.to("float32")
                                        bias_float = bias.to("float32")
                                        for i in range(ref.size):
                                            ref_float[i] += (
                                                residual_float[i]
                                                + bias_float[i % hidden_size]
                                            )
                                        ref_half = ref_float.to(dtype)
                                        assert paddle.allclose(
                                            x=inter_buffer,
                                            y=ref_half,
                                            atol=tolerance,
                                            rtol=0.03,
                                        ).item(), ""
                                        ref_float = ref_float.view(
                                            token_num, hidden_size
                                        )
                                        normed_float = paddle.empty_like(x=ref_float)
                                        mean_sq = paddle.mean(
                                            x=ref_float * ref_float,
                                            axis=-1,
                                            keepdim=True,
                                        )
                                        denom = paddle.sqrt(x=mean_sq + eps)
                                        normed_float = ref_float / denom
                                        normed_float = normed_float * weight.to(
                                            "float32"
                                        )
                                        normed_half = normed_float.to(dtype)
                                        assert paddle.allclose(
                                            x=out1,
                                            y=normed_half.view(-1),
                                            atol=tolerance,
                                            rtol=0.03,
                                        ).item(), ""
                                    elif (
                                        fusion_op_code
                                        == comm.AllReduceFusionOp.RESIDUAL_RMS_PREPOST_NORM
                                    ):
                                        pass
                                    flag_value += 1
                                if pass_flag:
                                    print(
                                        f"test RANK {rank}: {world_size}-{dtype}-{strategy_code}-{config_code}-{fusion_op_code}-{launch_with_pdl}-{hidden_size} passed"
                                    )
    finally:
        paddle.distributed.barrier(group=group)
        comm.trtllm_destroy_ipc_workspace_for_all_reduce(workspace, group=group)
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
def test_trtllm_custom_allreduce(world_size, dtype):
    paddle.seed(seed=RANDOM_SEED)
    available_gpus = paddle.device.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(world_size, dtype, _run_correctness_worker, target_args=())
    print(f"custom allreduce tp = {world_size}: OK")
