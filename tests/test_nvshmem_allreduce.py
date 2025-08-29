import sys

sys.path.append("/home/flashinfer_paddle")
import logging
import multiprocessing as mp
import socket
from typing import Any

import paddle
import pytest
from paddle_utils import *

from flashinfer.comm.nvshmem_allreduce import NVSHMEMAllReduce

logger = logging.getLogger(__name__)


def _run_correctness_worker(world_size, rank, distributed_init_port):
    assert rank >= 0
    paddle.device.set_device(device=device2str(rank))
    device = device2str("cuda", rank)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
>>>>>>    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method=distributed_init_method,
    )
>>>>>>    group = torch.distributed.group.WORLD
    num_ranks = paddle.distributed.get_world_size()
    rank_id = paddle.distributed.get_rank()
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    max_batch_size = 4096
    hidden_dim = 8192
    test_loop = 10
    tensor_dtype = "bfloat16"
    nvshmem_allreduce = NVSHMEMAllReduce(
        rank_id, num_ranks, max_batch_size * hidden_dim, tensor_dtype, device, group
    )
    try:
        for batch_size in batch_sizes:
            for _ in range(test_loop):
                tensor_size = batch_size * hidden_dim
                inp1 = paddle.full(
                    shape=[tensor_size], fill_value=rank_id, dtype=tensor_dtype
                )
                inp1_ref = inp1.clone()
                out1 = paddle.empty_like(x=inp1)
                nvshmem_allreduce.all_reduce(inp1, out1)
                paddle.distributed.all_reduce(tensor=inp1_ref, group=group)
                paddle.device.synchronize()
                assert paddle.allclose(x=out1, y=inp1_ref).item(), ""
                paddle.distributed.barrier(group=group)
    except Exception as e:
        print(f"Rank {rank_id}: Exception during test: {e}")
        raise
    finally:
        paddle.distributed.barrier(group=group)
        nvshmem_allreduce.shutdown()
>>>>>>        torch.distributed.destroy_process_group(group)


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
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)
    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        """Not Support auto convert *.start, please judge whether it is Pytorch API and convert by yourself"""
>>>>>>        proc.start()
        procs.append(proc)
    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


@pytest.mark.parametrize("world_size", [8])
def test_nvshmem_allreduce(world_size):
    available_gpus = paddle.device.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(world_size, _run_correctness_worker, target_args=())
    print(f"NVSHMEM allreduce tp = {world_size}: OK")
