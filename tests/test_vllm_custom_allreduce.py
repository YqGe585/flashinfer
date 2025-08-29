import sys


import logging
import multiprocessing as mp
import socket
from typing import Any

import paddle
import pytest
from flashinfer.paddle_utils import *

import flashinfer.comm as comm

logger = logging.getLogger(__name__)


def _run_correctness_worker(world_size, rank, distributed_init_port):
    device = device2str(f"cuda:{rank}")
    paddle.device.set_device(device=device2str(device))
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    paddle.distributed.init_parallel_env()
>>>>>>    group = torch.distributed.group.WORLD
    try:
        device = device2str(f"cuda:{rank}")
        max_size = 8192 * 1024
        meta_ptrs = comm.create_shared_buffer(
            comm.vllm_meta_size() + max_size, group=group
        )
        rank_data = paddle.empty(shape=8 * 1024 * 1024, dtype="uint8")
        buffer_ptrs = comm.create_shared_buffer(max_size, group=group)
        custom_ptr = comm.vllm_init_custom_ar(meta_ptrs, rank_data, rank, True)
        comm.vllm_register_buffer(custom_ptr, buffer_ptrs)
        test_sizes = [
            512,
            2560,
            4096,
            5120,
            7680,
            32768,
            262144,
            524288,
            1048576,
            2097152,
        ]
        num_ctas = [1, 2, 4, 8, 16, 32, 36]
        dtypes = ["float32", "float16", "bfloat16"]
        test_loop = 10
        for test_size in test_sizes:
            for num_cta in num_ctas:
                for dtype in dtypes:
                    for _ in range(test_loop):
                        inp1 = paddle.randint(
                            low=1, high=16, shape=(test_size,), dtype=dtype
                        )
                        inp1_ref = inp1.clone()
                        out1 = paddle.empty_like(x=inp1)
                        comm.vllm_all_reduce(
                            custom_ptr, inp1, out1, buffer_ptrs[rank], max_size, num_cta
                        )
                        paddle.distributed.all_reduce(tensor=inp1_ref, group=group)
                        assert paddle.allclose(x=out1, y=inp1_ref).item(), ""
    finally:
        paddle.distributed.barrier(group=group)
        if custom_ptr is not None:
            comm.vllm_dispose(custom_ptr)
        if buffer_ptrs:
            comm.free_shared_buffer(buffer_ptrs, group)
        if meta_ptrs:
            comm.free_shared_buffer(meta_ptrs, group)
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


@pytest.mark.parametrize("world_size", [2, 4])
def test_vllm_custom_allreduce(world_size):
    available_gpus = paddle.device.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(world_size, _run_correctness_worker, target_args=())
    print(f"custom allreduce tp = {world_size}: OK")
