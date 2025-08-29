import ctypes
import os
import subprocess
import sys

import paddle
import pytest

import flashinfer.comm as comm
from flashinfer.comm import CudaRTLibrary


def _run_ipc_test():
    if not paddle.distributed.is_initialized():
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    world_size = paddle.distributed.get_world_size()
    cudart = CudaRTLibrary()
    cudart.cudaSetDevice(rank)
    buffer_size_in_bytes = 1024
    byte_value = rank
    pointers = comm.create_shared_buffer(buffer_size_in_bytes)
    print(f"Rank {rank} init ipc buffer {pointers}", flush=True)
    paddle.distributed.barrier()
    paddle.device.synchronize()
    for p in pointers:
        pointer = ctypes.c_void_p(p + rank * (buffer_size_in_bytes // world_size))
        cudart.cudaMemset(pointer, byte_value, buffer_size_in_bytes // world_size)
    paddle.distributed.barrier()
    paddle.device.synchronize()
    host_data = (ctypes.c_char * buffer_size_in_bytes)()
    for p in pointers:
        for cur_rank in range(world_size):
            offset_pointer = ctypes.c_void_p(
                p + cur_rank * (buffer_size_in_bytes // world_size)
            )
            cudart.cudaMemcpy(
                host_data, offset_pointer, buffer_size_in_bytes // world_size
            )
            for i in range(buffer_size_in_bytes // world_size):
                assert (
                    ord(host_data[i]) == cur_rank
                ), f"Rank {rank} failed to verify buffer {p}. Expected {cur_rank}, got {ord(host_data[i])}"
    print(f"Rank {rank} verified all buffers.\n", flush=True)
    paddle.distributed.barrier()
    paddle.device.synchronize()
    comm.free_shared_buffer(pointers)


@pytest.mark.parametrize("world_size", [2, 4])
def test_ipc_distributed(world_size):
    script = os.path.abspath(__file__)
    result = subprocess.run(
        ["torchrun", f"--nproc_per_node={world_size}", script, "--run_ipc_test"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    assert result.returncode == 0


if __name__ == "__main__":
    if "--run_ipc_test" in sys.argv:
        _run_ipc_test()
