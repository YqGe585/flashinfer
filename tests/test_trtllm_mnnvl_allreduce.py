import sys


import sys
from typing import Tuple

import paddle
import pytest
from mpi4py import MPI
from flashinfer.paddle_utils import *

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping
from flashinfer.norm import rmsnorm


@paddle.no_grad()
def row_linear_residual_norm_fusion_forward(
    x: paddle.Tensor,
    residual: paddle.Tensor,
    norm_weight: paddle.Tensor,
    eps: float,
    hidden_size: int,
    dtype: paddle.dtype,
    mapping: Mapping,
    fusion: bool,
    reference_output: tuple[paddle.Tensor, ...],
    multicast_ptr: int,
    buffer_ptrs_dev: int,
    unicast_ptr: int,
    max_num_elements_mnnvl: int,
    buffer_flags_mnnvl: paddle.Tensor,
):
    x = x.cuda()
    residual = residual.cuda()
    norm_weight = norm_weight.cuda()
    reference_output = tuple(t.cuda() for t in reference_output)
    tensor_parallel_size = mapping.tp_size
    tensor_parallel_rank = mapping.tp_rank
    MPI.COMM_WORLD.barrier()

    def func(
        input,
        residual,
        norm_weight,
        eps,
        enable_fusion,
        multicast_ptr,
        buffer_ptrs_dev,
        unicast_ptr,
        max_num_elements_mnnvl,
    ):
        shape = tuple(input.shape)
        assert max_num_elements_mnnvl % hidden_size == 0
        input = input.view(-1, shape[-1])
        buffer_M = max_num_elements_mnnvl // hidden_size
        if enable_fusion:
            use_pdl = True
            prenorm_output = paddle.empty_like(x=residual)
            normed_output = paddle.empty_like(x=residual)
            trtllm_mnnvl_ar.mpi_barrier()
            trtllm_mnnvl_ar.trtllm_mnnvl_fused_allreduce_rmsnorm(
                prenorm_output,
                normed_output,
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                unicast_ptr,
                buffer_M,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                norm_weight,
                eps,
                residual,
                use_pdl,
            )
            return normed_output.view(shape), prenorm_output.view(shape)
        else:
            output = paddle.empty_like(x=input)
            trtllm_mnnvl_ar.trtllm_mnnvl_all_reduce(
                input,
                multicast_ptr,
                buffer_ptrs_dev,
                buffer_M,
                buffer_flags_mnnvl,
                tensor_parallel_size,
                tensor_parallel_rank,
                True,
                False,
                output,
            )
            return (output.view(shape),)

    output = func(
        x.clone(),
        residual.clone(),
        norm_weight,
        eps,
        fusion,
        multicast_ptr,
        buffer_ptrs_dev,
        unicast_ptr,
        max_num_elements_mnnvl,
    )
    assert tuple(output[0].shape) == tuple(reference_output[0].shape)
    if tensor_parallel_rank == 0:
        print("output[0] (first 10 values):", output[0].flatten()[:10])
        print(
            "reference_output[0] (first 10 values):", reference_output[0].flatten()[:10]
        )
        if fusion:
            print("output[1] (first 10 values):", output[1].flatten()[:10])
            print(
                "reference_output[1] (first 10 values):",
                reference_output[1].flatten()[:10],
            )
    assert paddle.allclose(
        x=output[0], y=reference_output[0], rtol=0.05, atol=0.15
    ).item(), ""
    if fusion:
        assert paddle.allclose(
            x=output[1], y=reference_output[1], rtol=0.05, atol=0.15
        ).item(), ""


"""Main test function that runs on each MPI rank"""


@pytest.mark.parametrize("seq_lens", [[1], [4], [15], [27, 11, 24], [127]])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("hidden_size", [2048, 4096, 5120, 7168, 8192])
def test_mnnvl_allreduce_full(
    monkeypatch,
    seq_lens: list[int],
    fusion: bool,
    dtype: paddle.dtype,
    hidden_size: int,
):
    monkeypatch.setenv("TRTLLM_FORCE_MNNVL_AR", "1")
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    gpus_per_node = paddle.device.cuda.device_count()
    if gpus_per_node == 0:
        pytest.skip("MNNVL allreduce test requires at least one CUDA device per node")
    if world_size < 2:
        if rank == 0:
            print(f"ERROR: This test requires at least 2 MPI ranks, got {world_size}")
        sys.exit(1)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=gpus_per_node,
        tp_size=world_size,
    )
    paddle.device.set_device(device=device2str(mapping.local_rank))
    if mapping.local_rank == 0:
        print(
            f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks"
        )
        print(
            f"[Node {mapping.node_rank}] Rank {rank} using GPU {paddle.device.get_device()}"
        )
    tensor_parallel_size = world_size
    eps = 1e-05
    paddle.seed(seed=42)
    rank_failed = False
    failure_message = ""
    try:
        (
            mcast_buffer_mnnvl,
            buffer_flags_mnnvl,
            max_num_elements_mnnvl,
        ) = trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(mapping, dtype)
        multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr()
        buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev()
        unicast_ptr = mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr(
            mapping.tp_rank
        )
        for seq_len in seq_lens:
            if rank == 0:
                print(
                    f"Testing seq_len={seq_len}, hidden_size={hidden_size}, fusion={fusion}, dtype={dtype}"
                )
            x_full = paddle.randn(
                shape=(tensor_parallel_size, seq_len, hidden_size), dtype=dtype
            )
            residual = paddle.randn(shape=(seq_len, hidden_size), dtype=dtype)
            norm_weight = paddle.randn(shape=(hidden_size,), dtype=dtype)
            x = x_full[rank, :, :]
            reference_output: Tuple[paddle.Tensor, ...] = None
            if fusion:
                allreduce_result = paddle.sum(x=x_full, axis=0)
                residual_out = allreduce_result + residual
                print(
                    "Device of residual_out:{}, norm_weight:{}".format(
                        residual_out.place, norm_weight.place
                    )
                )
                norm_out = rmsnorm(residual_out, norm_weight, eps, enable_pdl=False)
                reference_output = norm_out, residual_out
            else:
                allreduce_result = paddle.sum(x=x_full, axis=0)
                reference_output = (allreduce_result,)
            row_linear_residual_norm_fusion_forward(
                x,
                residual,
                norm_weight,
                eps,
                hidden_size,
                dtype,
                mapping,
                fusion,
                reference_output,
                multicast_ptr,
                buffer_ptrs_dev,
                unicast_ptr,
                max_num_elements_mnnvl,
                buffer_flags_mnnvl,
            )
            trtllm_mnnvl_ar.mpi_barrier()
            print(
                f"PASSED[rank={rank}]: seq_len={seq_len}, fusion={fusion}, dtype={dtype}"
            )
    except Exception as e:
        rank_failed = True
        failure_message = f"FAILED[rank={rank}]: seq_lens={seq_lens}, fusion={fusion}, dtype={dtype} failed: {e}"
        print(failure_message)
        all_failures = MPI.COMM_WORLD.allgather(rank_failed)
        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")
            pytest.fail(f"Test failed on ranks {failed_ranks}")
            trtllm_mnnvl_ar.mpi_barrier()
    finally:
        if "mcast_buffer_mnnvl" in locals():
            del mcast_buffer_mnnvl
    trtllm_mnnvl_ar.mpi_barrier()
