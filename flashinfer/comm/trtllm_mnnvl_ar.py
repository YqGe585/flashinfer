import sys

sys.path.append("/home/flashinfer")
import os

import paddle
from paddle_utils import *

"""
MNNVL (Multi-Node NVLink) communication operations for FlashInfer.

"""
import functools
import math
from types import SimpleNamespace
from typing import Optional, Tuple

from flashinfer.comm.mapping import Mapping

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec
from ..utils import register_custom_op
from .mnnvl import McastGPUBuffer


def mpi_barrier():
    from mpi4py import MPI

    """MPI barrier - could potentially be replaced with dist.barrier()"""
    MPI.COMM_WORLD.Barrier()


def gen_trtllm_mnnvl_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_mnnvl_comm", [jit_env.FLASHINFER_CSRC_DIR / "trtllm_mnnvl_allreduce.cu"]
    )


@functools.cache
def get_trtllm_mnnvl_comm_module():
    module = gen_trtllm_mnnvl_comm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_mnnvl_all_reduce",
        mutates_args=[
            "inp",
            "multicast_buffer_ptr",
            "buffer_ptrs_dev",
            "buffer_mnnvl",
            "buffer_flags_mnnvl",
            "nranks",
            "rank",
            "wait_for_results",
            "launch_with_pdl",
            "out",
        ],
    )
    def trtllm_mnnvl_all_reduce(
        inp: paddle.Tensor,
        multicast_buffer_ptr: int,
        buffer_ptrs_dev: int,
        buffer_mnnvl: paddle.Tensor,
        buffer_flags_mnnvl: paddle.Tensor,
        nranks: int,
        rank: int,
        wait_for_results: bool,
        launch_with_pdl: bool,
        out: Optional[paddle.Tensor],
    ) -> None:
        module.trtllm_mnnvl_all_reduce(
            inp,
            multicast_buffer_ptr,
            buffer_ptrs_dev,
            buffer_mnnvl,
            buffer_flags_mnnvl,
            nranks,
            rank,
            wait_for_results,
            launch_with_pdl,
            out,
        )

    @register_custom_op(
        "flashinfer::trtllm_mnnvl_rmsnorm",
        mutates_args=[
            "mcast_buffer_input",
            "prenorm_output",
            "normed_output",
            "gamma",
            "epsilon",
            "residual",
            "buffer_flags",
            "launch_with_pdl",
        ],
    )
    def trtllm_mnnvl_rmsnorm(
        mcast_buffer_input: int,
        prenorm_output: paddle.Tensor,
        normed_output: paddle.Tensor,
        gamma: paddle.Tensor,
        epsilon: float,
        residual: paddle.Tensor,
        buffer_flags: paddle.Tensor,
        launch_with_pdl: bool,
    ) -> None:
        """Performs MNNVL TwoShot RMSNorm on the communication buffer.

        Args:
            prenorm_output: Output tensor for prenorm results
            normed_output: Output tensor for normalized results
            mcast_buffer_input: Input tensor
            gamma: The gamma parameter for RMSNorm
            epsilon: The epsilon parameter for RMSNorm
            residual: The residual tensor to add
            buffer_flags: Buffer flags for synchronization
            launch_with_pdl: Whether to launch with PDL
        """
        return module.trtllm_mnnvl_rmsnorm(
            mcast_buffer_input,
            prenorm_output,
            normed_output,
            gamma,
            epsilon,
            residual,
            buffer_flags,
            launch_with_pdl,
        )

    return SimpleNamespace(
        trtllm_mnnvl_all_reduce=trtllm_mnnvl_all_reduce,
        trtllm_mnnvl_rmsnorm=trtllm_mnnvl_rmsnorm,
    )


def get_allreduce_mnnvl_workspace(
    mapping: Mapping, dtype: paddle.dtype
) -> Tuple[McastGPUBuffer, paddle.Tensor, int]:
    """Get workspace buffers needed for multi-node NVLink all-reduce operation.

    This function allocates and initializes the workspace buffers required for performing
    multi-node NVLink all-reduce operations. It creates:
    1. A multicast GPU buffer for communication between nodes
    2. A flags tensor to track buffer state
    3. Maximum number of elements that can fit in the buffer

    The buffer size is calculated to efficiently handle common hidden dimensions
    (2048, 4096, 5120, 7168, 8192) by using their LCM of 286720.

    Args:
        mapping: Tensor parallel mapping configuration containing rank info
        dtype: Data type of the tensors being reduced

    Returns:
        Tuple containing:
        - McastGPUBuffer: Multicast buffer for inter-node communication
        - torch.Tensor: Buffer flags tensor tracking state
        - int: Maximum number of elements that can fit in buffer
    """
    force_mn = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
    stride = 3 * 2 * dtype.element_size()
    lcm_hidden_dim = 286720
    TARGET_WORKSPACE_SIZE_BYTES = 12000000
    buffer_size_in_bytes = math.ceil(
        TARGET_WORKSPACE_SIZE_BYTES / (lcm_hidden_dim * stride)
    ) * (lcm_hidden_dim * stride)
    max_num_elements = buffer_size_in_bytes // stride
    mcast_buffer = McastGPUBuffer(
        buffer_size_in_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        device2str("cuda", mapping.local_rank),
        mapping.is_multi_node() or force_mn,
    )
    mcast_buffer.lamport_initialize(mapping.tp_rank, dtype)
    paddle.device.synchronize()
    mpi_barrier()
    buffer_flags = paddle.to_tensor(
        data=[0, 2, max_num_elements, 0, 0],
>>>>>>        dtype=torch.uint32,
        place=device2str("gpu", mapping.local_rank),
    )
    return mcast_buffer, buffer_flags, max_num_elements


def trtllm_mnnvl_all_reduce(
    inp: paddle.Tensor,
    multicast_buffer_ptr: int,
    buffer_ptrs_dev: int,
    buffer_M: int,
    buffer_flags_mnnvl: paddle.Tensor,
    nranks: int,
    rank: int,
    wait_for_results: bool,
    launch_with_pdl: bool,
    out: Optional[paddle.Tensor] = None,
) -> None:
    """Perform a multi-node NVLink all-reduce operation across multiple GPUs.

    This function performs an all-reduce (sum) operation using NVIDIA's multi-node NVLink (MNNVL)
    technology to efficiently combine tensors across multiple GPUs and nodes.

    There are 3 steps:
    1. scatter each GPU's input shard to the right unicast buffer
    2. perform all-reduce on each GPU
    3. broadcast the result to all GPUs

    Args:
        inp: Local Input Shard
        multicast_buffer_ptr: Pointer to the multicast buffer as an integer
        buffer_ptrs_dev: Pointer to device buffer pointers as an integer
        buffer_M: Maximum number of elements // hidden_dim
        buffer_flags_mnnvl: Tensor containing buffer state flags
        nranks: Total number of ranks participating in the all-reduce
        rank: Current process rank
        wait_for_results: If True, store the result to out
        launch_with_pdl: If True, launch using Programmatic Dependent Launch
        [Optional] out: Output tensor to store the result (required if wait_for_results is True)

    """
    module = get_trtllm_mnnvl_comm_module()
    module.trtllm_mnnvl_all_reduce(
        inp,
        multicast_buffer_ptr,
        buffer_ptrs_dev,
        buffer_M,
        buffer_flags_mnnvl,
        nranks,
        rank,
        wait_for_results,
        launch_with_pdl,
        out,
    )


def trtllm_mnnvl_fused_allreduce_rmsnorm(
    prenorm_output: paddle.Tensor,
    normed_output: paddle.Tensor,
    shard_input: paddle.Tensor,
    multicast_buffer_ptr: int,
    buffer_ptrs_dev: int,
    unicast_ptr: int,
    buffer_M: int,
    buffer_flags_mnnvl: paddle.Tensor,
    nranks: int,
    rank: int,
    gamma: paddle.Tensor,
    epsilon: float,
    residual: paddle.Tensor,
    launch_with_pdl: bool,
) -> None:
    """Performs MNNVL TwoShot Allreduce + RMSNorm.

    This function performs a multi-node all-reduce (sum) operation by first calling trtllm_mnnvl_all_reduce on the shard_input.
    After this, it performs RMSNorm on the all-reduced result, reading it directly from the multicast buffer.
    Note: multicast buffer is the same as the unicast buffer for the current rank.

    Args:
        prenorm_output: Output tensor for prenorm results
        normed_output: Output tensor for normalized results
        shard_input: Input tensor shard
        multicast_buffer_ptr: Pointer address as integer for multicast buffer
        buffer_ptrs_dev: Pointer address as integer for device buffer pointers
        unicast_ptr: Pointer address as integer for unicast buffer
        buffer_M: Maximum number of elements // hidden_dim
        buffer_flags_mnnvl: Buffer flags for synchronization
        nranks: Number of ranks in the tensor parallel group
        rank: Current rank in the tensor parallel group
        gamma: The gamma (norm weight) parameter for RMSNorm
        epsilon: The epsilon parameter for RMSNorm
        residual: The residual tensor to add
        launch_with_pdl: Whether to launch with PDL

    """
    trtllm_mnnvl_all_reduce(
        shard_input,
        multicast_buffer_ptr,
        buffer_ptrs_dev,
        buffer_M,
        buffer_flags_mnnvl,
        nranks,
        rank,
        False,
        launch_with_pdl,
        None,
    )
    get_trtllm_mnnvl_comm_module().trtllm_mnnvl_rmsnorm(
        unicast_ptr,
        prenorm_output,
        normed_output,
        gamma,
        epsilon,
        residual,
        buffer_flags_mnnvl,
        launch_with_pdl,
    )
