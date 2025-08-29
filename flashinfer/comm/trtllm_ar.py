import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import functools
import logging
from ctypes import c_void_p, cast
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec, sm100a_nvcc_flags
from ..utils import register_custom_op, round_up, version_at_least
from .cuda_ipc import create_shared_buffer, cudart, free_shared_buffer


class AllReduceStrategyType:
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6


class AllReduceStrategyConfig:
    USE_MEMCPY = 1 << 0
    PUSH_MODE = 1 << 1


class AllReduceFusionOp:
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8
    MOE_FINALIZE_ALLREDUCE_RESIDUAL_RMS_NORM = 9


class AllReduceFusionPattern:
    kAllReduce = 0
    kARResidualRMSNorm = 1
    kARResidualRMSNormFP8Quant = 2
    kARResidualRMSNormFP4Quant = 3
    kARResidualRMSNormOutFP8Quant = 4
    kARResidualRMSNormOutFP4Quant = 5


class QuantizationSFLayout:
    SWIZZLED_128x4 = 0
    SWIZZLED_8x4 = 1
    LINEAR = 2


def gen_trtllm_comm_module() -> JitSpec:
>>>>>>    gencode_flags = torch.utils.cpp_extension._get_cuda_arch_flags()
    has_sm100 = any(
        "compute_100" in flag for flag in gencode_flags
>>>>>>    ) and version_at_least(torch.version.cuda, "12.8")
    return gen_jit_spec(
        "trtllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce_fusion.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_allreduce_fusion.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags if has_sm100 else [],
    )


@functools.cache
def get_trtllm_comm_module():
    module = gen_trtllm_comm_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize", mutates_args=["buffer"]
    )
    def trtllm_lamport_initialize(
        buffer_ptr: int, size: int, dtype: paddle.dtype
    ) -> None:
        module.trtllm_lamport_initialize(buffer_ptr, size, dtype)

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize_all",
        mutates_args=["buffer_0_ptr", "buffer_1_ptr", "buffer_2_ptr", "size", "dtype"],
    )
    def trtllm_lamport_initialize_all(
        buffer_0_ptr: int,
        buffer_1_ptr: int,
        buffer_2_ptr: int,
        size: int,
        dtype: paddle.dtype,
    ) -> None:
        module.trtllm_lamport_initialize_all(
            buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
        )

    @register_custom_op(
        "flashinfer::trtllm_custom_all_reduce",
        mutates_args=[
            "inp",
            "out",
            "tp_size",
            "tp_rank",
            "token_num",
            "fusion_op_code",
            "strategy_code",
            "config_code",
            "launch_with_pdl",
            "flag_value",
            "peer_comm_buffer_ptrs",
            "peer_barrier_ptrs_in",
            "peer_barrier_ptrs_out",
            "bias",
            "residual",
            "weight",
            "weight_pre_residual_norm",
            "eps",
            "intermediate_buffer",
            "lamport_peer_comm_buffer_ptrs_0",
            "lamport_peer_comm_buffer_ptrs_1",
            "lamport_peer_comm_buffer_ptrs_2",
        ],
    )
    def trtllm_custom_all_reduce(
        inp: paddle.Tensor,
        out: paddle.Tensor,
        tp_size: int,
        tp_rank: int,
        token_num: int,
        fusion_op_code: AllReduceFusionOp,
        strategy_code: AllReduceStrategyType,
        config_code: AllReduceStrategyConfig,
        launch_with_pdl: bool,
        flag_value: int,
        peer_comm_buffer_ptrs: paddle.Tensor,
        peer_barrier_ptrs_in: paddle.Tensor,
        peer_barrier_ptrs_out: paddle.Tensor,
        bias: Optional[paddle.Tensor],
        residual: Optional[paddle.Tensor],
        weight: Optional[paddle.Tensor],
        weight_pre_residual_norm: Optional[paddle.Tensor],
        eps: Optional[float],
        intermediate_buffer: Optional[paddle.Tensor],
        lamport_peer_comm_buffer_ptrs_0: Optional[paddle.Tensor],
        lamport_peer_comm_buffer_ptrs_1: Optional[paddle.Tensor],
        lamport_peer_comm_buffer_ptrs_2: Optional[paddle.Tensor],
    ) -> None:
        module.trtllm_custom_all_reduce(
            inp,
            out,
            tp_size,
            tp_rank,
            token_num,
            fusion_op_code,
            strategy_code,
            config_code,
            launch_with_pdl,
            flag_value,
            peer_comm_buffer_ptrs,
            peer_barrier_ptrs_in,
            peer_barrier_ptrs_out,
            bias,
            residual,
            weight,
            weight_pre_residual_norm,
            eps,
            intermediate_buffer,
            lamport_peer_comm_buffer_ptrs_0,
            lamport_peer_comm_buffer_ptrs_1,
            lamport_peer_comm_buffer_ptrs_2,
        )

    @register_custom_op(
        "flashinfer::trtllm_allreduce_fusion",
        mutates_args=[
            "allreduce_in",
            "world_size",
            "world_rank",
            "token_num",
            "hidden_dim",
            "workspace_ptrs",
            "launch_with_pdl",
            "use_oneshot",
            "trigger_completion_at_end",
            "fp32_acc",
            "pattern_code",
            "allreduce_out",
            "residual_in",
            "residual_out",
            "norm_out",
            "quant_out",
            "scale_out",
            "rms_gamma",
            "rms_eps",
            "scale_factor",
            "layout_code",
        ],
    )
    def trtllm_allreduce_fusion(
        allreduce_in: paddle.Tensor,
        world_size: int,
        world_rank: int,
        token_num: int,
        hidden_dim: int,
        workspace_ptrs: paddle.Tensor,
        launch_with_pdl: bool,
        use_oneshot: bool,
        trigger_completion_at_end: bool,
        fp32_acc: bool,
        pattern_code: AllReduceFusionPattern,
        allreduce_out: Optional[paddle.Tensor],
        residual_in: Optional[paddle.Tensor],
        residual_out: Optional[paddle.Tensor],
        norm_out: Optional[paddle.Tensor],
        quant_out: Optional[paddle.Tensor],
        scale_out: Optional[paddle.Tensor],
        rms_gamma: Optional[paddle.Tensor],
        rms_eps: Optional[float],
        scale_factor: Optional[Union[paddle.Tensor, float]],
        layout_code: Optional[QuantizationSFLayout],
    ) -> None:
        module.trtllm_allreduce_fusion(
            allreduce_in,
            world_size,
            world_rank,
            token_num,
            hidden_dim,
            workspace_ptrs,
            launch_with_pdl,
            use_oneshot,
            trigger_completion_at_end,
            fp32_acc,
            pattern_code,
            allreduce_out,
            residual_in,
            residual_out,
            norm_out,
            quant_out,
            scale_out,
            rms_gamma,
            rms_eps,
            scale_factor,
            layout_code,
        )

    @register_custom_op(
        "flashinfer::trtllm_moe_allreduce_fusion",
        mutates_args=[
            "out",
            "tp_size",
            "tp_rank",
            "token_num",
            "hidden_dim",
            "workspace_ptrs",
            "launch_with_pdl",
            "residual_in",
            "rms_gamma",
            "rms_eps",
            "scale_factor",
            "moe_reduction_device_num_experts",
            "moe_reduction_scale_input",
            "moe_reduction_active_experts_token_input",
            "moe_reduction_token_input",
            "layout_code",
            "allreduce_out",
            "residual_out",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
    )
    def trtllm_moe_allreduce_fusion(
        world_size: int,
        world_rank: int,
        token_num: int,
        hidden_dim: int,
        workspace_ptrs: paddle.Tensor,
        launch_with_pdl: bool,
        residual_in: paddle.Tensor,
        rms_gamma: paddle.Tensor,
        rms_eps: float,
        scale_factor: float,
        moe_reduction_device_num_experts: int,
        moe_reduction_scale_input: paddle.Tensor,
        moe_reduction_active_experts_token_input: paddle.Tensor,
        moe_reduction_token_input: paddle.Tensor,
        layout_code: Optional[QuantizationSFLayout],
        moe_allreduce_out: Optional[paddle.Tensor],
        residual_out: Optional[paddle.Tensor],
        norm_out: Optional[paddle.Tensor],
        quant_out: Optional[paddle.Tensor],
        scale_out: Optional[paddle.Tensor],
    ) -> None:
        module.trtllm_moe_allreduce_fusion(
            world_size,
            world_rank,
            token_num,
            hidden_dim,
            workspace_ptrs,
            launch_with_pdl,
            residual_in,
            rms_gamma,
            rms_eps,
            scale_factor,
            moe_reduction_device_num_experts,
            moe_reduction_scale_input,
            moe_reduction_active_experts_token_input,
            moe_reduction_token_input,
            layout_code,
            moe_allreduce_out,
            residual_out,
            norm_out,
            quant_out,
            scale_out,
        )

    @register_custom_op(
        "flashinfer::trtllm_moe_finalize_allreduce_fusion",
        mutates_args=["residual_out", "norm_out"],
    )
    def trtllm_moe_finalize_allreduce_fusion(
        allreduce_in: paddle.Tensor,
        residual_in: paddle.Tensor,
        norm_weight: paddle.Tensor,
        expanded_idx_to_permuted_idx: paddle.Tensor,
        norm_out: paddle.Tensor,
        residual_out: paddle.Tensor,
        launch_with_pdl: bool,
        workspace: paddle.Tensor,
        world_rank: int,
        world_size: int,
        eps: float,
        shared_expert_output: Optional[paddle.Tensor],
        expert_scale_factor: Optional[paddle.Tensor],
    ) -> None:
        module.trtllm_moe_finalize_allreduce_fusion(
            allreduce_in,
            residual_in,
            norm_weight,
            expanded_idx_to_permuted_idx,
            norm_out,
            residual_out,
            launch_with_pdl,
            workspace,
            world_rank,
            world_size,
            eps,
            shared_expert_output,
            expert_scale_factor,
        )

    return SimpleNamespace(
        trtllm_lamport_initialize=trtllm_lamport_initialize,
        trtllm_lamport_initialize_all=trtllm_lamport_initialize_all,
        trtllm_custom_all_reduce=trtllm_custom_all_reduce,
        trtllm_allreduce_fusion=trtllm_allreduce_fusion,
        trtllm_moe_allreduce_fusion=trtllm_moe_allreduce_fusion,
        trtllm_moe_finalize_allreduce_fusion=trtllm_moe_finalize_allreduce_fusion,
    )


OneShotMaxToken = 128
MAX_ALL_REDUCE_BLOCKS = 24
LamportTokenNumThreshold = 16


def trtllm_create_ipc_workspace_for_all_reduce(
    rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
>>>>>>    group: Optional[torch.distributed.ProcessGroup] = None,
) -> List[List[int]]:
    """
    Parameters:
    - rank: the rank of the current process.
    - tp_size: the size of the process group.
    - max_token_num: the maximum number of tokens in a sequence.
    - hidden_dim: the dimension of the hidden states.
    - group: the process group to use.

    Note:
    This function is used to create a workspace for all reduce.
    The workspace is a list of IPC handles.
    The workspace should be initialized before calling trtllm_custom_all_reduce.
    The workspace should be destroyed after calling trtllm_custom_all_reduce.
    The workspace can be reused for multiple all reduce calls under the same configuration.

    We would init 7 IPC buffers for trtllm_custom_all_reduce.
    They are sized as follows:
    [buffer_size, buffer_size, flag_size, flag_size, lamport_buffer_size, lamport_buffer_size, lamport_buffer_size]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(float) * (maxBeamWidth)
    - flag_size: (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t) * tp_size * 2
    - lamport_buffer_size: tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * sizeof(half)

    They are for:
    ipcHandles[0] - peer_comm_buffer_ptrs
    ipcHandles[2] - peer_barrier_ptrs_in
    ipcHandles[3] - peer_barrier_ptrs_out
    ipcHandles[4] - lamport_peer_comm_buffer_ptrs[0:tp_size]
    ipcHandles[5] - lamport_peer_comm_buffer_ptrs[tp_size:tp_size * 2]
    ipcHandles[6] - lamport_peer_comm_buffer_ptrs[tp_size * 2:tp_size * 3]

    We use tp_size and world_size here interchangeably (customAllReduce).

    Reference: trtllm, cpp/tests/unit_tests/kernels/allReduce/allReduceKernelTest.cu, Workspace init
    """
    buffer_size = tp_size * max_token_num * hidden_dim * 4
    FLAG_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * 4
    flag_size = FLAG_SIZE * tp_size * 2
    lamport_buffer_size = tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * 2
    ipc_handles = list()
    for size in [
        buffer_size,
        buffer_size,
        flag_size,
        flag_size,
        lamport_buffer_size,
        lamport_buffer_size,
        lamport_buffer_size,
    ]:
        aligned_size = round_up(size, 1 << 21)
        ipc_handles.append(create_shared_buffer(aligned_size, group))
    print(
        f"rank {rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )
    trtllm_lamport_initialize_all(
        ipc_handles[4][rank],
        ipc_handles[5][rank],
        ipc_handles[6][rank],
        lamport_buffer_size // 2,
        "float16",
    )
    paddle.distributed.barrier(group=group)
    return ipc_handles


def trtllm_destroy_ipc_workspace_for_all_reduce(
>>>>>>    workspace: List[List[int]], group: Optional[torch.distributed.ProcessGroup] = None
) -> None:
    """
    Note:
    This function is used to destroy a workspace for all reduce.
    The workspace is a list of IPC handles.
    The workspace should be destroyed after calling trtllm_custom_all_reduce.
    The workspace can be reused for multiple all reduce calls under the same configuration.
    """
    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


BarrierFlagCount = 256
MAX_COMM_SIZE = 2147483647 & ~((1 << 21) - 1)


def trtllm_create_ipc_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    use_fp32_lamport: bool = False,
>>>>>>    group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tuple[List[List[int]], paddle.Tensor]:
    """
    Parameters:
    - tp_rank: the rank of the current process.
    - tp_size: the size of the process group.
    - max_token_num: the maximum number of tokens in a sequence.
    - hidden_dim: the dimension of the hidden states.
    - use_fp32_lamport: if True, we will use fp32 datatype in allreduce fusion.
    - group: the process group to use.

    Note:
    We would init 3 IPC buffers for trtllm_custom_all_reduce_fusion.
    They are sized as follows:
    [buffer_size, flag_size, lamport_buffer_size * 3]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(half)
    - flag_size: tp_size * BarrierFlagCount * sizeof(int)
    - lamport_buffer_size: tp_size * max(max_token_num, OneShotMaxToken) * tp_size * hidden_dim * sizeof(half)

    The workspace is passed as workspace field in AllReduceFusionParams.

    We use tp_size and world_size here interchangeably (allReduceFusion).

    Reference: trtllm, cpp/tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.cu, Workspace init
    """
    buffer_size = tp_size * max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    lamport_comm_size = (
        tp_size * max_token_num * hidden_dim * 2
        if not use_fp32_lamport
        else tp_size * max_token_num * hidden_dim * 4
    )
    if lamport_comm_size > MAX_COMM_SIZE:
        logging.warning(
            f"warning: lamport_comm_size {lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}, set to MAX_COMM_SIZE"
        )
        lamport_comm_size = MAX_COMM_SIZE
    lamport_buffer_size = lamport_comm_size * 3
    ipc_handles: List[List[int]] = list()
    for size in [buffer_size, flag_size, lamport_buffer_size]:
        aligned_size = round_up(size, 1 << 21)
        ipc_handles.append(create_shared_buffer(aligned_size, group))
    print(
        f"rank {tp_rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )
    aligned_lamport_buffer_size = round_up(lamport_buffer_size, 1 << 21)
    if use_fp32_lamport:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], aligned_lamport_buffer_size // 4, "float32"
        )
    else:
        trtllm_lamport_initialize(
            ipc_handles[2][tp_rank], aligned_lamport_buffer_size // 2, "float16"
        )
    workspace = list()
    for ipc_handle in ipc_handles:
        for rank in range(tp_size):
            workspace.append(ipc_handle[rank])
    """
    NOTE:
    The flags are for the lamport communication states.
    atomic flag read counter: kernel_flag_ptr[0] = 0;
    non-lamport flag: kernel_flag_ptr[1] = 0;
    lamport flag: kernel_flag_ptr[2] = 0;
    lamport triple buffer offset: kernel_flag_ptr[3] = lamport_comm_size;
    lamport clear size: kernel_flag_ptr[4] = 0;
    """
    flag_ptr = cudart.cudaMalloc(5 * 4)
    cudart.cudaMemset(flag_ptr, 0, 5 * 4)
    lamport_comm_size_bytes = lamport_comm_size.to_bytes(4, byteorder="little")
    cudart.cudaMemcpy(
        c_void_p(flag_ptr.value + 3 * 4), cast(lamport_comm_size_bytes, c_void_p), 4
    )
    print("set flag_ptr[3] = lamport_comm_size: ", lamport_comm_size)
    workspace.append(flag_ptr.value)
    for i in range(len(workspace)):
        print(f"Rank {tp_rank} workspace[{i}] {hex(workspace[i])}")
    workspace_tensor = paddle.to_tensor(
        data=workspace, dtype="int64", place=device2str("gpu")
    )
    paddle.distributed.barrier(group=group)
    return ipc_handles, workspace_tensor


def trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
>>>>>>    workspace: List[List[int]], group: Optional[torch.distributed.ProcessGroup] = None
) -> None:
    """
    Parameters:
    - workspace: the workspace to destroy.
    - group: the process group to use.

    Note:
    This function is used to destroy a workspace for all reduce fusion.
    The workspace is a list of IPC handles.
    The workspace should be destroyed after calling trtllm_custom_all_reduce_fusion.
    The workspace can be reused for multiple all reduce fusion calls under the same configuration.
    """
    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


def compute_fp4_swizzled_layout_sf_size(total_row, total_column):
    """
    Helper function to compute the padded size of the fp4 swizzled layout.

    Parameters:
    - total_row: the total number of rows.
    - total_column: the total number of columns.
    """

    def pad_up(x, y):
        return (x + y - 1) // y * y

    padded_row = pad_up(total_row, 128)
    padded_column = pad_up(total_column, 4)
    return padded_row * padded_column


def trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: paddle.dtype) -> None:
    get_trtllm_comm_module().trtllm_lamport_initialize(buffer_ptr, size, dtype)


def trtllm_lamport_initialize_all(
    buffer_0_ptr: int,
    buffer_1_ptr: int,
    buffer_2_ptr: int,
    size: int,
    dtype: paddle.dtype,
) -> None:
    """
    Initialize 3 lamport buffers by negative zero.

    Parameters:
    - buffer_0_ptr: the pointer to the first buffer.
    - buffer_1_ptr: the pointer to the second buffer.
    - buffer_2_ptr: the pointer to the third buffer.
    - size: the size of the buffer.
    - dtype: the data type of the buffer.
    """
    get_trtllm_comm_module().trtllm_lamport_initialize_all(
        buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
    )


def trtllm_custom_all_reduce(
    inp: paddle.Tensor,
    out: paddle.Tensor,
    tp_size: int,
    tp_rank: int,
    token_num: int,
    fusion_op_code: AllReduceFusionOp,
    strategy_code: AllReduceStrategyType,
    config_code: AllReduceStrategyConfig,
    launch_with_pdl: bool,
    flag_value: int,
    peer_comm_buffer_ptrs: paddle.Tensor,
    peer_barrier_ptrs_in: paddle.Tensor,
    peer_barrier_ptrs_out: paddle.Tensor,
    bias: Optional[paddle.Tensor],
    residual: Optional[paddle.Tensor],
    weight: Optional[paddle.Tensor],
    weight_pre_residual_norm: Optional[paddle.Tensor],
    eps: Optional[float],
    intermediate_buffer: Optional[paddle.Tensor],
    lamport_peer_comm_buffer_ptrs_0: Optional[paddle.Tensor],
    lamport_peer_comm_buffer_ptrs_1: Optional[paddle.Tensor],
    lamport_peer_comm_buffer_ptrs_2: Optional[paddle.Tensor],
) -> None:
    """
    Parameters:
    - inp: the input tensor. [token_num, hidden_dim]
    - out: the output tensor. [token_num, hidden_dim]
    - tp_size: the size of the process group.
    - tp_rank: the rank of the current process.
    - token_num: the number of tokens in the sequence.
    - fusion_op_code: the fusion operation code.
    - strategy_code: the strategy code.
    - config_code: the config code.
    - launch_with_pdl: whether to launch with pdl.
    - flag_value: the flag value.
    - peer_comm_buffer_ptrs: the peer communication buffer pointers.
    - peer_barrier_ptrs_in: the peer barrier pointers in.
    - peer_barrier_ptrs_out: the peer barrier pointers out.
    - bias: the bias tensor. [hidden_dim]
    - residual: the residual tensor. [token_num, hidden_dim]
    - weight: the weight tensor. [hidden_dim]
    - weight_pre_residual_norm: the weight pre residual norm tensor. [hidden_dim]
    - eps: the epsilon value.
    - intermediate_buffer: the intermediate buffer tensor.
    - lamport_peer_comm_buffer_ptrs_0: the lamport peer communication buffer pointers 0.
    - lamport_peer_comm_buffer_ptrs_1: the lamport peer communication buffer pointers 1.
    - lamport_peer_comm_buffer_ptrs_2: the lamport peer communication buffer pointers 2.
    """
    get_trtllm_comm_module().trtllm_custom_all_reduce(
        inp,
        out,
        tp_size,
        tp_rank,
        token_num,
        fusion_op_code,
        strategy_code,
        config_code,
        launch_with_pdl,
        flag_value,
        peer_comm_buffer_ptrs,
        peer_barrier_ptrs_in,
        peer_barrier_ptrs_out,
        bias,
        residual,
        weight,
        weight_pre_residual_norm,
        eps,
        intermediate_buffer,
        lamport_peer_comm_buffer_ptrs_0,
        lamport_peer_comm_buffer_ptrs_1,
        lamport_peer_comm_buffer_ptrs_2,
    )


def trtllm_allreduce_fusion(
    allreduce_in: paddle.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: paddle.Tensor,
    launch_with_pdl: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    pattern_code: AllReduceFusionPattern,
    use_oneshot: Optional[bool],
    allreduce_out: Optional[paddle.Tensor],
    residual_in: Optional[paddle.Tensor],
    residual_out: Optional[paddle.Tensor],
    norm_out: Optional[paddle.Tensor],
    quant_out: Optional[paddle.Tensor],
    scale_out: Optional[paddle.Tensor],
    rms_gamma: Optional[paddle.Tensor],
    rms_eps: Optional[float],
    scale_factor: Optional[Union[paddle.Tensor, float]],
    layout_code: Optional[QuantizationSFLayout],
) -> None:
    """
    Parameters:
    - allreduce_in: the input tensor. [token_num, hidden_dim]
    - world_size: the size of the process group.
    - world_rank: the rank of the current process.
    - token_num: the number of tokens in the sequence.
    - hidden_dim: the dimension of the hidden states.
    - workspace_ptrs: the workspace pointers.
    - launch_with_pdl: whether to launch with pdl.
    - use_oneshot: whether to use oneshot.
    - trigger_completion_at_end: whether to trigger completion at the end.
    - fp32_acc: whether to use fp32 accumulation.
    - pattern_code: the pattern code.
    - allreduce_out: the output tensor. [token_num, hidden_dim]
    - residual_in: the residual input tensor. [token_num, hidden_dim]
    - residual_out: the residual output tensor. [token_num, hidden_dim]
    - norm_out: the norm output tensor. [token_num, hidden_dim]
    - quant_out: the quant output tensor. [token_num, hidden_dim]
    - scale_out: the scale output tensor. Initialization referece: tests/test_trtllm_allreduce_fusion.py
    - rms_gamma: the rms gamma tensor. [hidden_dim]
    - rms_eps: the rms epsilon value.
    - scale_factor: the scale factor. For cudaGraphs safety, it should be a tensor.
    - layout_code: the layout code.

    Note:
    Regarding the `use_oneshot` parameter, you could force to use the one-shot strategy based on your use case.
    Otherwise, it would be enabled if token_num is less than the one-shot max token number (currently 128) for min-latency mode.
    """
    if use_oneshot is None:
        use_oneshot = token_num <= 128
    if not use_oneshot:
        assert token_num > world_size, "sequence length should be larger than tp_size"
    required_lamport_comm_size = (
        token_num * hidden_dim * 2 * world_size
        if allreduce_in.dtype != "float32"
        else token_num * hidden_dim * 4 * world_size
    )
    if required_lamport_comm_size > MAX_COMM_SIZE and use_oneshot:
        logging.warning(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}. Cannot use oneshot in this case."
        )
        use_oneshot = False
    if scale_factor is not None:
        if isinstance(scale_factor, paddle.Tensor):
            scale_factor = scale_factor.to("float32")
        else:
            scale_factor = paddle.to_tensor(
                data=[scale_factor], dtype="float32", place=allreduce_in.place
            )
    get_trtllm_comm_module().trtllm_allreduce_fusion(
        allreduce_in=allreduce_in,
        world_size=world_size,
        world_rank=world_rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        use_oneshot=use_oneshot,
        trigger_completion_at_end=trigger_completion_at_end,
        fp32_acc=fp32_acc,
        pattern_code=pattern_code,
        allreduce_out=allreduce_out,
        residual_in=residual_in,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
    )


def trtllm_moe_allreduce_fusion(
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: paddle.Tensor,
    launch_with_pdl: bool,
    residual_in: paddle.Tensor,
    rms_gamma: paddle.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: paddle.Tensor,
    moe_reduction_active_experts_token_input: paddle.Tensor,
    moe_reduction_token_input: paddle.Tensor,
    layout_code: Optional[QuantizationSFLayout],
    moe_allreduce_out: Optional[paddle.Tensor],
    residual_out: Optional[paddle.Tensor],
    norm_out: Optional[paddle.Tensor],
    quant_out: Optional[paddle.Tensor],
    scale_out: Optional[paddle.Tensor],
) -> None:
    """
    Parameters:
    - world_size: the size of the process group.
    - world_rank: the rank of the current process.
    - token_num: the number of tokens in the sequence.
    - hidden_dim: the dimension of the hidden states.
    - workspace_ptrs: the workspace pointers.
    - launch_with_pdl: whether to launch with pdl.
    - residual_in: the residual input tensor. [token_num, hidden_dim]
    - rms_gamma: the rms gamma tensor. [hidden_dim]
    - rms_eps: the rms epsilon value.
    - scale_factor: the scale factor.
    - moe_reduction_device_num_experts: the number of experts.
    - moe_reduction_scale_input: the scale input tensor. [token_num, hidden_dim]
    - moe_reduction_active_experts_token_input: the active experts token input tensor. [token_num, hidden_dim]
    - moe_reduction_token_input: the token input tensor. [token_num, hidden_dim]
    - layout_code: the layout code.
    - moe_allreduce_out: the moe allreduce output tensor. [token_num, hidden_dim]
    - residual_out: the residual output tensor. [token_num, hidden_dim]
    - norm_out: the norm output tensor. [token_num, hidden_dim]
    - quant_out: the quant output tensor. [token_num // 4, hidden_dim], fp16/bf16 -> fp4
    - scale_out: the scale output tensor. Initialization referece: tests/test_trtllm_moe_allreduce_fusion.py
    """
    required_lamport_comm_size = moe_reduction_token_input.size * 2 * world_size
    if required_lamport_comm_size > MAX_COMM_SIZE:
        raise ValueError(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}. Cannot use oneshot in this case."
        )
    get_trtllm_comm_module().trtllm_moe_allreduce_fusion(
        world_size=world_size,
        world_rank=world_rank,
        token_num=token_num,
        hidden_dim=hidden_dim,
        workspace_ptrs=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        residual_in=residual_in,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        moe_reduction_device_num_experts=moe_reduction_device_num_experts,
        moe_reduction_scale_input=moe_reduction_scale_input,
        moe_reduction_active_experts_token_input=moe_reduction_active_experts_token_input,
        moe_reduction_token_input=moe_reduction_token_input,
        layout_code=layout_code,
        moe_allreduce_out=moe_allreduce_out,
        residual_out=residual_out,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
    )


def trtllm_moe_finalize_allreduce_fusion(
    allreduce_in: paddle.Tensor,
    residual_in: paddle.Tensor,
    norm_weight: paddle.Tensor,
    expanded_idx_to_permuted_idx: paddle.Tensor,
    norm_out: paddle.Tensor,
    residual_out: paddle.Tensor,
    workspace_ptrs: paddle.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: Optional[paddle.Tensor],
    expert_scale_factor: Optional[paddle.Tensor],
) -> None:
    """
    Parameters:
    - allreduce_in: the input tensor. [token_num, top_k, hidden_dim]
    - residual_in: the residual input tensor. [token_num, hidden_dim]
    - norm_weight: the norm weight tensor. [hidden_dim]
    - expanded_idx_to_permuted_idx: the expanded index to permuted index tensor. [token_num, top_k]
    - norm_out: the norm output tensor. [token_num, hidden_dim]
    - residual_out: the residual output tensor. [token_num, hidden_dim]
    - workspace_ptrs: the workspace pointers.
    - launch_with_pdl: whether to launch with pdl.
    - world_rank: the rank of the current process.
    - world_size: the size of the process group.
    - eps: the epsilon value.
    - shared_expert_output: the shared expert output tensor. [token_num, hidden_dim]
    - expert_scale_factor: the expert scale factor tensor. [token_num, top_k]
    """
    required_lamport_comm_size = allreduce_in.size * 2 * world_size
    if required_lamport_comm_size > MAX_COMM_SIZE:
        raise ValueError(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater than MAX_COMM_SIZE {MAX_COMM_SIZE}. Cannot use oneshot in this case."
        )
    get_trtllm_comm_module().trtllm_moe_finalize_allreduce_fusion(
        allreduce_in=allreduce_in,
        residual_in=residual_in,
        norm_weight=norm_weight,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        norm_out=norm_out,
        residual_out=residual_out,
        workspace=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        world_rank=world_rank,
        world_size=world_size,
        eps=eps,
        shared_expert_output=shared_expert_output,
        expert_scale_factor=expert_scale_factor,
    )
