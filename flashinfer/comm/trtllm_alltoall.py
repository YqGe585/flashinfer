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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec
from ..utils import register_custom_op
from .mapping import Mapping
from .mnnvl import MnnvlMemory


def gen_comm_alltoall_module() -> JitSpec:
    return gen_jit_spec("comm", [jit_env.FLASHINFER_CSRC_DIR / "trtllm_alltoall.cu"])


@functools.cache
def get_comm_alltoall_module():
    module = gen_comm_alltoall_module().build_and_load()

    @register_custom_op("flashinfer::moe_comm_prepare_indices", mutates_args=[])
    def moe_comm_prepare_indices(
        gathered_target_rank_ids: paddle.Tensor,
        real_rank_token_count_cum_sum: Optional[paddle.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ) -> Tuple[
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
        paddle.Tensor,
    ]:
        device = gathered_target_rank_ids.place
        max_send_ranks_per_token = max(top_k, ep_size)
        local_gather_indices = paddle.empty(
            shape=max_token_count_per_rank * ep_size, dtype="int32"
        )
        send_rank_count_cum_sum = paddle.empty(shape=(ep_size,), dtype="int32")
        send_rank_local_indices = paddle.empty(
            shape=max_token_count_per_rank * max_send_ranks_per_token, dtype="int32"
        )
        recv_rank_count_cum_sum = paddle.empty(shape=ep_size, dtype="int32")
        recv_rank_local_indices = paddle.empty(
            shape=max_token_count_per_rank * ep_size, dtype="int32"
        )
        backward_recv_rank_local_indice = paddle.empty(
            shape=max_token_count_per_rank * max_send_ranks_per_token, dtype="int32"
        )
        module.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cum_sum,
            local_gather_indices,
            send_rank_count_cum_sum,
            send_rank_local_indices,
            recv_rank_count_cum_sum,
            recv_rank_local_indices,
            backward_recv_rank_local_indice,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )
        return (
            local_gather_indices,
            send_rank_count_cum_sum,
            send_rank_local_indices,
            recv_rank_count_cum_sum,
            recv_rank_local_indices,
            backward_recv_rank_local_indice,
        )

    @register_custom_op(
        "flashinfer::moe_local_gather",
        mutates_args=["local_expert_ids", "local_scales"],
    )
    def moe_local_gather(
        recv_rank_cum_sum: paddle.Tensor,
        local_gather_indices: paddle.Tensor,
        gathered_expert_ids: paddle.Tensor,
        gathered_scales: paddle.Tensor,
        local_expert_ids: paddle.Tensor,
        local_scales: paddle.Tensor,
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        module.moe_local_gather(
            recv_rank_cum_sum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

    @register_custom_op("flashinfer::moe_comm", mutates_args=["output"])
    def moe_comm(
        input: paddle.Tensor,
        send_rank_cum_sum: paddle.Tensor,
        send_indices: paddle.Tensor,
        output: paddle.Tensor,
        recv_rank_cum_sum: paddle.Tensor,
        recv_indices: paddle.Tensor,
        all_workspaces: paddle.Tensor,
        ep_rank: int,
        ep_size: int,
    ) -> None:
        module.moe_comm(
            input,
            send_rank_cum_sum,
            send_indices,
            output,
            recv_rank_cum_sum,
            recv_indices,
            all_workspaces,
            ep_rank,
            ep_size,
        )

    @register_custom_op("flashinfer::set_moe_max_usable_sm_count", mutates_args=[])
    def set_moe_max_usable_sm_count(max_sm_count: int) -> None:
        module.set_moe_max_usable_sm_count(max_sm_count)

    @register_custom_op(
        "flashinfer::get_moe_commworkspace_size_per_rank", mutates_args=[]
    )
    def get_moe_commworkspace_size_per_rank(ep_size: int) -> int:
        return module.get_moe_commworkspace_size_per_rank(ep_size)

    return SimpleNamespace(
        moe_comm_prepare_indices=moe_comm_prepare_indices,
        moe_local_gather=moe_local_gather,
        moe_comm=moe_comm,
        set_moe_max_usable_sm_count=set_moe_max_usable_sm_count,
        get_moe_commworkspace_size_per_rank=get_moe_commworkspace_size_per_rank,
    )


def moe_comm_prepare_indices(
    gathered_target_rank_ids: paddle.Tensor,
    real_rank_token_count_cum_sum: Optional[paddle.Tensor],
    max_token_count_per_rank: int,
    expert_count: int,
    top_k: int,
    ep_rank: int,
    ep_size: int,
) -> Tuple[
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
]:
    return get_comm_alltoall_module().moe_comm_prepare_indices(
        gathered_target_rank_ids,
        real_rank_token_count_cum_sum,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )


def moe_local_gather(
    recv_rank_cum_sum: paddle.Tensor,
    local_gather_indices: paddle.Tensor,
    gathered_expert_ids: paddle.Tensor,
    gathered_scales: paddle.Tensor,
    local_expert_ids: paddle.Tensor,
    local_scales: paddle.Tensor,
    max_token_count_per_rank: int,
    expert_count: int,
    top_k: int,
    ep_rank: int,
    ep_size: int,
) -> None:
    get_comm_alltoall_module().moe_local_gather(
        recv_rank_cum_sum,
        local_gather_indices,
        gathered_expert_ids,
        gathered_scales,
        local_expert_ids,
        local_scales,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )


def moe_comm(
    input: paddle.Tensor,
    send_rank_cum_sum: paddle.Tensor,
    send_indices: paddle.Tensor,
    output: paddle.Tensor,
    recv_rank_cum_sum: paddle.Tensor,
    recv_indices: paddle.Tensor,
    all_workspaces: paddle.Tensor,
    ep_rank: int,
    ep_size: int,
) -> None:
    get_comm_alltoall_module().moe_comm(
        input,
        send_rank_cum_sum,
        send_indices,
        output,
        recv_rank_cum_sum,
        recv_indices,
        all_workspaces,
        ep_rank,
        ep_size,
    )


def set_moe_max_usable_sm_count(max_sm_count: int) -> None:
    get_comm_alltoall_module().set_moe_max_usable_sm_count(max_sm_count)


def get_moe_commworkspace_size_per_rank(ep_size: int) -> int:
    return get_comm_alltoall_module().get_moe_commworkspace_size_per_rank(ep_size)


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: paddle.Tensor
    send_rank_count_cumsum: paddle.Tensor
    send_rank_local_indices: paddle.Tensor
    recv_rank_count_cumsum: paddle.Tensor
    recv_rank_local_indices: paddle.Tensor
    backward_recv_rank_local_indices: paddle.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_workspace_tensor: paddle.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor
        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = get_moe_commworkspace_size_per_rank(mapping.tp_size)
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(
>>>>>>            torch.uint64
        )
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: paddle.Tensor, expert_count: int, ep_size: int
    ):
        assert (
            expert_count % ep_size == 0
        ), "expert_count should be divisible by ep_size"
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: paddle.Tensor,
        real_rank_token_count_cumsum: paddle.Tensor,
        gathered_expert_ids: paddle.Tensor,
        gathered_scales: paddle.Tensor,
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )
        local_token_allocation_count = max_token_count_per_rank * ep_size
        local_expert_ids = paddle.empty(
            shape=[local_token_allocation_count, top_k], dtype="int32"
        )
        local_scales = paddle.empty(
            shape=[local_token_allocation_count, top_k], dtype="float32"
        )
        moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )
        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: paddle.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: paddle.Tensor,
        ep_rank: int,
        ep_size: int,
    ):
        assert x.dim() == 2, "only 2D tensor supported, please reshape."
        output_tensor = paddle.empty(
            shape=[alltoall_info.local_token_allocation_count, tuple(x.shape)[1]],
            dtype=x.dtype,
        )
        moe_comm(
            x,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.send_rank_local_indices,
            output_tensor,
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            workspace,
            ep_rank,
            ep_size,
        )
        return output_tensor

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: paddle.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: paddle.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensor = paddle.zeros(
            shape=[token_count * top_k, tuple(x.shape)[1]], dtype=x.dtype
        )
        moe_comm(
            x,
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            output_tensor,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            ep_rank,
            ep_size,
        )
        return paddle.sum(
            x=output_tensor.reshape(token_count, top_k, tuple(x.shape)[1]),
            axis=1,
            keepdim=False,
        )
