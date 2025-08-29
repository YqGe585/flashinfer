import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

"""
Copyright (c) 2024 by FlashInfer team.

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
import pytest

import flashinfer.comm.trtllm_alltoall as tllm_alltoall

has_setup_max_sm_count = False


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up test environment and warm up JIT compilation."""
    global has_setup_max_sm_count
    if not has_setup_max_sm_count:
        sm_count = paddle.device.cuda.get_device_properties(
            device="gpu:0"
        ).multi_processor_count
        max_sm_count = sm_count // 8
        tllm_alltoall.set_moe_max_usable_sm_count(max_sm_count)
        has_setup_max_sm_count = True
    paddle.seed(seed=4660)
    yield


SINGLE_GPU_PARAMS = [
    (902, 701, 32768, 100, "float16"),
    (101, 75, 288, 10, "float16"),
    (10, 5, 8, 1, "float16"),
    (902, 701, 7168, 100, "bfloat16"),
    (101, 75, 288, 10, "bfloat16"),
]
MULTI_RANK_PARAMS = [
    (2, 5, 8, "float16"),
    (4, 901, 32768, "bfloat16"),
    (8, 16384, 128, "float16"),
]
PREPARE_INDICES_PARAMS = [
    (0, 8, 256, 4, 3, False),
    (1, 8, 256, 4, 3, True),
    (7, 8, 256, 8, 1025, False),
    (7, 64, 1024, 32, 1029, True),
]
LOCAL_GATHER_PARAMS = [(0, 8, 256, 4, 3), (7, 8, 256, 8, 32), (7, 64, 1024, 32, 1029)]
CROSS_GPU_PARAMS = [
    (2, 100, 256, "float16"),
    (2, 300, 512, "bfloat16"),
    (4, 150, 256, "float16"),
    (4, 400, 512, "float16"),
]


def get_available_gpu_count():
    """Get the number of available GPUs."""
    if not paddle.device.cuda.device_count() >= 1:
        return 0
    return paddle.device.cuda.device_count()


def requires_gpus(min_gpus):
    """Decorator to skip test if insufficient GPUs are available."""

    def decorator(func):
        return pytest.mark.skipif(
            get_available_gpu_count() < min_gpus,
            reason=f"Requires at least {min_gpus} GPUs, but only {get_available_gpu_count()} available",
        )(func)

    return decorator


@pytest.mark.parametrize(
    "input_entry_count,output_entry_count,vector_dim,send_recv_count,dtype",
    SINGLE_GPU_PARAMS,
)
def test_moe_alltoall_single_gpu(
    input_entry_count, output_entry_count, vector_dim, send_recv_count, dtype
):
    """Test MOE alltoall communication on single GPU."""
    paddle.device.set_device(device="gpu:0")
    input_tensor = paddle.randn(shape=[input_entry_count, vector_dim], dtype=dtype)
    output_tensor = paddle.zeros(shape=[output_entry_count, vector_dim], dtype=dtype)
    send_cumsum = paddle.ones(shape=(1,), dtype="int32") * send_recv_count
    recv_cumsum = paddle.ones(shape=(1,), dtype="int32") * send_recv_count
    send_indices = paddle.randperm(n=input_entry_count, dtype="int32")[:send_recv_count]
    recv_indices = paddle.randperm(n=output_entry_count, dtype="int32")[
        :send_recv_count
    ]
    ref_output_tensor = paddle.zeros(
        shape=[output_entry_count, vector_dim], dtype=dtype
    )
    ref_output_tensor[recv_indices] = input_tensor[send_indices]
    workspace_size = tllm_alltoall.get_moe_commworkspace_size_per_rank(1)
>>>>>>    all_workspaces = paddle.zeros(shape=[1, workspace_size], dtype=torch.uint64)
    tllm_alltoall.moe_comm(
        input_tensor,
        send_cumsum,
        send_indices,
        output_tensor,
        recv_cumsum,
        recv_indices,
        all_workspaces,
        0,
        1,
    )
    assert paddle.allclose(
        x=output_tensor, y=ref_output_tensor, atol=1e-05, rtol=1e-05
    ).item(), ""


@pytest.mark.parametrize(
    "world_size,input_entry_per_rank,vector_dim,dtype", MULTI_RANK_PARAMS
)
def test_moe_alltoall_multi_rank_single_gpu(
    world_size, input_entry_per_rank, vector_dim, dtype
):
    """Test MOE alltoall communication with multiple ranks on single GPU."""
    paddle.device.set_device(device="gpu:0")
    max_world_size = 8
    assert (
        world_size <= max_world_size
    ), f"should run with world_size at most {max_world_size}"
    input_tensor = paddle.randn(
        shape=[input_entry_per_rank * world_size, vector_dim], dtype=dtype
    )
    output_tensor = paddle.zeros(
        shape=[input_entry_per_rank * world_size, vector_dim], dtype=dtype
    )
    ref_output_tensor = paddle.zeros(
        shape=[input_entry_per_rank * world_size, vector_dim], dtype=dtype
    )
    target_rank_ids = paddle.randint(
        low=0,
        high=world_size,
        shape=(input_entry_per_rank * world_size,),
        dtype="int32",
    )
    input_tensors_all_ranks = list(
        paddle_split(x=input_tensor, num_or_sections=input_entry_per_rank)
    )
    target_rank_ids_all_ranks = list(
        paddle_split(x=target_rank_ids, num_or_sections=input_entry_per_rank)
    )
    send_ids_all_ranks = []
    send_counts_all_ranks = []
    send_cumsum_all_ranks = []
    send_start_end_all_ranks = []
    for rank in range(world_size):
        send_start_end = []
        local_target_rank_ids = target_rank_ids_all_ranks[rank]
        sorted_local_target_rank_ids, local_send_id = paddle.sort(
            x=local_target_rank_ids
        ), paddle.argsort(x=local_target_rank_ids)
        local_send_id = local_send_id.to("int32")
        padded_sorted_local_target_rank_ids = paddle.concat(
            x=(
                sorted_local_target_rank_ids,
                paddle.arange(dtype="int32", end=world_size),
            )
        )
        unique_target_rank_ids, local_send_counts = paddle.unique(
            x=padded_sorted_local_target_rank_ids, return_counts=True
        )
        local_send_counts = local_send_counts.to("int32")
        assert (
            unique_target_rank_ids.size == world_size
        ), "unique_target_rank_ids must be equal to world_size"
        local_send_counts -= 1
        local_send_cumsum = paddle.cumsum(x=local_send_counts, axis=0).to("int32")
        send_ids_all_ranks.append(local_send_id)
        send_counts_all_ranks.append(local_send_counts)
        send_cumsum_all_ranks.append(local_send_cumsum)
        local_send_cumsum_cpu = local_send_cumsum.cpu().tolist()
        for i in range(len(local_send_cumsum_cpu)):
            send_start_end.append(
                (local_send_cumsum_cpu[i - 1] if i > 0 else 0, local_send_cumsum_cpu[i])
            )
        send_start_end_all_ranks.append(send_start_end)
    recv_ids_all_ranks = []
    recv_cumsum_all_ranks = []
    output_tensors_all_ranks = []
    total_recv_all_ranks_cpu = []
    output_indice_offset = 0
    output_start_current_rank = 0
    for rank in range(world_size):
        local_recv_counts = paddle.zeros(shape=world_size, dtype="int32")
        for other_rank in range(world_size):
            local_recv_counts[other_rank] = send_counts_all_ranks[other_rank][rank]
            local_recv_count_pair = local_recv_counts[other_rank].cpu().item()
            send_rank_start_end = send_start_end_all_ranks[other_rank][rank]
            ref_output_tensor[
                output_indice_offset : output_indice_offset + local_recv_count_pair
            ] = input_tensors_all_ranks[other_rank][
                send_ids_all_ranks[other_rank][
                    send_rank_start_end[0] : send_rank_start_end[1]
                ]
            ]
            output_indice_offset += local_recv_count_pair
        local_recv_cumsum = paddle.cumsum(x=local_recv_counts, axis=0).to("int32")
        recv_cumsum_all_ranks.append(local_recv_cumsum)
        total_recv_count = local_recv_cumsum[-1].cpu()
        total_recv_all_ranks_cpu.append(total_recv_count)
        output_tensors_all_ranks.append(
            output_tensor[
                output_start_current_rank : output_start_current_rank + total_recv_count
            ]
        )
        output_start_current_rank += total_recv_count
        local_recv_ids = paddle.arange(dtype="int32", end=total_recv_count)
        recv_ids_all_ranks.append(local_recv_ids)
    cuda_streams_all_ranks = [paddle.device.Stream() for _ in range(world_size)]
    workspace_size = tllm_alltoall.get_moe_commworkspace_size_per_rank(world_size)
    all_workspaces = paddle.zeros(
>>>>>>        shape=[world_size, workspace_size], dtype=torch.uint64
    )
    paddle.device.synchronize()
    for rank in range(world_size):
        with paddle.device.stream_guard(stream=cuda_streams_all_ranks[rank]):
            tllm_alltoall.moe_comm(
                input_tensors_all_ranks[rank],
                send_cumsum_all_ranks[rank],
                send_ids_all_ranks[rank],
                output_tensors_all_ranks[rank],
                recv_cumsum_all_ranks[rank],
                recv_ids_all_ranks[rank],
                all_workspaces,
                rank,
                world_size,
            )
    for rank in range(world_size):
        cuda_streams_all_ranks[rank].synchronize()
    assert paddle.allclose(
        x=output_tensor, y=ref_output_tensor, atol=1e-05, rtol=1e-05
    ).item(), ""


@pytest.mark.parametrize(
    "ep_rank,ep_size,expert_count,top_k,max_token_count_per_rank,use_real_rank_token_count_cumsum",
    PREPARE_INDICES_PARAMS,
)
def test_moe_alltoall_prepare_indices(
    ep_rank,
    ep_size,
    expert_count,
    top_k,
    max_token_count_per_rank,
    use_real_rank_token_count_cumsum,
):
    """Test MOE alltoall prepare indices functionality."""
    paddle.device.set_device(device="gpu:0")

    def generate_references():
        rank_token_count = max_token_count_per_rank
        if use_real_rank_token_count_cumsum:
            rank_token_counts = [
                max(
                    1,
                    paddle.randint(
                        low=1, high=max_token_count_per_rank + 1, shape=(1,)
                    ).item(),
                )
                for _ in range(ep_size - 1)
            ]
            rank_token_counts.append(max_token_count_per_rank)
            real_rank_token_count_cumsum = (
                paddle.to_tensor(
                    data=rank_token_counts, dtype="int32", place=device2str("gpu")
                )
                .cumsum(axis=0)
                .to("int32")
            )
            rank_token_count = rank_token_counts[ep_rank]
        else:
            real_rank_token_count_cumsum = None
        target_rank_ids = paddle.randint(
            low=0, high=ep_size, shape=(rank_token_count, top_k), dtype="int32"
        )
        if not use_real_rank_token_count_cumsum:
            gathered_target_rank_ids = paddle.zeros(
                shape=[ep_size * max_token_count_per_rank, top_k], dtype="int32"
            )
            gathered_target_rank_ids[
                ep_rank * max_token_count_per_rank : ep_rank * max_token_count_per_rank
                + rank_token_count
            ] = target_rank_ids
        else:
            total_tokens = real_rank_token_count_cumsum[-1].item()
            gathered_target_rank_ids = paddle.zeros(
                shape=[total_tokens, top_k], dtype="int32"
            )
            start_pos = (
                0 if ep_rank == 0 else real_rank_token_count_cumsum[ep_rank - 1].item()
            )
            gathered_target_rank_ids[
                start_pos : start_pos + rank_token_count
            ] = target_rank_ids
        return (gathered_target_rank_ids, real_rank_token_count_cumsum, target_rank_ids)

    (
        gathered_target_rank_ids,
        real_rank_token_count_cumsum,
        target_rank_ids,
    ) = generate_references()
    (
        local_gather_indices,
        send_rank_count_cumsum,
        send_rank_local_indices,
        recv_rank_count_cumsum,
        recv_rank_local_indices,
        backward_recv_rank_local_indices,
    ) = tllm_alltoall.moe_comm_prepare_indices(
        gathered_target_rank_ids,
        real_rank_token_count_cumsum,
        max_token_count_per_rank,
        expert_count,
        top_k,
        ep_rank,
        ep_size,
    )
    assert tuple(local_gather_indices.shape)[0] <= max_token_count_per_rank * ep_size
    assert tuple(send_rank_count_cumsum.shape)[0] == ep_size
    assert tuple(recv_rank_count_cumsum.shape)[0] == ep_size
    assert tuple(send_rank_local_indices.shape)[0] <= max_token_count_per_rank * max(
        ep_size, top_k
    )
    assert tuple(recv_rank_local_indices.shape)[0] <= max_token_count_per_rank * ep_size
    assert tuple(backward_recv_rank_local_indices.shape)[
        0
    ] <= max_token_count_per_rank * max(ep_size, top_k)
    assert paddle.all(x=send_rank_count_cumsum[1:] >= send_rank_count_cumsum[:-1])
    assert paddle.all(x=recv_rank_count_cumsum[1:] >= recv_rank_count_cumsum[:-1])


@pytest.mark.parametrize(
    "ep_rank,ep_size,expert_count,top_k,max_token_count_per_rank", LOCAL_GATHER_PARAMS
)
def test_moe_local_gather(
    ep_rank, ep_size, expert_count, top_k, max_token_count_per_rank
):
    """Test MOE local gather functionality."""
    paddle.device.set_device(device="gpu:0")
    rank_token_count_cumsum = paddle.randint(
        low=0, high=max_token_count_per_rank + 1, shape=(ep_size,), dtype="int32"
    )
    rank_token_count_cumsum = paddle.cumsum(x=rank_token_count_cumsum, axis=0).to(
        "int32"
    )
    local_token_count = rank_token_count_cumsum[ep_size - 1].cpu().item()
    local_max_token_count = max_token_count_per_rank * ep_size
    local_gather_indices = paddle.randint(
        low=0,
        high=max_token_count_per_rank * ep_size,
        shape=(local_max_token_count,),
        dtype="int32",
    )
    gathered_expert_ids = paddle.randint(
        low=0,
        high=expert_count,
        shape=(max_token_count_per_rank * ep_size, top_k),
        dtype="int32",
    )
    gathered_scales = paddle.rand(
        shape=(max_token_count_per_rank * ep_size, top_k), dtype="float32"
    )
    ref_local_expert_ids = paddle.zeros(
        shape=[local_max_token_count, top_k], dtype="int32"
    )
    ref_local_scales = paddle.zeros(
        shape=[local_max_token_count, top_k], dtype="float32"
    )
    ref_local_expert_ids += expert_count
    valid_local_gather_indices = local_gather_indices[:local_token_count]
    ref_local_expert_ids[:local_token_count] = gathered_expert_ids[
        valid_local_gather_indices
    ]
    ref_local_scales[:local_token_count] = gathered_scales[valid_local_gather_indices]
    local_expert_ids = paddle.empty(shape=[local_max_token_count, top_k], dtype="int32")
    local_scales = paddle.empty(shape=[local_max_token_count, top_k], dtype="float32")
    tllm_alltoall.moe_local_gather(
        rank_token_count_cumsum,
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
    assert paddle.equal_all(x=local_expert_ids, y=ref_local_expert_ids).item()
    assert paddle.equal_all(x=local_scales, y=ref_local_scales).item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
