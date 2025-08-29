import sys


import socket

import paddle
import pynvml
import pytest
from flashinfer.paddle_utils import *

from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory, MpiComm
from flashinfer.comm.trtllm_alltoall import MnnvlMoe, MoEAlltoallInfo

pynvml.nvmlInit()


@pytest.mark.skipif(
    not MnnvlMemory.supports_mnnvl(),
    reason="Mnnvl memory is not supported on this platform",
)
class TestMnnvlMemory:
    @pytest.fixture(autouse=True)
    def setup(self):
        hostname = socket.gethostname()
        self.comm = MpiComm()
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        all_hostnames = self.comm.allgather(hostname)
        local_ntasks_per_node = all_hostnames.count(hostname)
        all_ntasks_per_node = self.comm.allgather(local_ntasks_per_node)
        uniform_ntasks = all(x == all_ntasks_per_node[0] for x in all_ntasks_per_node)
        assert uniform_ntasks, "Not all nodes has same ntasks_per_node"
        self.local_world_size = local_ntasks_per_node
        self.local_rank = self.rank % self.local_world_size
        local_dev_count = paddle.device.cuda.device_count()
        assert (
            self.local_world_size <= local_dev_count
        ), "ntasks_per_node should be less than local device count"
        paddle.device.set_device(device=device2str(self.local_rank))
        MnnvlMemory.initialize()
        self.mapping = Mapping(
            self.world_size, self.rank, self.local_world_size, tp_size=self.world_size
        )

    @staticmethod
    def align_memory(size: int):
        align_size = 2 * 1024 * 1024
        return (size + align_size - 1) // align_size * align_size

    @pytest.mark.skipif(
        not MnnvlMemory.supports_mnnvl(),
        reason="Mnnvl memory is not supported on this platform",
    )
    def test_mnnvl_memory(self):
        allocate0_size = 4 * 1024 * 1024 - 3 * 1024
        mnnvl_memory0 = MnnvlMemory(self.mapping, allocate0_size)
        allocate0_size_aligned = TestMnnvlMemory.align_memory(allocate0_size)
        assert MnnvlMemory.current_mem_offset == allocate0_size_aligned
        tensor0 = mnnvl_memory0.as_torch_strided_tensor("int32")
        numel_per_rank = allocate0_size // 4
        tensor0[(self.rank + 1) % self.world_size] = paddle.arange(
            start=self.rank, end=self.rank + numel_per_rank
        )
        self.comm.Barrier()
        for r in range(self.world_size):
            paddle.equal_all(
                x=tensor0[(r + 1) % self.world_size],
                y=paddle.arange(start=r, end=r + numel_per_rank),
            ).item()
        allocate1_size = 30 * 1024 * 1024 - 2 * 1024
        mnnvl_memory1 = MnnvlMemory(self.mapping, allocate1_size)
        allocate1_size_aligned = TestMnnvlMemory.align_memory(allocate1_size)
        assert (
            MnnvlMemory.current_mem_offset
            == allocate0_size_aligned + allocate1_size_aligned
        )
        tensor1 = mnnvl_memory1.as_torch_strided_tensor("float32")
        numel_per_rank = allocate1_size // 4
        tensor1[(self.rank + 5) % self.world_size] = paddle.arange(
            start=self.rank, end=self.rank + numel_per_rank, dtype="float32"
        )
        self.comm.Barrier()
        for r in range(self.world_size):
            paddle.equal_all(
                x=tensor1[(r + 5) % self.world_size],
                y=paddle.arange(start=r, end=r + numel_per_rank, dtype="float32"),
            ).item()
        self.comm.Barrier()
        del tensor0, mnnvl_memory0
        self.comm.Barrier()
        large_allocation2_size = 768 * 1024 * 1024
        large_mnnvl_memory2 = MnnvlMemory(self.mapping, large_allocation2_size)
        allocate2_size_aligned = TestMnnvlMemory.align_memory(large_allocation2_size)
        assert MnnvlMemory.current_mem_offset == allocate2_size_aligned
        assert large_mnnvl_memory2.rank_stride == 1 << 30
        del tensor1

    @pytest.mark.skipif(
        not MnnvlMemory.supports_mnnvl(),
        reason="Mnnvl memory is not supported on this platform",
    )
    def test_moe_alltoall_multi_rank_single_gpu(self):
        paddle.device.set_device(device=device2str(self.rank))
        max_world_size = 8
        assert (
            self.world_size <= max_world_size
        ), f"should run with world_size at most {max_world_size}"
        paddle.seed(seed=self.world_size)
        input_entry_per_rank, vector_dim, dtype = 128, 256, "float16"
        input_tensor = paddle.randn(
            shape=[input_entry_per_rank * self.world_size, vector_dim], dtype=dtype
        )
        ref_output_tensor = paddle.zeros(
            shape=[input_entry_per_rank * self.world_size, vector_dim], dtype=dtype
        )
        target_rank_ids = paddle.randint(
            low=0,
            high=self.world_size,
            shape=(input_entry_per_rank * self.world_size,),
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
        for rank in range(self.world_size):
            send_start_end = []
            local_target_rank_ids = target_rank_ids_all_ranks[rank]
            sorted_local_target_rank_ids, local_send_id = paddle.sort(
                x=local_target_rank_ids
            ), paddle.argsort(x=local_target_rank_ids)
            local_send_id = local_send_id.to("int32")
            padded_sorted_local_target_rank_ids = paddle.concat(
                x=(
                    sorted_local_target_rank_ids,
                    paddle.arange(dtype="int32", end=self.world_size),
                )
            )
            unique_target_rank_ids, local_send_counts = paddle.unique(
                x=padded_sorted_local_target_rank_ids, return_counts=True
            )
            local_send_counts = local_send_counts.to("int32")
            assert (
                unique_target_rank_ids.size == self.world_size
            ), "unique_target_rank_ids must be equal to world_size"
            local_send_counts -= 1
            local_send_cumsum = paddle.cumsum(x=local_send_counts, axis=0).to("int32")
            send_ids_all_ranks.append(local_send_id)
            send_counts_all_ranks.append(local_send_counts)
            send_cumsum_all_ranks.append(local_send_cumsum)
            local_send_cumsum_cpu = local_send_cumsum.cpu().tolist()
            for i in range(len(local_send_cumsum_cpu)):
                send_start_end.append(
                    (
                        local_send_cumsum_cpu[i - 1] if i > 0 else 0,
                        local_send_cumsum_cpu[i],
                    )
                )
            send_start_end_all_ranks.append(send_start_end)
        recv_ids_all_ranks = []
        recv_cumsum_all_ranks = []
        ref_output_tensors_all_ranks = []
        total_recv_all_ranks_cpu = []
        output_indice_offset = 0
        output_start_current_rank = 0
        for rank in range(self.world_size):
            local_recv_counts = paddle.zeros(shape=self.world_size, dtype="int32")
            for other_rank in range(self.world_size):
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
            ref_output_tensors_all_ranks.append(
                ref_output_tensor[
                    output_start_current_rank : output_start_current_rank
                    + total_recv_count
                ]
            )
            output_start_current_rank += total_recv_count
            local_recv_ids = paddle.arange(dtype="int32", end=total_recv_count)
            recv_ids_all_ranks.append(local_recv_ids)
        alltoall_info = MoEAlltoallInfo(
            None,
            send_cumsum_all_ranks[self.rank],
            send_ids_all_ranks[self.rank],
            recv_cumsum_all_ranks[self.rank],
            recv_ids_all_ranks[self.rank],
            None,
            tuple(ref_output_tensors_all_ranks[self.rank].shape)[0],
        )
        alltoall_workspace = MnnvlMoe.get_moe_workspaces(self.mapping)
        self.comm.Barrier()
        output = MnnvlMoe.mnnvl_moe_alltoallv(
            input_tensors_all_ranks[self.rank],
            alltoall_info,
            alltoall_workspace,
            self.rank,
            self.world_size,
        )
        self.comm.Barrier()
        assert paddle.allclose(
            x=output, y=ref_output_tensors_all_ranks[self.rank], atol=1e-05, rtol=1e-05
        ).item(), ""
