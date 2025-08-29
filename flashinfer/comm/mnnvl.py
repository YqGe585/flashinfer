import sys

sys.path.append("/home/flashinfer")
import ctypes
import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional

import paddle
from cuda import cuda
from paddle_utils import *

from ..cuda_utils import checkCudaErrors
from .dlpack_utils import create_dlpack_capsule, pack_strided_memory
from .mapping import Mapping

IS_BUILDING_DOCS = os.environ.get("FLASHINFER_BUILDING_DOCS") == "1"
OMPI_COMM_TYPE_HOST = 9
SIGNAL_PAD_SIZE = 2048
MNNVL_DEBUG = False


def round_up(val: int, gran: int) -> int:
    """Efficient implementation assuming gran is a power of 2"""
    return val + gran - 1 & ~(gran - 1)


def create_tensor_from_cuda_memory(
    ptr: int, shape: tuple, dtype: paddle.dtype, device_id: int
) -> paddle.Tensor:
    """
    Create a PyTorch tensor from a CUDA memory pointer using DLPack.

    Args:
        ptr: CUDA memory pointer address as integer
        shape: Desired tensor shape
        dtype: PyTorch data type
        device_id: CUDA device ID

    Returns:
        PyTorch tensor that wraps the CUDA memory
    """
    numel = 1
    for dim in shape:
        numel *= dim
    element_size = paddle.to_tensor(data=[], dtype=dtype).element_size()
    capsule_wrapper = create_dlpack_capsule(
        ptr, element_size, element_size, numel, dtype, device_id
    )
    tensor = paddle.utils.dlpack.from_dlpack(dlpack=capsule_wrapper.capsule)
    tensor._capsule_wrapper = capsule_wrapper
    return tensor.view(shape)


def test_cuda_memory_access(ptr: int, size: int, device_id: int) -> bool:
    """
    Test if CUDA memory at ptr is accessible by trying to read/write a small amount.

    Args:
        ptr: CUDA memory pointer
        size: Size of memory region
        device_id: CUDA device ID

    Returns:
        True if memory is accessible, False otherwise
    """
    try:
        test_size = min(4, size)
        host_data = bytearray(test_size)
        checkCudaErrors(cuda.cuMemcpyDtoH(host_data, ptr, test_size))
        checkCudaErrors(cuda.cuMemcpyHtoD(ptr, host_data, test_size))
        print(f"DEBUG: Memory access test PASSED for ptr=0x{ptr:x}")
        return True
    except Exception as e:
        print(f"DEBUG: Memory access test FAILED for ptr=0x{ptr:x}: {e}")
        return False


def alloc_and_copy_to_cuda(host_ptr_array: List[int]) -> Optional[int]:
    """
    A helper function that allocates memory on cuda and copies the data from the host to the device.
    """
    if not host_ptr_array:
        return None
    ArrayType = ctypes.c_uint64 * len(host_ptr_array)
    c_array = ArrayType(*host_ptr_array)
    size_in_bytes = ctypes.sizeof(c_array)
    device_ptr: cuda.CUdeviceptr = checkCudaErrors(cuda.cuMemAlloc(size_in_bytes))
    checkCudaErrors(
        cuda.cuMemcpyHtoD(device_ptr, ctypes.addressof(c_array), size_in_bytes)
    )
    return device_ptr


if IS_BUILDING_DOCS:

    class MpiComm:
        @classmethod
        def set_mpi_comm(cls, new_comm):
            pass

        def __getattr__(self, name):
            return None

    class MnnvlMemory:
        initialized: bool = False
        current_mem_offset: int = 0
        current_rank_stride: int = 0
        current_start_address: int = 0
        allocation_granularity: int = 0
        fabric_page_size: int = 1 << 29
        comm = None
        dev_id: int = None
        allocated_map: Dict[int, Any] = {}
        address_refcnt: Dict[int, Any] = {}

        def __init__(self, mapping: Mapping, size: int):
            pass

        def __del__(self):
            pass

        def as_torch_strided_tensor(self, dtype):
            return None

        @staticmethod
        def initialize():
            pass

        @staticmethod
        def get_comm(mapping: Mapping):
            return None

        @staticmethod
        def get_allocation_prop(dev_id: int):
            return None

        @staticmethod
        def get_allocation_granularity(dev_id: int):
            return None

        @staticmethod
        def new_mnnvl_memory_address(mapping: Mapping, size: int):
            pass

        @staticmethod
        def open_mnnvl_memory(mapping: Mapping, size: int):
            return None

        @staticmethod
        def close_mnnvl_memory(ptr: int):
            pass

        @staticmethod
        def support_nvlink(need_all_up: bool = True):
            return None

        @staticmethod
        def supports_mnnvl() -> bool:
            return False

else:
    import pynvml
    from mpi4py import MPI

    class MpiComm:
        _comm: MPI.Intracomm = MPI.COMM_WORLD

        @classmethod
        def set_mpi_comm(cls, new_comm: MPI.Intracomm):
            cls._comm = new_comm

        def __getattr__(self, name):
            return getattr(self._comm, name)

    class MnnvlMemory:
        initialized: bool = False
        current_mem_offset: int = 0
        current_rank_stride: int = 0
        current_start_address: int = 0
        allocation_granularity: int = 0
        fabric_page_size: int = 1 << 29
        comm = None
        dev_id: int = None
        allocated_map: Dict[int, Any] = {}
        address_refcnt: Dict[int, Any] = {}

        def __init__(self, mapping: Mapping, size: int):
            self.mapping = mapping
            self.segment_size = size
            self.ptr, self.rank_stride = MnnvlMemory.open_mnnvl_memory(
                self.mapping, size
            )

        def __del__(self):
            if not sys.is_finalizing():
                MnnvlMemory.close_mnnvl_memory(self.ptr)

        def as_torch_strided_tensor(self, dtype):
            num_segments = MnnvlMemory.comm.Get_size()
            return pack_strided_memory(
                self.ptr,
                self.segment_size,
                self.rank_stride,
                num_segments,
                dtype,
                MnnvlMemory.dev_id,
            )

        @staticmethod
        def initialize():
            if not MnnvlMemory.initialized:
                _ = paddle.empty(shape=[1])
                try:
                    pynvml.nvmlDeviceGetCount()
                except pynvml.NVMLError_Uninitialized:
                    pynvml.nvmlInit()
                MnnvlMemory.initialized = True

        @staticmethod
        def get_comm(mapping: Mapping):
            if MnnvlMemory.comm is not None:
                return MnnvlMemory.comm
            comm = MpiComm().Split(
                mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
            )
            MnnvlMemory.comm = comm
            return comm

        @staticmethod
        def get_allocation_prop(dev_id: int):
            location = cuda.CUmemLocation()
            location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            location.id = dev_id
            allocation_prop = cuda.CUmemAllocationProp()
            allocation_prop.type = (
                cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            )
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
            allocation_prop.location = location
            return allocation_prop

        @staticmethod
        def get_allocation_granularity(dev_id: int):
            if MnnvlMemory.allocation_granularity != 0:
                return MnnvlMemory.allocation_granularity
            allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
            option = cuda.CUmemAllocationGranularity_flags(
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
            )
            granularity = checkCudaErrors(
                cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option)
            )
            MnnvlMemory.allocation_granularity = granularity
            return MnnvlMemory.allocation_granularity

        @staticmethod
        def new_mnnvl_memory_address(mapping: Mapping, size: int):
            page_count = (
                size + MnnvlMemory.fabric_page_size - 1
            ) // MnnvlMemory.fabric_page_size
            current_rank_stride = page_count * MnnvlMemory.fabric_page_size
            logging.info(
                f"[MnnvlMemory] creating address with stride={current_rank_stride}"
            )
            comm = MnnvlMemory.get_comm(mapping)
            comm_size = comm.Get_size()
            address_size = current_rank_stride * comm_size
            ptr = checkCudaErrors(
                cuda.cuMemAddressReserve(
                    address_size, MnnvlMemory.fabric_page_size, 0, 0
                )
            )
            MnnvlMemory.current_start_address = int(ptr)
            MnnvlMemory.current_rank_stride = current_rank_stride
            MnnvlMemory.current_mem_offset = 0

        @staticmethod
        def open_mnnvl_memory(mapping: Mapping, size: int):
            dev = checkCudaErrors(cuda.cuCtxGetDevice())
            dev_id = int(dev)
            if MnnvlMemory.dev_id is None:
                MnnvlMemory.dev_id = dev_id
            assert (
                dev_id == MnnvlMemory.dev_id
            ), f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
            comm = MnnvlMemory.get_comm(mapping)
            comm_rank = comm.Get_rank()
            comm_size = comm.Get_size()
            all_rank_allocate_sizes = comm.allgather(size)
            assert len(all_rank_allocate_sizes) == comm_size
            assert all(
                x == size for x in all_rank_allocate_sizes
            ), "Not all rank allocating same size."
            granularity = MnnvlMemory.get_allocation_granularity(dev_id)
            aligned_size = (size + granularity - 1) // granularity * granularity
            if (
                MnnvlMemory.current_mem_offset + aligned_size
                > MnnvlMemory.current_rank_stride
            ):
                MnnvlMemory.new_mnnvl_memory_address(mapping, aligned_size)
            assert (
                MnnvlMemory.current_mem_offset + aligned_size
                <= MnnvlMemory.current_rank_stride
            )
            allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
            allocated_mem_handle = checkCudaErrors(
                cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
            )
            exported_fabric_handle = checkCudaErrors(
                cuda.cuMemExportToShareableHandle(
                    allocated_mem_handle,
                    cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    0,
                )
            )
            all_handles_data = comm.allgather(exported_fabric_handle.data)
            madesc = cuda.CUmemAccessDesc()
            madesc.location = allocation_prop.location
            madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            mem_handles = [None] * comm_size
            for i, remote_handle_data in enumerate(all_handles_data):
                rank_ptr = (
                    MnnvlMemory.current_start_address
                    + MnnvlMemory.current_rank_stride * i
                    + MnnvlMemory.current_mem_offset
                )
                if i == comm_rank:
                    mem_handles[i] = allocated_mem_handle
                    checkCudaErrors(
                        cuda.cuMemMap(
                            rank_ptr, aligned_size, 0, allocated_mem_handle, 0
                        )
                    )
                else:
                    imported_mem_handle = checkCudaErrors(
                        cuda.cuMemImportFromShareableHandle(
                            remote_handle_data,
                            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                        )
                    )
                    mem_handles[i] = imported_mem_handle
                    checkCudaErrors(
                        cuda.cuMemMap(rank_ptr, aligned_size, 0, imported_mem_handle, 0)
                    )
                checkCudaErrors(
                    cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1)
                )
            ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
            stride = MnnvlMemory.current_rank_stride
            MnnvlMemory.allocated_map[ptr] = (
                mapping,
                aligned_size,
                mem_handles,
                MnnvlMemory.current_start_address,
                MnnvlMemory.current_rank_stride,
                MnnvlMemory.current_mem_offset,
            )
            MnnvlMemory.address_refcnt[MnnvlMemory.current_start_address] = (
                MnnvlMemory.address_refcnt.get(MnnvlMemory.current_start_address, 0) + 1
            )
            MnnvlMemory.current_mem_offset += aligned_size
            return ptr, stride

        @staticmethod
        def close_mnnvl_memory(ptr: int):
            (
                mapping,
                aligned_size,
                mem_handles,
                start_address,
                rank_stride,
                address_offset,
            ) = MnnvlMemory.allocated_map.pop(ptr)
            comm = MnnvlMemory.get_comm(mapping)
            comm_size = comm.Get_size()
            for i in range(comm_size):
                rank_ptr = start_address + i * rank_stride + address_offset
                checkCudaErrors(cuda.cuMemUnmap(rank_ptr, aligned_size))
                checkCudaErrors(cuda.cuMemRelease(mem_handles[i]))
            MnnvlMemory.address_refcnt[start_address] -= 1
            if MnnvlMemory.address_refcnt[start_address] == 0:
                MnnvlMemory.address_refcnt.pop(start_address)
                device_ptr = cuda.CUdeviceptr(start_address)
                checkCudaErrors(
                    cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride)
                )
                if start_address == MnnvlMemory.current_start_address:
                    MnnvlMemory.current_start_address = 0
                    MnnvlMemory.current_rank_stride = 0
                    MnnvlMemory.current_mem_offset = 0

        @staticmethod
        def support_nvlink(need_all_up: bool = True):
>>>>>>            dev_id = torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            link_count = pynvml.NVML_NVLINK_MAX_LINKS
            active_links = 0
            available_links = 0
            for link_idx in range(link_count):
                try:
                    if pynvml.nvmlDeviceGetNvLinkCapability(
                        handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                    ):
                        available_links += 1
                        is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                        if is_active:
                            active_links += 1
                except pynvml.NVMLError_NotSupported:
                    continue
            return (
                active_links == available_links and available_links > 0
                if need_all_up
                else available_links > 0
            )

        @staticmethod
        def supports_mnnvl() -> bool:
            arch = platform.machine().lower()
            if "aarch64" not in arch:
                return False
            return MnnvlMemory.support_nvlink(True)


class McastDeviceMemory:
    """Python port of McastDeviceMemory from TensorRT-LLM"""

    def __init__(
        self,
        buf_size: int,
        group_size: int,
        group_rank: int,
        device_idx: int,
        is_multi_node: bool = True,
    ):
        cu_device = checkCudaErrors(cuda.cuDeviceGet(device_idx))
        primary_ctx = checkCudaErrors(cuda.cuDevicePrimaryCtxRetain(cu_device))
        checkCudaErrors(cuda.cuCtxSetCurrent(primary_ctx))
        from flashinfer.utils import has_cuda_cudart

        if has_cuda_cudart():
            import cuda.cudart as cudart
        else:
            import cuda.bindings.runtime as cudart
        checkCudaErrors(cudart.cudaSetDevice(device_idx))
        self.is_multi_node = is_multi_node
        self.device_idx = device_idx
        self.group_size = group_size
        self.group_rank = group_rank
        self.buf_size = buf_size
        self.signal_pad_offset = 0
        self.allocation_size = 0
        self.mc_ptr = 0
        self.uc_ptrs: List[int] = []
        self.signal_pads: List[int] = []
        self.signal_pads_dev = 0
        self.uc_ptrs_dev = 0
        self.mc_handle = 0
        self.uc_handles: List[int] = []
        self.SIGNAL_PAD_ALIGNMENT = 16
        self.SIGNAL_PAD_SIZE = SIGNAL_PAD_SIZE
        multicast_supported = checkCudaErrors(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                device_idx,
            )
        )
        if multicast_supported == 0:
            raise RuntimeError(
                "[McastDeviceMemory] Device does not support multicasting."
            )
        self.signal_pad_offset = round_up(buf_size, self.SIGNAL_PAD_ALIGNMENT)
        logging.info(
            f"[McastDeviceMemory] Rank: {group_rank}, Group size: {group_size}, mnNvlink: {is_multi_node}, device_idx: {device_idx}, Signal pad offset: {self.signal_pad_offset}"
        )
        if self.is_multi_node:
            fabric_handle_supported = checkCudaErrors(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                    device_idx,
                )
            )
            if fabric_handle_supported == 0:
                raise RuntimeError(
                    "[McastDeviceMemory] Device does not support fabric handle."
                )
            self._alloc_mn_mcast_mem(buf_size)
        else:
            raise NotImplementedError("Single-node NVLS allocation not implemented yet")
        self.signal_pads = [0] * self.group_size
        for i in range(self.group_size):
            self.signal_pads[i] = self.uc_ptrs[i] + self.signal_pad_offset
            if i == self.group_rank:
                checkCudaErrors(
                    cuda.cuMemsetD8(self.signal_pads[i], 0, self.SIGNAL_PAD_SIZE)
                )
        self.signal_pads_dev = alloc_and_copy_to_cuda(self.signal_pads)
        self.uc_ptrs_dev = alloc_and_copy_to_cuda(self.uc_ptrs)

    def __del__(self):
        """Destructor - cleanup allocated memory"""
        if not hasattr(self, "is_multi_node"):
            return
        if not self.is_multi_node:
            return
        if sys.is_finalizing():
            return
        try:
            cuda.cuCtxGetCurrent()
        except Exception as e:
            print(f"Destructor: CUDA context invalid, skipping cleanup: {e}")
            return
        if self.signal_pads_dev:
            checkCudaErrors(cuda.cuMemFree(self.signal_pads_dev))
        if self.uc_ptrs_dev:
            checkCudaErrors(cuda.cuMemFree(self.uc_ptrs_dev))
        if hasattr(self, "uc_handles") and self.uc_handles:
            for rank in range(self.group_size):
                if self.uc_handles[rank] != 0:
                    try:
                        checkCudaErrors(cuda.cuMemRelease(self.uc_handles[rank]))
                        if rank < len(self.uc_ptrs) and self.uc_ptrs[rank]:
                            checkCudaErrors(
                                cuda.cuMemUnmap(
                                    self.uc_ptrs[rank], self.allocation_size
                                )
                            )
                    except Exception as e:
                        print(
                            f"Destructor: Failed to release UC handle for rank {rank}: {e}"
                        )
            if hasattr(self, "uc_base_ptr") and self.uc_base_ptr:
                checkCudaErrors(
                    cuda.cuMemAddressFree(self.uc_base_ptr, self.total_uc_size)
                )
        if hasattr(self, "mc_handle") and self.mc_handle and self.mc_handle != 0:
            try:
                checkCudaErrors(cuda.cuMemUnmap(self.mc_ptr, self.allocation_size))
                checkCudaErrors(
                    cuda.cuMemAddressFree(self.mc_ptr, self.allocation_size)
                )
                checkCudaErrors(cuda.cuMemRelease(self.mc_handle))
            except Exception as e:
                print(f"Destructor: Failed to release MC handle: {e}")

    def get_signal_pad_ptrs_host(self) -> List[int]:
        """Get the raw array of signal pad pointers to all ranks (including self)"""
        return self.signal_pads

    def get_buffer_ptrs_host(self) -> List[int]:
        """Get the raw array of unicast pointers to all ranks (including self)"""
        return self.uc_ptrs

    def get_signal_pad_ptrs_dev(self) -> int:
        """Get the raw array of signal pad pointers to all ranks (including self)"""
        return self.signal_pads_dev

    def get_buffer_ptrs_dev(self) -> int:
        """Get the raw array of unicast pointers to all ranks (including self)"""
        return self.uc_ptrs_dev

    def get_unicast_ptr(self, rank: int) -> int:
        """Get the raw unicast pointer to a given rank"""
        if rank >= len(self.uc_ptrs):
            raise ValueError(f"Rank {rank} out of range (0-{len(self.uc_ptrs) - 1})")
        data_ptr = self.uc_ptrs[rank]
        return data_ptr

    def get_multicast_ptr(self) -> int:
        """Get the raw multicast pointer"""
        return int(self.mc_ptr)

    def get_rank(self) -> int:
        """Get the rank of this device in the group"""
        return self.group_rank

    def get_world_size(self) -> int:
        """Get the total number of devices in the group"""
        return self.group_size

    def _alloc_mn_mcast_mem(self, buf_size: int):
        """Allocate multi-node multicast memory using MNNVL"""
        try:
            current_device = checkCudaErrors(cuda.cuCtxGetDevice())
            if int(current_device) != self.device_idx:
                print(
                    f"CUDA context device mismatch! Current: {current_device}, Expected: {self.device_idx}"
                )
        except Exception as e:
            print(f"Error checking CUDA context: {e}")
        comm = MpiComm()
        handle_type = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.requestedHandleTypes = handle_type
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        allocation_prop.location = cuda.CUmemLocation()
        allocation_prop.location.type = (
            cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        )
        allocation_prop.location.id = self.device_idx
        allocation_prop.allocFlags.gpuDirectRDMACapable = 1
        alloc_granularity = checkCudaErrors(
            cuda.cuMemGetAllocationGranularity(
                allocation_prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        self.allocation_size = round_up(
            buf_size + self.SIGNAL_PAD_SIZE, alloc_granularity
        )
        mc_prop = cuda.CUmulticastObjectProp()
        mc_prop.numDevices = self.group_size
        mc_prop.size = self.allocation_size
        mc_prop.handleTypes = handle_type
        mc_granularity = checkCudaErrors(
            cuda.cuMulticastGetGranularity(
                mc_prop,
                cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
            )
        )
        self.allocation_size = round_up(self.allocation_size, mc_granularity)
        self.uc_handles = [0] * self.group_size
        self.uc_handles[self.group_rank] = checkCudaErrors(
            cuda.cuMemCreate(self.allocation_size, allocation_prop, 0)
        )
        my_fabric_handle = checkCudaErrors(
            cuda.cuMemExportToShareableHandle(
                self.uc_handles[self.group_rank],
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                0,
            )
        )
        all_fabric_handles = comm.allgather(my_fabric_handle.data)
        cuda.cuCtxSynchronize()
        for p in range(self.group_size):
            if p != self.group_rank:
                self.uc_handles[p] = checkCudaErrors(
                    cuda.cuMemImportFromShareableHandle(
                        all_fabric_handles[p],
                        cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    )
                )
        if self.group_rank == 0:
            self.mc_handle = checkCudaErrors(cuda.cuMulticastCreate(mc_prop))
            mc_fabric_handle = checkCudaErrors(
                cuda.cuMemExportToShareableHandle(
                    self.mc_handle,
                    cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    0,
                )
            )
        else:
            mc_fabric_handle = None
        mc_fabric_handle_data = comm.bcast(
            mc_fabric_handle.data if mc_fabric_handle else None, root=0
        )
        cuda.cuCtxSynchronize()
        if self.group_rank != 0:
            self.mc_handle = checkCudaErrors(
                cuda.cuMemImportFromShareableHandle(
                    mc_fabric_handle_data,
                    cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                )
            )
        checkCudaErrors(cuda.cuMulticastAddDevice(self.mc_handle, self.device_idx))
        self.uc_ptrs = [0] * self.group_size
        total_uc_size = self.allocation_size * self.group_size
        self.total_uc_size = total_uc_size
        uc_base_ptr = checkCudaErrors(
            cuda.cuMemAddressReserve(total_uc_size, mc_granularity, 0, 0)
        )
        self.uc_base_ptr = uc_base_ptr
        access_desc = cuda.CUmemAccessDesc()
        access_desc.location = cuda.CUmemLocation()
        access_desc.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = self.device_idx
        access_desc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        for i in range(self.group_size):
            offset = self.allocation_size * i
            self.uc_ptrs[i] = int(uc_base_ptr) + offset
            checkCudaErrors(
                cuda.cuMemMap(
                    self.uc_ptrs[i], self.allocation_size, 0, self.uc_handles[i], 0
                )
            )
        checkCudaErrors(
            cuda.cuMemSetAccess(uc_base_ptr, total_uc_size, [access_desc], 1)
        )
        self.mc_ptr = checkCudaErrors(
            cuda.cuMemAddressReserve(self.allocation_size, mc_granularity, 0, 0)
        )
        checkCudaErrors(
            cuda.cuMemMap(self.mc_ptr, self.allocation_size, 0, self.mc_handle, 0)
        )
        checkCudaErrors(
            cuda.cuMemSetAccess(self.mc_ptr, self.allocation_size, [access_desc], 1)
        )
        checkCudaErrors(
            cuda.cuMulticastBindMem(
                self.mc_handle,
                0,
                self.uc_handles[self.group_rank],
                0,
                self.allocation_size,
                0,
            )
        )

    def lamport_initialize(self, rank: int, dtype: paddle.dtype):
        if dtype == "bfloat16" or dtype == "float16":
            neg_zero = 32768
            dsize = 2
            memset_func = cuda.cuMemsetD16
        elif dtype == "float32":
            neg_zero = 2147483648
            dsize = 4
            memset_func = cuda.cuMemsetD32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        num_elements = self.allocation_size // dsize
        checkCudaErrors(
            memset_func(int(self.uc_ptrs[self.group_rank]), neg_zero, num_elements)
        )


class McastGPUBuffer:
    """
    Wrapper class for McastDeviceMemory to facilitate PyTorch tensor creation.
    It manages a buffer accessible via unicast or multicast for multi-node communication.

    Python port of McastGPUBuffer from TensorRT-LLM
    """

    def __init__(
        self,
        buf_size: int,
        group_size: int,
        group_rank: int,
        device: str,
        mn_nvlink: bool = True,
    ):
        """
        Constructor for McastGpuBuffer.

        Args:
            buf_size: The total size of the buffer in bytes
            group_size: The number of ranks in the communication group
            group_rank: The rank of the local process within the group
            device: The CUDA device for buffer allocation
            mn_nvlink: Flag indicating if multi-node NVLink is used
        """
        self.mcast_device_memory = McastDeviceMemory(
            buf_size, group_size, group_rank, device.index, mn_nvlink
        )
        self.buf_size = buf_size
        self.local_device = device

    def lamport_initialize(self, rank: int, dtype: paddle.dtype):
        self.mcast_device_memory.lamport_initialize(rank, dtype)

    def get_mc_buffer(
        self, sizes: tuple, dtype: paddle.dtype, storage_offset: int = 0
    ) -> paddle.Tensor:
        """
        Returns a PyTorch tensor view of the multicast buffer portion.

        Args:
            sizes: The desired shape (dimensions) of the tensor
            dtype: The data type of the tensor elements
            storage_offset: The offset in elements from the start of the buffer

        Returns:
            A PyTorch tensor wrapping the multicast buffer section
        """
        raise NotImplementedError("Not implemented yet")

    def get_multicast_ptr(self) -> int:
        """Get the raw multicast pointer"""
        return self.mcast_device_memory.get_multicast_ptr()

    def get_buffer_ptrs_dev(self) -> int:
        """Get the buffer pointers device array"""
        return self.mcast_device_memory.get_buffer_ptrs_dev()
