import ctypes
import functools
import os
import shlex
from typing import Sequence

import paddle

from ..jit import JitSpec
from ..jit import env as jit_env
from ..jit import gen_jit_spec


def gen_nvshmem_module() -> JitSpec:
    lib_dirs = jit_env.get_nvshmem_lib_dirs()
    ldflags = (
        [f"-L{lib_dir}" for lib_dir in lib_dirs]
        + ["-lnvshmem_device"]
        + shlex.split(os.environ.get("NVSHMEM_LDFLAGS", ""))
    )
    return gen_jit_spec(
        "nvshmem",
        [jit_env.FLASHINFER_CSRC_DIR / "nvshmem_binding.cu"],
        extra_include_paths=[str(p) for p in jit_env.get_nvshmem_include_dirs()],
        extra_ldflags=ldflags,
        needs_device_linking=True,
    )


@functools.cache
def get_nvshmem_module():
    lib_dirs = jit_env.get_nvshmem_lib_dirs()
    lib_path = None
    lib_names = ["libnvshmem_host.so", "libnvshmem_host.so.3"]
    for lib_dir in lib_dirs:
        for lib_name in lib_names:
            candidate_path = lib_dir / lib_name
            if candidate_path.exists():
                lib_path = candidate_path
                break
        if lib_path is not None:
            break
    if lib_path is None:
        raise FileNotFoundError(
            f"Could not find libnvshmem_host.so or libnvshmem_host.so.3 in {lib_dirs}"
        )
    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    module = gen_nvshmem_module().build_and_load()
    return module


def get_unique_id() -> paddle.Tensor:
    return get_nvshmem_module().nvshmem_get_unique_id()


def unique_id_size() -> int:
    return get_nvshmem_module().nvshmem_unique_id_size()


def alloc_empty_unique_id() -> paddle.Tensor:
    return paddle.zeros(shape=unique_id_size(), dtype="uint8")


def init(uid: paddle.Tensor, rank: int, world_size: int) -> int:
    status = get_nvshmem_module().nvshmem_init(uid, rank, world_size)
    paddle.device.synchronize()
    return status


def alltoall(dest: paddle.Tensor, source: paddle.Tensor) -> None:
    return get_nvshmem_module().nvshmem_alltoall(dest, source)


def finalize() -> None:
    paddle.device.synchronize()
    get_nvshmem_module().nvshmem_finalize()


def my_pe() -> int:
    return get_nvshmem_module().nvshmem_my_pe()


def n_pes() -> int:
    return get_nvshmem_module().nvshmem_n_pes()


def malloc(shape: Sequence[int], dtype: paddle.dtype, device: str) -> paddle.Tensor:
    """Allocates memory using NVSHMEM collective malloc operation.

    This is a collective operation that requires participation by all PEs (Processing Elements).
    All participants must call this function with the same parameters.

    Note: This tensor should be explicitly deleted (del tensor) to ensure proper ordering
    of nvshmem_free operations rather than relying on garbage collection.

    Args:
        shape: The shape of the tensor to allocate.
        dtype: The data type of the tensor.
        device: The device to allocate the tensor on.

    Returns:
        A tensor allocated using NVSHMEM collective malloc.

    Reference:
        https://docs.nvidia.com/nvshmem/api/gen/api/memory.html#nvshmem-malloc-nvshmem-free-nvshmem-align
    """
    return get_nvshmem_module().nvshmem_malloc(shape, dtype, device)


def barrier_all() -> None:
    get_nvshmem_module().nvshmem_barrier_all()


def barrier_all_on_current_stream() -> None:
    get_nvshmem_module().nvshmem_barrier_all_on_current_stream()
