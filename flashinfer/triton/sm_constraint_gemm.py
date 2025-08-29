from typing import Optional

import paddle
import triton

from .kernels.sm_constraint_gemm import (gemm_kernel,
                                         gemm_kernel_descriptor_persistent,
                                         gemm_kernel_persistent)
from .utils import check_device, check_dim, check_input


def gemm_persistent(a, b, c=None, alpha=1.0, beta=0.0, out_dtype=None, num_sms=None):
    """
    GEMM operation with SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
    """
    check_input(a)
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)
    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)
    assert (
        tuple(a.shape)[1] == tuple(b.shape)[0]
    ), "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"
    if c is not None:
        assert (
            tuple(a.shape)[0] == tuple(c.shape)[0]
        ), "Incompatible dimensions between a and c"
        assert (
            tuple(b.shape)[1] == tuple(c.shape)[1]
        ), "Incompatible dimensions between b and c"
    M, K = tuple(a.shape)
    K, N = tuple(b.shape)
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype
        if dtype != paddle.float8_e4m3fn
        else "bfloat16"
    )
    assert (
        c is None or c.dtype == out_dtype
    ), "Incompatible dtypes between c and out_dtype"
    c = paddle.empty(shape=(M, N), dtype=out_dtype) if c is None else c
    NUM_SMS = paddle.device.cuda.get_device_properties(
        device="gpu"
    ).multi_processor_count
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)
    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    gemm_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.get_strides()[0],
        a.get_strides()[1],
        b.get_strides()[0],
        b.get_strides()[1],
        c.get_strides()[0],
        c.get_strides()[1],
        alpha=alpha,
        beta=beta,
        NUM_SMS=num_sms,
    )
    return c


def gemm(a, b, c=None, alpha=1.0, beta=0.0, out_dtype=None):
    """
    GEMM operation without SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
    """
    check_input(a)
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)
    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)
    assert (
        tuple(a.shape)[1] == tuple(b.shape)[0]
    ), "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"
    if c is not None:
        assert (
            tuple(a.shape)[0] == tuple(c.shape)[0]
        ), "Incompatible dimensions between a and c"
        assert (
            tuple(b.shape)[1] == tuple(c.shape)[1]
        ), "Incompatible dimensions between b and c"
    M, K = tuple(a.shape)
    K, N = tuple(b.shape)
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype
        if dtype != paddle.float8_e4m3fn
        else "bfloat16"
    )
    assert (
        c is None or c.dtype == out_dtype
    ), "Incompatible dtypes between c and out_dtype"
    c = paddle.empty(shape=(M, N), dtype=out_dtype) if c is None else c
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.get_strides()[0],
        a.get_strides()[1],
        b.get_strides()[0],
        b.get_strides()[1],
        c.get_strides()[0],
        c.get_strides()[1],
        alpha=alpha,
        beta=beta,
    )
    return c


def gemm_descriptor_persistent(
    a,
    b,
    c=None,
    alpha=1.0,
    beta=0.0,
    out_dtype=None,
    num_sms=None,
    EPILOGUE_SUBTILE=False,
):
    """
    GEMM operation with SM constraint by Triton.
    Requires TMA support and descriptor creation.
    C = alpha * (a @ b.T) + beta * C

    Note:
        - K and N must be greater than 16B.
        - Support float16, float8_e4m3fn, bfloat16.
        - float32 is not supported due to performance issues.

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (N, K)
        c: The output matrix. Shape: (M, N). In-place epilogue is supported. Expected to be out_dtype (if not specified, same as a.dtype, but fp8 --> bf16).
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        out_dtype: The dtype of the output matrix. Default: fp8 --> bf16. Otherwise, same as a.dtype.
        num_sms: The number of SMs to use for the computation.
        EPILOGUE_SUBTILE: Whether to use the epilogue subtile optimization.
    """
    check_input(a)
    check_input(b)
    check_device([a, b])
    check_dim(2, a)
    check_dim(2, b)
    if c is not None:
        check_input(c)
        check_device([c])
        check_dim(2, c)
    assert (
        tuple(a.shape)[1] == tuple(b.shape)[1]
    ), "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"
    if c is not None:
        assert (
            tuple(a.shape)[0] == tuple(c.shape)[0]
        ), "Incompatible dimensions between a and c"
        assert (
            tuple(b.shape)[0] == tuple(c.shape)[1]
        ), "Incompatible dimensions between b and c"
    M, K = tuple(a.shape)
    N, K = tuple(b.shape)
    dtype = a.dtype
    out_dtype = (
        out_dtype
        if out_dtype
        else dtype
        if dtype != paddle.float8_e4m3fn
        else "bfloat16"
    )
    if dtype == paddle.float8_e4m3fn:
        assert K >= 16, "Least chunk size must be 16B"
        assert N >= 16, "Least chunk size must be 16B"
    else:
        assert K >= 8, "Least chunk size must be 16B"
        assert N >= 8, "Least chunk size must be 16B"
    c = paddle.empty(shape=(M, N), dtype=out_dtype) if c is None else c
    NUM_SMS = paddle.device.cuda.get_device_properties(
        device="gpu"
    ).multi_processor_count
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return paddle.empty(shape=size, dtype="int8")

    triton.set_allocator(alloc_fn)
    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    gemm_kernel_descriptor_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        alpha,
        beta,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128 if dtype != "float32" else 64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_stages=3,
        num_warps=8,
        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
    )
    return c
