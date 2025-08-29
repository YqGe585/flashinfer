import numpy as np
import paddle
import triton

import flashinfer
import flashinfer.triton
from flashinfer.testing.utils import bench_gpu_time


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and paddle.device.cuda.get_device_capability()[0] >= 9


def bench_gemm_persistent(num_sms, dtype, M, N, K, reps=1000, warmup_reps=10000):
    measurements = bench_gpu_time(
        lambda: flashinfer.triton.sm_constraint_gemm.gemm_persistent(
            a=paddle.randn(shape=(M, K), dtype="float16").to(dtype),
            b=paddle.randn(shape=(N, K), dtype="float16").to(dtype),
            alpha=1.0,
            beta=0.0,
            num_sms=num_sms,
        ),
        dry_run_time_ms=warmup_reps,
        repeat_time_ms=reps,
    )
    ms = np.median(measurements)
    flops = (2 * M * N * K + 3 * M * N) / ms / 1000000000.0
    print(
        f"GEMM Persistent | num_sms: {num_sms}, M: {M}, N: {N}, K: {K}, {dtype}: {flops:.3f} TFLOPs/s"
    )


def bench_gemm_descriptor_persistent(
    num_sms, dtype, M, N, K, reps=1000, warmup_reps=10000
):
    if dtype == "float32":
        return
    measurements = bench_gpu_time(
        lambda: flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(
            a=paddle.randn(shape=(M, K), dtype="float16").to(dtype),
            b=paddle.randn(shape=(N, K), dtype="float16").to(dtype),
            alpha=1.0,
            beta=0.0,
            num_sms=num_sms,
        ),
        dry_run_time_ms=warmup_reps,
        repeat_time_ms=reps,
    )
    ms = np.median(measurements)
    flops = (2 * M * N * K + 3 * M * N) / ms / 1000000000.0
    print(
        f"GEMM Descriptor | num_sms: {num_sms}, M: {M}, N: {N}, K: {K}, {dtype}: {flops:.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    assert supports_tma()
    for M, N, K in [(4096, 4096, 4096), (8192, 8192, 8192)]:
>>>>>>        for dtype in [torch.float8_e4m3fn, "float16", "bfloat16", "float32"]:
            for num_sms in [1, 16, 32, 64, 128, 132, 133, 256]:
                bench_gemm_persistent(num_sms, dtype, M, N, K)
                bench_gemm_descriptor_persistent(num_sms, dtype, M, N, K)
