import paddle

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
import numpy as np

from flashinfer.gemm import (batch_deepgemm_fp8_nt_groupwise,
                             group_deepgemm_fp8_nt_groupwise)
from flashinfer.testing.utils import bench_gpu_time, quantize_fp8


def bench_deepgemm_grouped_fp8_blackwell(batch_size, m, n, k, in_dtype, out_dtype):
    """Benchmark DeepGEMM-based grouped GEMM with FP8 quantization."""
    a_f32 = paddle.randn(shape=[batch_size * m, k], dtype="float32")
    b_f32 = paddle.randn(shape=[batch_size, n, k], dtype="float32")
    a_fp8, a_scale = quantize_fp8(a_f32, (batch_size * m, k // 128), (1, 128), "K")
    b_fp8, b_scale = quantize_fp8(
        b_f32, (batch_size, n // 128, k // 128), (1, 128, 128), "K"
    )
    m_indices = paddle.arange(dtype="int32", end=batch_size).repeat_interleave(
        repeats=m
    )
    out = paddle.empty(shape=[batch_size * m, n], dtype=out_dtype)
    measurements = bench_gpu_time(
        lambda: group_deepgemm_fp8_nt_groupwise(
            a_fp8, b_fp8, a_scale, b_scale, m_indices, out=out, out_dtype=out_dtype
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * batch_size * m * n * k * 1e-09 / ms
    memory_bandwidth_per_second = (
        sum(
            [
                (_.size * _.element_size())
                for _ in [a_fp8, b_fp8, a_scale, b_scale, m_indices, out]
            ]
        )
        * 1e-09
        / ms
    )
    print(
        f"group_deepgemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/smemory_bandwidth: {memory_bandwidth_per_second:.2f} TB/s"
    )
    return tflops_per_second


def bench_deepgemm_batch_fp8_blackwell(batch_size, m, n, k, in_dtype, out_dtype):
    """Benchmark DeepGEMM-based batch GEMM with FP8 quantization."""
    a = paddle.randn(shape=(batch_size, m, k), dtype="float32")
    b = paddle.randn(shape=(batch_size, n, k), dtype="float32")
    masked_m = paddle.randint(low=0, high=m, shape=(batch_size,), dtype="int32")
    a_fp8, a_scale = quantize_fp8(a, (batch_size, m, k // 128), (1, 1, 128), "K")
    b_fp8, b_scale = quantize_fp8(
        b, (batch_size, n // 128, k // 128), (1, 128, 128), "K"
    )
    expected_m = min(int(masked_m.astype(dtype="float32").mean()) + 1, m)
    out = paddle.empty(shape=(batch_size, m, n), dtype=out_dtype)
    measurements = bench_gpu_time(
        lambda: batch_deepgemm_fp8_nt_groupwise(
            a_fp8,
            b_fp8,
            a_scale,
            b_scale,
            masked_m,
            expected_m,
            out=out,
            out_dtype=out_dtype,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * batch_size * m * n * k * 1e-09 / ms
    memory_bandwidth_per_second = (
        sum(
            [
                (_.size * _.element_size())
                for _ in [a_fp8, b_fp8, a_scale, b_scale, masked_m, out]
            ]
        )
        * 1e-09
        / ms
    )
    print(
        f"group_deepgemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/smemory_bandwidth: {memory_bandwidth_per_second:.2f} TB/s"
    )
    return tflops_per_second


if __name__ == "__main__":
    print("=== DeepGEMM Grouped FP8 GEMM Benchmark ===\n")
    for batch_size in [1, 4, 8, 64, 128, 256]:
        for m in [128, 256, 1024, 8192, 16384]:
            for n, k in [(128, 512), (512, 128), (4096, 7168), (7168, 2048)]:
                if m // batch_size < 128:
                    continue
                if m * batch_size <= 16384:
                    bench_deepgemm_grouped_fp8_blackwell(
>>>>>>                        batch_size, m, n, k, paddle.float8_e4m3fn, "bfloat16"
                    )
    for batch_size in [1, 4, 8, 64, 128, 256]:
        for m in [128, 256, 1024, 8192, 16384]:
            for n, k in [(128, 512), (512, 128), (4096, 7168), (7168, 2048)]:
                if m * batch_size <= 16384:
                    bench_deepgemm_batch_fp8_blackwell(
>>>>>>                        batch_size, m, n, k, paddle.float8_e4m3fn, "bfloat16"
                    )
