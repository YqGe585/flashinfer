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

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench_groupwise_grouped_gemm_fp8_blackwell(
    batch_size, m, n, k, in_dtype, out_dtype
):
    paddle.seed(seed=0)
    a = paddle.randn(shape=[batch_size * m, k]).to(in_dtype)
    b = paddle.randn(shape=[batch_size, n, k]).to(in_dtype)
    out = paddle.empty(shape=[batch_size * m, n], dtype=out_dtype)
    a_scale = paddle.randn(shape=(k // 128, batch_size * m), dtype="float32")
    b_scale = paddle.randn(shape=(batch_size, k // 128, n // 128), dtype="float32")
    segment_offsets = paddle.arange(
        start=0, end=(batch_size + 1) * m, step=m, dtype="int32"
    )
    measurements = bench_gpu_time(
        lambda: flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, segment_offsets, out=out, mma_sm=2
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * batch_size * m * n * k * 1e-09 / ms
    print(
        f"group_gemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops_per_second:.2f} TFLOPs/s"
    )


if __name__ == "__main__":
    for batch_size in [1, 3, 8, 16]:
        for m in [128, 512, 1024, 2048, 4096, 8192]:
            for n in [1024, 2048, 4096, 8192]:
                for k in [1024, 2048, 4096, 8192]:
                    bench_groupwise_grouped_gemm_fp8_blackwell(
>>>>>>                        batch_size, m, n, k, torch.float8_e5m2, "bfloat16"
                    )
