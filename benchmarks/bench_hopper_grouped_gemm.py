import paddle

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
import numpy as np

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench_grouped_gemm(
    batch_size, num_tokens_per_group, d_in, d_out, dtype, output_dtype
):
    np.random.seed(42)
    W = paddle.randn(shape=[batch_size, d_out, d_in]).to(dtype)
    X = paddle.randn(shape=[batch_size * num_tokens_per_group, d_in]).to(dtype)
    Y = paddle.empty(
        shape=[batch_size * num_tokens_per_group, d_out], dtype=output_dtype
    )
    workspace_buffer = paddle.empty(shape=32 * 1024 * 1024, dtype="int8")
    segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer, backend="auto")
    seg_indptr = paddle.arange(
        start=0,
        end=(batch_size + 1) * num_tokens_per_group,
        step=num_tokens_per_group,
        dtype="int64",
    )
    measurements = bench_gpu_time(
        lambda: segment_gemm.run(X, W, batch_size, True, out=Y, seg_indptr=seg_indptr)
    )
    ms = np.median(measurements)
    flops = 2 * batch_size * num_tokens_per_group * d_in * d_out
    print(
        f"Config: batch_size={batch_size}, num_tokens_per_group={num_tokens_per_group}, d_in={d_in}, d_out={d_out}, dtype={dtype}, output_dtype={output_dtype}"
    )
    print(f"FLOPs: {flops / ms * 1e-09:.2f} TFLOPs/s")


if __name__ == "__main__":
    device_capability = paddle.device.cuda.get_device_capability()
    if device_capability[0] != 9:
        print(f"Current device capability: {device_capability}.")
        print("Current benchmark targets capability (9, 0). Returning...")
        exit()
>>>>>>    for dtype_in in [paddle.float8_e4m3fn, "bfloat16"]:
        for dtype_out in ["bfloat16"]:
            for batch_size in [1, 3, 8, 16]:
                for num_tokens_per_group in [32, 64, 128, 256, 512]:
                    for d_in in [4096, 8192]:
                        for d_out in [4096, 8192]:
                            bench_grouped_gemm(
                                batch_size,
                                num_tokens_per_group,
                                d_in,
                                d_out,
                                dtype_in,
                                dtype_out,
                            )
