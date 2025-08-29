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


def bench_single_prefill(seq_len, num_heads, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = paddle.randn(shape=[seq_len, num_qo_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    sm80_ms, sm90_ms = (
        np.median(
            bench_gpu_time(
                lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                    q, k, v, causal=causal, backend=backend
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
        for backend in ["fa2", "fa3"]
    )

    def flops(ms):
        if causal:
            return seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1000000000.0
        else:
            return seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1000000000.0

    print(
        f"bench_single_prefill (seq_len={seq_len}, num_heads={num_heads}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )


def bench_batch_ragged_prefill(batch_size, num_heads, seq_len, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = paddle.randn(
        shape=[batch_size * seq_len, num_qo_heads, head_dim], dtype="float16"
    )
    k = paddle.randn(
        shape=[batch_size * seq_len, num_kv_heads, head_dim], dtype="float16"
    )
    v = paddle.randn(
        shape=[batch_size * seq_len, num_kv_heads, head_dim], dtype="float16"
    )
    sm80_wrapper, sm90_wrapper = (
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            paddle.empty(shape=256 * 1024 * 1024, dtype="uint8"),
            kv_layout="NHD",
            backend=backend,
        )
        for backend in ["fa2", "fa3"]
    )
    qo_indptr = paddle.arange(
        start=0, end=batch_size * seq_len + 1, step=seq_len
    ).astype(dtype="int32")
    kv_indptr = paddle.arange(
        start=0, end=batch_size * seq_len + 1, step=seq_len
    ).astype(dtype="int32")
    for wrapper in [sm80_wrapper, sm90_wrapper]:
        wrapper.plan(
            qo_indptr, kv_indptr, num_qo_heads, num_kv_heads, head_dim, causal=causal
        )
    sm80_ms, sm90_ms = (
        np.median(
            bench_gpu_time(
                lambda: wrapper.run(q, k, v), dry_run_time_ms=100, repeat_time_ms=1000
            )
        )
        for wrapper in [sm80_wrapper, sm90_wrapper]
    )

    def flops(ms):
        if causal:
            return (
                batch_size
                * seq_len
                * seq_len
                * num_qo_heads
                * head_dim
                * 2
                / ms
                / 1000000000.0
            )
        else:
            return (
                batch_size
                * seq_len
                * seq_len
                * num_qo_heads
                * head_dim
                * 4
                / ms
                / 1000000000.0
            )

    print(
        f"bench_batch_ragged_prefill (batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )


def bench_batch_paged_prefill(
    page_size, batch_size, num_heads, seq_len, causal, head_dim
):
    num_qo_heads = num_kv_heads = num_heads
    q = paddle.randn(
        shape=[batch_size * seq_len, num_qo_heads, head_dim], dtype="float16"
    )
    k = paddle.randn(
        shape=[batch_size * seq_len // page_size, page_size, num_kv_heads, head_dim],
        dtype="float16",
    )
    v = paddle.randn(
        shape=[batch_size * seq_len // page_size, page_size, num_kv_heads, head_dim],
        dtype="float16",
    )
    sm80_wrapper, sm90_wrapper = (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            paddle.empty(shape=256 * 1024 * 1024, dtype="uint8"),
            kv_layout="NHD",
            backend=backend,
        )
        for backend in ["fa2", "fa3"]
    )
    qo_indptr = paddle.arange(
        start=0, end=batch_size * seq_len + 1, step=seq_len
    ).astype(dtype="int32")
    kv_indptr = paddle.arange(
        start=0, end=batch_size * (seq_len // page_size) + 1, step=seq_len // page_size
    ).astype(dtype="int32")
    kv_indices = paddle.arange(start=0, end=batch_size * (seq_len // page_size)).astype(
        dtype="int32"
    )
    last_page_len = paddle.ones(shape=batch_size, dtype="int32") * page_size
    for wrapper in [sm80_wrapper, sm90_wrapper]:
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
        )
    sm80_ms, sm90_ms = (
        np.median(
            bench_gpu_time(
                lambda: wrapper.run(q, (k, v)), dry_run_time_ms=100, repeat_time_ms=1000
            )
        )
        for wrapper in [sm80_wrapper, sm90_wrapper]
    )

    def flops(ms):
        if causal:
            return (
                batch_size
                * seq_len
                * seq_len
                * num_qo_heads
                * head_dim
                * 2
                / ms
                / 1000000000.0
            )
        else:
            return (
                batch_size
                * seq_len
                * seq_len
                * num_qo_heads
                * head_dim
                * 4
                / ms
                / 1000000000.0
            )

    print(
        f"bench_batch_paged_prefill (page_size={page_size} batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    device_capability = paddle.device.cuda.get_device_capability()
    if device_capability[0] != 9:
        print(f"Current device capability: {device_capability}.")
        print("Current benchmark targets capability (9, 0). Returning...")
        exit()
    bench_batch_paged_prefill(1, 128, 32, 1024, True, 128)
    bench_batch_paged_prefill(1, 64, 32, 2048, True, 128)
    bench_batch_paged_prefill(1, 32, 32, 4096, True, 128)
    bench_batch_paged_prefill(1, 16, 32, 8192, True, 128)
    bench_batch_paged_prefill(1, 1, 32, 32768, True, 128)
    bench_batch_paged_prefill(16, 128, 32, 1024, True, 128)
    bench_batch_paged_prefill(16, 64, 32, 2048, True, 128)
    bench_batch_paged_prefill(16, 32, 32, 4096, True, 128)
    bench_batch_paged_prefill(16, 16, 32, 8192, True, 128)
    bench_batch_paged_prefill(16, 1, 32, 32768, True, 128)
    bench_batch_ragged_prefill(128, 32, 1024, True, 128)
    bench_batch_ragged_prefill(64, 32, 2048, True, 128)
    bench_batch_ragged_prefill(32, 32, 4096, True, 128)
    bench_batch_ragged_prefill(16, 32, 8192, True, 128)
    bench_batch_ragged_prefill(1, 32, 32768, True, 128)
