import numpy as np
import paddle

import flashinfer
from flashinfer.testing.utils import (bench_gpu_time,
                                      bench_gpu_time_with_cudagraph)

page_size = 16
num_kv_heads = 4
num_qo_heads = 32
head_dim = 128
workspace_buffer = paddle.empty(shape=1024 * 1024 * 1024, dtype="uint8")


def bench_trtllm_fmha(batch_size, seq_len, kv_cache_dtype):
    paddle.seed(seed=42)
    seq_lens = paddle.full(shape=(batch_size,), fill_value=seq_len, dtype="int32")
    seq_lens_blocks = paddle.ceil(x=seq_lens / page_size).astype(dtype="int32")
    kv_indptr = paddle.zeros(shape=batch_size + 1, dtype="int32")
    kv_indptr[1:] = paddle.cumsum(x=seq_lens_blocks, axis=0)
    last_page_len = (seq_lens - (seq_lens_blocks - 1) * page_size).astype(dtype="int32")
    last_page_len[last_page_len == 0] = page_size
    num_blocks = kv_indptr[-1].item()
    kv_indices = paddle.arange(dtype="int32", end=num_blocks)
    q = paddle.rand(shape=[batch_size, num_qo_heads, head_dim]).to("bfloat16")
    kv_data = paddle.randn(shape=[num_blocks, 2, num_kv_heads, page_size, head_dim]).to(
        paddle.float8_e4m3fn if kv_cache_dtype == "fp8" else "float16"
    )
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "HND", backend="trtllm-gen"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=kv_data.dtype,
    )
    wrapper.run(q, kv_data)
    paddle.device.synchronize()
    measurements = bench_gpu_time(lambda: wrapper.run(q, kv_data))
    ms = np.median(measurements)
    io = q.size * q.element_size() + kv_data.size * kv_data.element_size()
    print(
        f"batch_size={batch_size}, seq_len={seq_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}"
    )
    print(f"execution time: {ms}ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024:.2f} GB/s")


def to_float8(x, dtype=paddle.float8_e4m3fn):
    finfo = paddle.finfo(dtype=dtype)
    min_val, max_val = tuple(
        [
            paddle.amin(x, axis=None, keepdim=False),
            paddle.max(x, axis=None, keepdim=False),
        ]
    )
    amax = paddle.maximum(x=min_val.abs(), y=max_val.abs()).clip(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clip(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.astype(dtype="float32").reciprocal()


def bench_trtllm_fmha_wrapper(
    kv_layout,
    batch_size,
    max_seq_len,
    page_size,
    num_kv_heads,
    head_dim,
    q_dtype,
    head_grp_size,
    kv_cache_dtype,
    window_left,
    bench_with_sink,
):
    paddle.seed(seed=42)
    device = "cuda:0"
    num_qo_heads = num_kv_heads * head_grp_size
    batch_size = batch_size
    num_tokens = max_seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    dtype_map = {"half": "float16", "bf16": "bfloat16", "fp8": paddle.float8_e4m3fn}
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim]).to(dtype_map[q_dtype])
    seq_lens = paddle.full(shape=(batch_size,), fill_value=max_seq_len)
    seq_lens_tensor = paddle.to_tensor(data=seq_lens, dtype="int32", place=device)
    blocks_per_seq = [((seq_len + page_size - 1) // page_size) for seq_len in seq_lens]
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = paddle.randperm(n=total_blocks_needed)
    kv_cache_shape = num_blocks, 2, num_kv_heads, page_size, head_dim
    kv_cache = paddle.randn(shape=kv_cache_shape).to(q.dtype)
    if kv_cache_dtype.startswith("fp8") and q_dtype != "fp8":
        kv_cache, _ = to_float8(kv_cache)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    sinks = (
        paddle.randn(shape=num_qo_heads, dtype="float32") if bench_with_sink else None
    )
    kv_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(x=blocks_per_seq, axis=0),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indices = all_block_ids.astype(dtype="int32")
    kv_last_page_len = seq_lens_tensor % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "HND", backend="trtllm-gen"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=kv_cache.dtype,
        q_data_type=q.dtype,
        window_left=window_left,
    )
    wrapper.run(q, kv_cache, sinks=sinks)
    paddle.device.synchronize()
    measurements = bench_gpu_time_with_cudagraph(
        lambda: wrapper.run(q, kv_cache, sinks=sinks),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)
    io = q.size * q.element_size() + kv_cache.size * kv_cache.element_size()
    print(
        f"batch_size={batch_size}, seq_len={max_seq_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, page_size={page_size}"
    )
    print(f"execution time: {ms}ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024:.2f} GB/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark TRTLLM FMHA")
    parser.add_argument(
        "--head_dim", type=int, default=64, help="Dimension of each head"
    )
    parser.add_argument(
        "--num_kv_heads", type=int, default=8, help="Number of key/value heads"
    )
    parser.add_argument(
        "--page_size", type=int, default=16, help="Size of each page [16, 32, 64]"
    )
    parser.add_argument(
        "--head_grp_size",
        type=int,
        default=8,
        help="Number of query heads per key-value head (group size)",
    )
    parser.add_argument("--sink", action="store_true", help="Whether to test with sink")
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[4, 128, 256],
        help="List of batch sizes to test",
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 16384],
        help="List of sequence lengths to test",
    )
    args = parser.parse_args()
    for batch_size in args.batch_sizes:
        for seq_len in args.seq_lens:
            bench_trtllm_fmha_wrapper(
                kv_layout="HND",
                batch_size=batch_size,
                max_seq_len=seq_len,
                page_size=args.page_size,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                q_dtype="bf16",
                head_grp_size=args.head_grp_size,
                kv_cache_dtype="auto",
                window_left=-1,
                bench_with_sink=args.sink,
            )
