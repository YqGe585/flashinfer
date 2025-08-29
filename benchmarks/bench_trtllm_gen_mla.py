import sys

sys.path.append("/home/flashinfer_paddle")
import numpy as np
import paddle
from paddle_utils import *

import flashinfer
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

num_q_heads = 128
num_kv_heads = 1
qk_nope_head_dim = 128
qk_rope_head_dim = 64
kv_lora_rank = 512


def bench_trtllm_mla(batch_size, q_len_per_request, seq_len, page_size, dtype):
    paddle.seed(seed=42)
    device = "cuda:0"
    query = paddle.randn(
        shape=[
            batch_size,
            q_len_per_request,
            num_q_heads,
            kv_lora_rank + qk_rope_head_dim,
        ]
    ).to(dtype)
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    seq_lens = [
        paddle.randint(low=1, high=seq_len, shape=(1,)).item()
        for _ in range(batch_size)
    ]
    seq_lens[-1] = seq_len
    max_seq_len = max(seq_lens)
    seq_lens_tensor = paddle.to_tensor(data=seq_lens, dtype="int32", place=device)
    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq._max().item()
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = paddle.randperm(n=total_blocks_needed)
    block_id = 0
    block_tables = paddle.zeros(
        shape=(batch_size, max_num_blocks_per_seq), dtype="int32"
    )
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed
    kv_cache = paddle.randn(
        shape=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim)
    ).to(dtype)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(axis=1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (128 + 64) ** 0.5,
        bmm2_scale=1.0,
    )
    measurements = bench_gpu_time_with_cudagraph(
        lambda: flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(axis=1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (128 + 64) ** 0.5,
            bmm2_scale=1.0,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    io = query.size * query.element_size() + kv_cache.size * kv_cache.element_size()
    ms = np.median(measurements)
    flops = (
        2
        * batch_size
        * num_q_heads
        * (2 * kv_lora_rank + qk_rope_head_dim)
        * seq_len
        * q_len_per_request
    )
    print(
        f"batch_size={batch_size}, q_len_per_request={q_len_per_request}, seq_len={seq_len}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, qk_nope_head_dim={qk_nope_head_dim}, qk_rope_head_dim={qk_rope_head_dim}, kv_lora_rank={kv_lora_rank}, page_size={page_size}"
    )
    print(f"execution time: {ms} ms")
    print(f"memory bandwidth: {io / ms / 1024 / 1024:.2f} GB/s")
    print(f"FLOPs: {flops * 1e-09 / ms:.2f} TFLOPs/s")


if __name__ == "__main__":
>>>>>>    for dtype in ["bfloat16", torch.float8_e4m3fn]:
        for page_size in [32, 64]:
            for batch_size in [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024]:
                for seq_len in [1024, 4096, 8192]:
                    for q_len_per_request in [1, 2, 4, 8, 16]:
                        bench_trtllm_mla(
                            batch_size, q_len_per_request, seq_len, page_size, dtype
                        )
