import sys


import einops
import paddle
from flashinfer.paddle_utils import *

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
import pytest
from sink_attention_reference import sink_attention_unified

import flashinfer


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_decode_attention_sink(
    dtype, batch_size, page_size, seq_len, num_qo_heads, num_kv_heads, head_dim
):
    seed = 0
    paddle.seed(seed=seed)
    device = "cuda:0"
    seq_lens = paddle.full(shape=(batch_size,), fill_value=seq_len, dtype="int32")
    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = paddle.max(x=blocks_per_seq).item()
    block_tables = paddle.arange(
        dtype="int32", end=batch_size * max_num_blocks_per_seq
    ).reshape(batch_size, max_num_blocks_per_seq)
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype=dtype)
    k_cache = paddle.randn(
        shape=[num_blocks, num_kv_heads, page_size, head_dim], dtype=dtype
    )
    v_cache = paddle.randn(
        shape=[num_blocks, num_kv_heads, page_size, head_dim], dtype=dtype
    )
    sink = paddle.rand(shape=num_qo_heads, dtype="float32") * 5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        (k_cache, v_cache),
        workspace_buffer,
        block_tables,
        seq_lens,
        seq_len,
        1.0,
        1.0,
        -1,
        out_dtype=dtype,
        sinks=sink,
    )
    k = einops.rearrange(
        k_cache,
        "(b num_pages_per_b) h p d -> b (num_pages_per_b p) h d",
        num_pages_per_b=max_num_blocks_per_seq,
    )
    v = einops.rearrange(
        v_cache,
        "(b num_pages_per_b) h p d -> b (num_pages_per_b p) h d",
        num_pages_per_b=max_num_blocks_per_seq,
    )
    o_ref = sink_attention_unified(q, k, v, sink, -1, False, 1.0, mode="incremental")
    if dtype == "float16":
        atol, rtol = 0.001, 0.001
    elif dtype == "bfloat16":
        atol, rtol = 0.01, 0.01
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    assert paddle.allclose(x=o_ref, y=output, atol=atol, rtol=rtol).item(), ""


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_context_attention_sink(
    dtype, batch_size, page_size, seq_len, num_qo_heads, num_kv_heads, head_dim
):
    seed = 0
    paddle.seed(seed=seed)
    device = "cuda:0"
    seq_lens = paddle.full(shape=(batch_size,), fill_value=seq_len, dtype="int32")
    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = paddle.max(x=blocks_per_seq).item()
    block_tables = paddle.arange(
        dtype="int32", end=batch_size * max_num_blocks_per_seq
    ).reshape(batch_size, max_num_blocks_per_seq)
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = paddle.randn(shape=[num_tokens, num_qo_heads, head_dim], dtype=dtype)
    k_cache = paddle.randn(
        shape=[num_blocks, num_kv_heads, page_size, head_dim], dtype=dtype
    )
    v_cache = paddle.randn(
        shape=[num_blocks, num_kv_heads, page_size, head_dim], dtype=dtype
    )
    sink = paddle.rand(shape=num_qo_heads, dtype="float32") * 5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * seq_len
    kv_indptr = paddle.arange(start=0, end=num_blocks + 1, dtype="int32") * page_size
    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        (k_cache, v_cache),
        workspace_buffer,
        block_tables,
        seq_lens,
        seq_len,
        seq_len,
        1.0,
        1.0,
        batch_size,
        q_indptr,
        kv_indptr,
        -1,
        out_dtype=dtype,
        sinks=sink,
    )
    k = einops.rearrange(k_cache, "num_pages h p d -> (num_pages p) h d")
    v = einops.rearrange(v_cache, "num_pages h p d -> (num_pages p) h d")
    print(tuple(q.shape), tuple(k.shape), tuple(v.shape))
    o_ref = sink_attention_unified(
        q, k, v, sink, -1, True, 1.0, mode="prefill", batch_size=batch_size
    )
    if dtype == "float16":
        atol, rtol = 0.001, 0.001
    elif dtype == "bfloat16":
        atol, rtol = 0.01, 0.01
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    assert paddle.allclose(x=o_ref, y=output, atol=atol, rtol=rtol).item(), ""
