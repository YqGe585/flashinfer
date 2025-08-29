import sys

sys.path.append("/home/flashinfer")
import paddle
import pytest
from jit_utils import (gen_decode_attention_modules,
                       gen_prefill_attention_modules)
from paddle_utils import *

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            ["float16"], ["float16"], [64, 128, 256], [0], [False], [False]
        )
        + gen_prefill_attention_modules(
            ["float16"], ["float16"], [64, 128, 256], [0], [False], [False], [False]
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("page_size", [1, 5])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_batch_paged_decode_packed_input(
    batch_size, page_size, seq_len, num_kv_heads, num_qo_heads, head_dim
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    nnz = batch_size * seq_len
    num_pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_req
    last_page_len = (seq_len - 1) % page_size + 1
    k_cache = paddle.randn(
        shape=(num_pages, page_size, num_kv_heads, head_dim), dtype="float16"
    )
    v_cache = paddle.randn(shape=k_cache.shape, dtype=k_cache.dtype)
    paged_kv_cache = k_cache, v_cache
    workspace_buffer = paddle.empty(shape=(256 * 1024 * 1024,), dtype="uint8")
    paged_kv_indptr = paddle.to_tensor(
        data=[(i * num_pages_per_req) for i in range(batch_size + 1)],
        dtype="int32",
        place="gpu:0",
    )
    paged_kv_indices = paddle.to_tensor(
        data=list(range(num_pages)), dtype="int32", place="gpu:0"
    )
    paged_kv_last_page_len = paddle.to_tensor(
        data=[last_page_len for _ in range(batch_size)], dtype="int32", place="gpu:0"
    )
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
    wrapper.plan(
        indptr=paged_kv_indptr,
        indices=paged_kv_indices,
        last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    qkv_packed = paddle.randn(
        shape=(nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim), dtype="float16"
    )
    qkv_split_idx = (
        num_qo_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, _, _ = qkv_packed.split(qkv_split_idx, dim=-1)
    q = q.view(-1, num_qo_heads, head_dim)
    o_packed = wrapper.run(q, paged_kv_cache)
    o_contiguous = wrapper.run(q.contiguous(), paged_kv_cache)
    assert paddle.allclose(
        x=o_packed, y=o_contiguous, rtol=0.001, atol=0.001
    ).item(), ""
