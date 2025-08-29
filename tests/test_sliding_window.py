import sys


import paddle
from flashinfer.paddle_utils import *

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
import pytest
from jit_utils import (gen_decode_attention_modules,
                       gen_prefill_attention_modules)

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            ["float16"], ["float16"], [64, 128, 256], [0], [False, True], [False]
        )
        + gen_prefill_attention_modules(
            ["float16"],
            ["float16"],
            [64, 128, 256],
            [0],
            [False, True],
            [False],
            [False],
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_decode_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = paddle.randn(shape=[num_qo_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    k_sliced = k[-(window_left + 1) :]
    v_sliced = v[-(window_left + 1) :]
    o_ref = flashinfer.single_decode_with_kv_cache(q, k_sliced, v_sliced)
    o = flashinfer.single_decode_with_kv_cache(q, k, v, window_left=window_left)
    assert paddle.allclose(x=o.cpu(), y=o_ref.cpu(), rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 3, 13, 32])
@pytest.mark.parametrize("kv_len", [1, 3, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [33, 533])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_decode_sliding_window(
    batch_size, kv_len, window_left, num_kv_heads, num_qo_heads, head_dim, page_size
):
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype="float16")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = paddle.randn(
        shape=[total_num_pages, page_size, num_kv_heads, head_dim], dtype="float16"
    )
    v_data = paddle.randn(
        shape=[total_num_pages, page_size, num_kv_heads, head_dim], dtype="float16"
    )
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages, dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=32 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
    )
    o = wrapper.run(q, (k_data, v_data))
    for i in range(batch_size):
        qi = q[i]
        ki = paddle.concat(
            x=[
                k_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                k_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            axis=0,
        )
        vi = paddle.concat(
            x=[
                v_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                v_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            axis=0,
        )
        o_ref_i = flashinfer.single_decode_with_kv_cache(
            qi, ki, vi, window_left=window_left
        )
        assert paddle.allclose(x=o[i], y=o_ref_i, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("seq_len", [1, 3, 19, 99, 199, 1999])
@pytest.mark.parametrize("window_left", [3, 13, 23, 43])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_decode_prefill_sliding_window_match(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = paddle.randn(shape=[1, num_qo_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, window_left=window_left, causal=True
    )
    o_decoded = flashinfer.single_decode_with_kv_cache(
        q[0], k, v, window_left=window_left
    )
    assert paddle.allclose(
        x=o.cpu()[0], y=o_decoded.cpu(), rtol=0.001, atol=0.001
    ).item(), ""


@pytest.mark.parametrize("seq_len", [99, 199, 1999])
@pytest.mark.parametrize("window_left", [43, 233])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_single_prefill_sliding_window(
    seq_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = paddle.randn(shape=[seq_len, num_qo_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    row_idx = paddle.arange(dtype="int32", end=seq_len)[:, None]
    col_idx = paddle.arange(dtype="int32", end=seq_len)[None, :]
    mask = (row_idx >= col_idx) & (row_idx - window_left <= col_idx)
    o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, window_left=window_left, causal=True
    )
    assert paddle.allclose(x=o.cpu(), y=o_ref.cpu(), rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 397])
@pytest.mark.parametrize("qo_len", [37, 47])
@pytest.mark.parametrize("window_left", [13, 33])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_paged_prefill_sliding_window(
    batch_size,
    kv_len,
    qo_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
):
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim], dtype="float16"
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = paddle.randn(
        shape=[total_num_pages, page_size, num_kv_heads, head_dim], dtype="float16"
    )
    v_data = paddle.randn(
        shape=[total_num_pages, page_size, num_kv_heads, head_dim], dtype="float16"
    )
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages, dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        window_left=window_left,
        causal=True,
    )
    o = wrapper.run(q, (k_data, v_data))
    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = paddle.concat(
            x=[
                k_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                k_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            axis=0,
        )
        vi = paddle.concat(
            x=[
                v_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                v_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            axis=0,
        )
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi, ki, vi, window_left=window_left, causal=True, backend="fa2"
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        assert paddle.allclose(x=o_i, y=o_ref_i, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 397])
@pytest.mark.parametrize("qo_len", [37, 47])
@pytest.mark.parametrize("window_left", [13, 33])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_batch_ragged_prefill_sliding_window(
    batch_size, kv_len, qo_len, window_left, num_kv_heads, num_qo_heads, head_dim
):
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim], dtype="float16"
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    k = paddle.randn(
        shape=[batch_size * kv_len, num_kv_heads, head_dim], dtype="float16"
    )
    v = paddle.randn(
        shape=[batch_size * kv_len, num_kv_heads, head_dim], dtype="float16"
    )
    kv_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * kv_len
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        window_left=window_left,
        causal=True,
    )
    o = wrapper.run(q, k, v)
    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = k[kv_indptr[i] : kv_indptr[i + 1]]
        vi = v[kv_indptr[i] : kv_indptr[i + 1]]
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi, ki, vi, window_left=window_left, causal=True
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        assert paddle.allclose(x=o_i, y=o_ref_i, rtol=0.001, atol=0.001).item(), ""


if __name__ == "__main__":
    test_single_decode_sliding_window(13, 20, 1, 4, 128)
    test_single_prefill_sliding_window(13, 20, 1, 4, 128)
    test_batch_paged_prefill_sliding_window(12, 54, 37, 13, 1, 4, 128, 1)
    test_batch_ragged_prefill_sliding_window(12, 54, 37, 13, 1, 4, 128)
