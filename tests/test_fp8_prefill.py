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
import pytest

import flashinfer


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("qo_len", [1, 7, 53])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
>>>>>>@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
):
    paddle.seed(seed=42)
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim], dtype="float16"
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.05
        * paddle.randn(
            shape=[total_num_pages, 2, num_kv_heads, page_size, head_dim],
            dtype="float16",
        ).to(0)
        if kv_layout == "HND"
        else 0.05
        * paddle.randn(
            shape=[total_num_pages, 2, page_size, num_kv_heads, head_dim],
            dtype="float16",
        ).to(0)
    )
    qo_indptr = (
        paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32") * qo_len
    )
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
        * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages).to(0).astype(dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    ).to(0)
    workspace_buffer = paddle.empty(shape=32 * 1024 * 1024, dtype="int8").to(0)
    wrapper_f16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper_f16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="float16",
        kv_data_type="float16",
    )
    o_fp16 = wrapper_f16.run(q, kv_data)
    k_data, v_data = paddle.chunk(x=kv_data, chunks=2, axis=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256
    k_fp8 = (k_data / k_scale).to(dtype)
    v_fp8 = (v_data / v_scale).to(dtype)
    kv_data_fp8 = paddle.concat(x=[k_fp8, v_fp8], axis=1)
    wrapper_f8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper_f8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="float16",
        kv_data_type=dtype,
    )
    o_fp8 = wrapper_f8.run(q, kv_data_fp8.to(dtype), k_scale=k_scale, v_scale=v_scale)
    assert paddle.allclose(x=o_fp16, y=o_fp8, atol=0.01, rtol=0.2).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
>>>>>>@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_decode_with_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
):
    paddle.seed(seed=42)
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype="float16").to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.1
        * paddle.randn(
            shape=[total_num_pages, 2, num_kv_heads, page_size, head_dim],
            dtype="float16",
        ).to(0)
        if kv_layout == "HND"
        else 0.1
        * paddle.randn(
            shape=[total_num_pages, 2, page_size, num_kv_heads, head_dim],
            dtype="float16",
        ).to(0)
    ).to(dtype)
    qo_indptr = paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
        * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages).to(0).astype(dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    ).to(0)
    workspace_buffer = paddle.empty(shape=32 * 1024 * 1024, dtype="int8").to(0)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="float16",
        kv_data_type=dtype,
    )
    o_fp8 = wrapper.run(q, kv_data)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="float16",
        kv_data_type=dtype,
    )
    o_decode_fp8 = decode_wrapper.run(q, kv_data)
    assert paddle.allclose(x=o_decode_fp8, y=o_fp8, atol=0.01, rtol=0.01).item(), ""


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale(
>>>>>>        12, 7, 54, 1, 4, 4, 128, "NHD", torch.float8_e5m2
    )
    test_batch_decode_with_prefill_with_paged_kv_cache(
>>>>>>        12, 54, 1, 4, 4, 128, "NHD", torch.float8_e5m2
    )
