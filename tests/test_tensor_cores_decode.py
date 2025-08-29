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
from jit_utils import (gen_decode_attention_modules,
                       gen_prefill_attention_modules)

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            ["float16"], ["float16"], [64, 128, 256], [0, 1], [False], [False]
        )
        + gen_prefill_attention_modules(
            ["float16"], ["float16"], [64, 128, 256], [0, 1], [False], [False], [False]
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("kv_len", [54, 128, 999, 32789])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_single_decode_tensor_cores(
    kv_len: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = paddle.randn(shape=[num_qo_heads, head_dim], dtype="float16")
    k = (
        paddle.randn(shape=[num_kv_heads, kv_len, head_dim], dtype="float16")
        if kv_layout == "HND"
        else paddle.randn(shape=[kv_len, num_kv_heads, head_dim], dtype="float16")
    )
    v = (
        paddle.randn(shape=[num_kv_heads, kv_len, head_dim], dtype="float16")
        if kv_layout == "HND"
        else paddle.randn(shape=[kv_len, num_kv_heads, head_dim], dtype="float16")
    )
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout, pos_encoding_mode, use_tensor_cores=False
    )
    o_tensor_cores = flashinfer.single_decode_with_kv_cache(
        q, k, v, kv_layout, pos_encoding_mode, use_tensor_cores=True
    )
    assert paddle.allclose(x=o, y=o_tensor_cores, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97, 512])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_decode_tensor_cores(
    batch_size: int,
    kv_len: int,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype="float16")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        paddle.randn(
            shape=[total_num_pages, 2, num_kv_heads, page_size, head_dim],
            dtype="float16",
        )
        / 10
        if kv_layout == "HND"
        else paddle.randn(
            shape=[total_num_pages, 2, page_size, num_kv_heads, head_dim],
            dtype="float16",
        )
        / 10
    )
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages, dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type="float16",
        q_data_type="float16",
    )
    o, lse = wrapper.run(q, kv_data, return_lse=True)
    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type="float16",
        q_data_type="float16",
    )
    o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
        q, kv_data, return_lse=True
    )
    assert paddle.allclose(x=o, y=o_tensor_cores, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=lse, y=lse_tensor_cores, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97, 512])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_decode_tensor_cores_cuda_graph(
    batch_size: int,
    kv_len: int,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype="float16")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        paddle.randn(
            shape=[total_num_pages, 2, num_kv_heads, page_size, head_dim],
            dtype="float16",
        )
        / 10
        if kv_layout == "HND"
        else paddle.randn(
            shape=[total_num_pages, 2, page_size, num_kv_heads, head_dim],
            dtype="float16",
        )
        / 10
    )
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages, dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_cuda_graph=True,
        paged_kv_indptr_buffer=kv_indptr,
        paged_kv_indices_buffer=kv_indices,
        paged_kv_last_page_len_buffer=kv_last_page_len,
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type="float16",
        q_data_type="float16",
    )
    s = paddle.device.Stream()
    s.wait_stream(paddle.device.current_stream())
    with paddle.device.stream_guard(stream=s):
        for _ in range(3):
            o, lse = wrapper.run(q, kv_data, return_lse=True)
    paddle.device.current_stream().wait_stream(s)
>>>>>>    g = torch.cuda.CUDAGraph()
>>>>>>    with torch.cuda.graph(g):
        o, lse = wrapper.run(q, kv_data, return_lse=True)
    g.replay()
    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=kv_indptr,
        paged_kv_indices_buffer=kv_indices,
        paged_kv_last_page_len_buffer=kv_last_page_len,
    )
    wrapper_tensor_cores.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type="float16",
        q_data_type="float16",
    )
    s = paddle.device.Stream()
    s.wait_stream(paddle.device.current_stream())
    with paddle.device.stream_guard(stream=s):
        for _ in range(3):
            o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
                q, kv_data, return_lse=True
            )
    paddle.device.current_stream().wait_stream(s)
>>>>>>    g = torch.cuda.CUDAGraph()
>>>>>>    with torch.cuda.graph(g):
        o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
            q, kv_data, return_lse=True
        )
    g.replay()
    assert paddle.allclose(x=o, y=o_tensor_cores, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=lse, y=lse_tensor_cores, rtol=0.001, atol=0.001).item(), ""
