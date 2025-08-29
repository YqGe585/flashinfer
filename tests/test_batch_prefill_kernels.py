import sys


import paddle
from flashinfer.paddle_utils import *

"""
Copyright (c) 2023 by FlashInfer team.

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
import numpy
import pytest
from jit_utils import gen_prefill_attention_modules

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            ["float16"],
>>>>>>            ["float16", paddle.float8_e4m3fn, paddle.float8_e5m2],
            [128, 256],
            [0, 1],
            [False],
            [False],
            [False],
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("use_cuda_graph", [True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim], dtype="float16"
    )
    q_indptr_cpu = (
        paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32") * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = paddle.randn(shape=kv_shape, dtype="float32")
        kv_data = kv_data_fp32.astype(dtype="float16")
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        assert (
            kv_data.get_strides()[-4]
            != tuple(kv_data.shape)[-3]
            * tuple(kv_data.shape)[-2]
            * tuple(kv_data.shape)[-1]
        )
    else:
        kv_data_fp32 = paddle.randn(shape=kv_shape, dtype="float32")
        kv_data = kv_data_fp32.astype(dtype="float16")
    kv_indptr_cpu = (
        paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32")
        * num_pages_per_seq
    )
    kv_indices_cpu = paddle.arange(start=0, end=total_num_pages).astype(dtype="int32")
    kv_last_page_len_cpu = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=256 * 1024 * 1024, dtype="int8")
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
        o_buffer = paddle.empty_like(x=o)
        wrapper.run(q, kv_data, out=o_buffer)
        assert paddle.allclose(x=o, y=o_buffer, rtol=0.001, atol=0.001).item(), ""
    else:
        q_indptr_buffer = paddle.empty(shape=batch_size + 1, dtype="int32")
        kv_indptr_buffer = paddle.empty(shape=batch_size + 1, dtype="int32")
        kv_indices_buffer = paddle.empty(shape=total_num_pages, dtype="int32")
        kv_last_page_len_buffer = paddle.empty(shape=batch_size, dtype="int32")
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = (
            paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32") * qo_len
        )
        kv_indptr_warmup = paddle.arange(start=0, end=batch_size + 1).astype(
            dtype="int32"
        )
        kv_indices_warmup = paddle.arange(start=0, end=batch_size).astype(dtype="int32")
        kv_last_page_len_warmup = paddle.full(
            shape=(batch_size,), fill_value=page_size, dtype="int32"
        )
        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        s = paddle.device.Stream()
        s.wait_stream(paddle.device.current_stream())
        with paddle.device.stream_guard(stream=s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        paddle.device.current_stream().wait_stream(s)
>>>>>>        g = torch.cuda.CUDAGraph()
>>>>>>        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)
        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        g.replay()
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = paddle.concat(
            x=[
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .transpose(perm=perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            axis=0,
        ).astype(dtype="float16")
        vi = paddle.concat(
            x=[
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .transpose(perm=perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            axis=0,
        ).astype(dtype="float16")
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        assert paddle.allclose(x=o_i, y=o_ref_i, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim], dtype="float16"
    )
    q_indptr_cpu = (
        paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32") * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [paddle.randn(shape=kv_shape, dtype="float32") for _ in range(2)]
        kv_data = [kv_data_fp32[i].astype(dtype="float16") for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            assert (
                kv_data[i].get_strides()[-4]
                != tuple(kv_data[i].shape)[-3]
                * tuple(kv_data[i].shape)[-2]
                * tuple(kv_data[i].shape)[-1]
            )
    else:
        kv_data_fp32 = [paddle.randn(shape=kv_shape, dtype="float32") for _ in range(2)]
        kv_data = [kv_data_fp32[i].astype(dtype="float16") for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr_cpu = (
        paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32")
        * num_pages_per_seq
    )
    kv_indices_cpu = paddle.arange(start=0, end=total_num_pages).astype(dtype="int32")
    kv_last_page_len_cpu = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=256 * 1024 * 1024, dtype="int8")
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
    else:
        q_indptr_buffer = paddle.empty(shape=batch_size + 1, dtype="int32")
        kv_indptr_buffer = paddle.empty(shape=batch_size + 1, dtype="int32")
        kv_indices_buffer = paddle.empty(shape=total_num_pages, dtype="int32")
        kv_last_page_len_buffer = paddle.empty(shape=batch_size, dtype="int32")
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = (
            paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32") * qo_len
        )
        kv_indptr_warmup = paddle.arange(start=0, end=batch_size + 1).astype(
            dtype="int32"
        )
        kv_indices_warmup = paddle.arange(start=0, end=batch_size).astype(dtype="int32")
        kv_last_page_len_warmup = paddle.full(
            shape=(batch_size,), fill_value=page_size, dtype="int32"
        )
        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        s = paddle.device.Stream()
        s.wait_stream(paddle.device.current_stream())
        with paddle.device.stream_guard(stream=s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        paddle.device.current_stream().wait_stream(s)
>>>>>>        g = torch.cuda.CUDAGraph()
>>>>>>        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)
        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        g.replay()
    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = paddle.concat(
            x=[
                k_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .transpose(perm=perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            axis=0,
        ).astype(dtype="float16")
        vi = paddle.concat(
            x=[
                v_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .transpose(perm=perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            axis=0,
        ).astype(dtype="float16")
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        assert paddle.allclose(x=o_i, y=o_ref_i, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_paged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim], dtype="float16"
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data = paddle.randn(shape=kv_shape, dtype="float16")
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        assert (
            kv_data.get_strides()[-4]
            != tuple(kv_data.shape)[-3]
            * tuple(kv_data.shape)[-2]
            * tuple(kv_data.shape)[-1]
        )
    else:
        kv_data = paddle.randn(shape=kv_shape, dtype="float16")
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * num_pages_per_seq
    )
    kv_indices = paddle.arange(start=0, end=total_num_pages, dtype="int32")
    kv_last_page_len = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=256 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    custom_mask = paddle.tril(
        x=paddle.full(shape=(batch_size, qo_len, kv_len), fill_value=True),
        diagonal=kv_len - qo_len,
    ).reshape(-1)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        custom_mask=custom_mask,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_custom, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o_custom = wrapper.run(q, kv_data)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_causal, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o_causal = wrapper.run(q, kv_data)
    assert paddle.allclose(x=o_custom, y=o_causal, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    kv_layout = "NHD"
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
    workspace_buffer = paddle.empty(shape=256 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o = wrapper.run(q, k, v)
    for i in range(batch_size):
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            q[q_indptr[i] : q_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        assert paddle.allclose(x=o_i, y=o_ref_i, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_ragged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    kv_layout = "NHD"
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
    workspace_buffer = paddle.empty(shape=256 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    custom_mask = paddle.tril(
        x=paddle.full(shape=(batch_size, qo_len, kv_len), fill_value=True),
        diagonal=kv_len - qo_len,
    ).reshape(-1)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        custom_mask=custom_mask,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_custom, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o_custom = wrapper.run(q, k, v)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o_causal, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o_causal = wrapper.run(q, k, v)
    assert paddle.allclose(x=o_custom, y=o_causal, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
        (97, 81, 16, list(range(80)) + [0], 97, [79]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
):
    q = (
        paddle.randn(shape=[batch_size * qo_len, num_qo_heads, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    q_indptr_cpu = (
        paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32") * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        paddle.randn(shape=[total_num_pages, 2, num_kv_heads, page_size, head_dim])
        .to(0)
        .astype(dtype="float16")
        if kv_layout == "HND"
        else paddle.randn(shape=[total_num_pages, 2, page_size, num_kv_heads, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    kv_indptr_cpu = (
        paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32")
        * num_pages_per_seq
    )
    kv_indices_cpu = paddle.arange(start=0, end=total_num_pages).astype(dtype="int32")
    kv_last_page_len_cpu = paddle.full(
        shape=(batch_size,), fill_value=(kv_len - 1) % page_size + 1, dtype="int32"
    )
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8").to(0)
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        prefix_len_ptr=paddle.to_tensor(data=prefix_len_ptr)
>>>>>>        .to(dtype=torch.uint32)
        .to(0),
        token_pos_in_items_ptr=paddle.to_tensor(data=token_pos_in_items_ptr)
>>>>>>        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=paddle.to_tensor(data=token_pos_in_items_len)
>>>>>>        .to(dtype=torch.uint32)
        .to(0),
        max_item_len_ptr=paddle.to_tensor(data=max_item_len_ptr)
>>>>>>        .to(dtype=torch.uint16)
        .to(0),
    )
    if return_lse:
        o, _ = wrapper.run_return_lse(q, kv_data)
    else:
        o = wrapper.run(q, kv_data)
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = paddle.concat(
            x=[
                kv_data[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .transpose(perm=perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data[kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else kv_data[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            axis=0,
        )
        vi = paddle.concat(
            x=[
                kv_data[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .transpose(perm=perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data[kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else kv_data[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            axis=0,
        )

        def create_2D_multi_item_mask_dense(
            is_delimiter, sliding_window_size=-1, prefix_cache_len=None
        ):
            delimiter_idx = is_delimiter.nonzero(as_tuple=True)[0]
            if len(delimiter_idx) == 0:
                return None
            else:
                first_delimiter_pos = delimiter_idx[0]
            seq_len = len(is_delimiter)
            pos = paddle.arange(end=seq_len)
            group_ids = paddle.cumsum(x=is_delimiter, axis=0)
            within_group_causal = (
                group_ids.unsqueeze(axis=1) == group_ids.unsqueeze(axis=0)
            ) & (pos.unsqueeze(axis=0) <= pos.unsqueeze(axis=1))
            attention_mask = (
                (
                    within_group_causal
                    | (pos >= first_delimiter_pos).unsqueeze(axis=1)
                    & (pos < first_delimiter_pos).unsqueeze(axis=0)
                )
                & ~is_delimiter.unsqueeze(axis=0)
                & ~is_delimiter.unsqueeze(axis=1)
            )
            if sliding_window_size > 0 and sliding_window_size < len(is_delimiter):
                group_size = paddle.sum(
                    x=within_group_causal & ~is_delimiter.unsqueeze(axis=0), axis=1
                )
                prefix_window = paddle.where(
                    condition=pos >= first_delimiter_pos,
                    x=sliding_window_size - group_size,
                    y=paddle.where(
                        condition=pos < sliding_window_size,
                        x=first_delimiter_pos,
                        y=sliding_window_size,
                    ),
                )
                prefix_start = first_delimiter_pos - prefix_window.unsqueeze(axis=1)
                attention_mask = attention_mask & (pos >= prefix_start)
            if prefix_cache_len:
                patch = paddle.ones(shape=[seq_len, prefix_cache_len], dtype="bool")
                attention_mask = paddle.concat(x=[patch, attention_mask], axis=1)
            return attention_mask.unsqueeze(axis=0).reshape(-1)

        custom_mask = create_2D_multi_item_mask_dense(
            is_delimiter=paddle.to_tensor(data=token_pos_in_items_ptr).to(0) == 0,
            sliding_window_size=-1,
            prefix_cache_len=prefix_len_ptr,
        )
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
        )
        o_i_np = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].cpu().numpy()
        o_ref_i_np = o_ref_i.cpu().numpy()
        numpy.testing.assert_allclose(o_i_np, o_ref_i_np, rtol=0.001, atol=0.001)


if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_tuple_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "HND", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 8, 8, 128, True, "HND", "NONE", False, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache_custom_mask(
        1, 137, 137, 1, 8, 8, 128, "HND", "NONE", 0.0, False, True
    )
    test_batch_prefill_with_ragged_kv_cache(
        12, 54, 37, 8, 8, 128, True, "NONE", 0.0, False
    )
    test_batch_prefill_with_ragged_kv_cache_custom_mask(
        1, 137, 137, 8, 8, 128, "NONE", 0.0, False
    )
