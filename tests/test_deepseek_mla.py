import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

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
import math

import pytest
from conftest import clear_cuda_cache

import flashinfer
from flashinfer.jit import build_jit_specs
from flashinfer.jit.attention import (gen_batch_mla_module,
                                      gen_batch_prefill_module,
                                      gen_single_prefill_module)
from flashinfer.utils import is_sm90a_supported, is_sm100a_supported


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    try:
        modules = []
        for backend in ["fa2", "fa3"]:
            if backend == "fa3" and not is_sm90a_supported(device2str("cuda")):
                continue
            modules.append(
                gen_single_prefill_module(
                    backend,
                    "float16",
                    "float16",
                    "float16",
                    192,
                    128,
                    0,
                    False,
                    False,
                    False,
                )
            )
        for backend in ["fa2", "fa3"]:
            if backend == "fa3" and not is_sm90a_supported(device2str("cuda")):
                continue
            modules.append(
                gen_batch_prefill_module(
                    backend,
                    "float16",
                    "float16",
                    "float16",
                    "int32",
                    192,
                    128,
                    0,
                    False,
                    False,
                    False,
                )
            )
        for backend in ["fa2", "fa3"]:
            if backend == "fa3" and not is_sm90a_supported(device2str("cuda")):
                continue
            modules.append(
                gen_batch_mla_module(
                    backend, "float16", "float16", "float16", "int32", 512, 64, False
                )
            )
        build_jit_specs(modules, verbose=False)
    except Exception as e:
        pytest.exit(str(e))
    finally:
        yield


def attention_ref(
    batch_size,
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    causal: bool,
    sm_scale: float,
) -> paddle.Tensor:
    qo_len = tuple(q.shape)[0] // batch_size
    kv_len = tuple(k.shape)[0] // batch_size
    num_qo_heads = tuple(q.shape)[1]
    head_dim_qk = tuple(q.shape)[2]
    head_dim_vo = tuple(v.shape)[2]
    logits = (
        paddle.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).astype(
                dtype="float32"
            ),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).astype(
                dtype="float32"
            ),
        )
        * sm_scale
    )
    if causal:
        mask = paddle.arange(start=kv_len - qo_len, end=kv_len).unsqueeze(
            axis=1
        ) >= paddle.arange(start=0, end=kv_len).unsqueeze(axis=0)
    else:
        mask = paddle.ones(shape=[qo_len, kv_len])
    logits = logits.masked_fill(
        mask=mask.unsqueeze(axis=0).unsqueeze(axis=0) == 0, value=float("-inf")
    )
    lse_ref = paddle.logsumexp(x=logits, axis=-1).transpose(
        perm=dim2perm(paddle.logsumexp(x=logits, axis=-1).ndim, -1, -2)
    )
    p = paddle.nn.functional.softmax(x=logits, axis=-1)
    o_ref = (
        paddle.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).astype(
                dtype="float32"
            ),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )
    return o_ref, lse_ref * math.log2(math.e)


@pytest.mark.parametrize("kv_len", [5532, 7563])
@pytest.mark.parametrize("qo_len", [1832, 3928])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_single_prefill_with_kv_cache(
    kv_len, qo_len, num_heads, causal, backend, dtype
):
    device = device2str("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    paddle.seed(seed=42)
    head_dim_qk = 192
    head_dim_vo = 128
    q = paddle.randn(shape=[qo_len, num_heads, head_dim_qk], dtype=dtype)
    k = paddle.randn(shape=[kv_len, num_heads, head_dim_qk], dtype=dtype)
    v = paddle.randn(shape=[kv_len, num_heads, head_dim_vo], dtype=dtype)
    o, lse = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend=backend, return_lse=True
    )
    sm_scale = 1.0 / head_dim_qk**0.5
    o_ref, lse_ref = attention_ref(1, q, k, v, causal, sm_scale)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(
        x=lse, y=lse_ref.squeeze(axis=0), rtol=0.001, atol=0.001
    ).item(), ""


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [544, 977])
@pytest.mark.parametrize("qo_len", [377, 177])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size, kv_len, qo_len, num_heads, causal, backend, dtype
):
    device = device2str("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    paddle.seed(seed=42)
    kv_layout = "NHD"
    head_dim_qk = 192
    head_dim_vo = 128
    q = paddle.randn(shape=[batch_size * qo_len, num_heads, head_dim_qk], dtype=dtype)
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    k = paddle.zeros(shape=[batch_size * kv_len, num_heads, head_dim_qk], dtype=dtype)
    v = paddle.zeros(shape=[batch_size * kv_len, num_heads, head_dim_vo], dtype=dtype)
    kv_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * kv_len
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend=backend
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
    )
    o, lse = wrapper.run_return_lse(q, k, v)
    sm_scale = 1.0 / head_dim_qk**0.5
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(start_axis=0, stop_axis=1)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=lse, y=lse_ref, rtol=0.001, atol=0.001).item(), ""
    o_buffer = paddle.empty_like(x=o)
    lse_buffer = paddle.empty_like(x=lse)
    wrapper.run(q, k, v, out=o_buffer, lse=lse_buffer)
    assert paddle.allclose(x=o, y=o_buffer, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=lse, y=lse_buffer, rtol=0.001, atol=0.001).item(), ""


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = tuple(ckv.shape)
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = tuple(kpe.shape)
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        paddle.concat(x=[ckv, kpe], axis=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(repeats=num_heads, axis=1)
    )
    v = ckv.repeat_interleave(repeats=num_heads, axis=1)
    return k, v


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7])
@pytest.mark.parametrize("kv_len_0", [0, 1, 3, 11])
@pytest.mark.parametrize("kv_len_1", [17, 33, 79, 114])
@pytest.mark.parametrize("kv_len_2", [514, 2743, 8736])
@pytest.mark.parametrize("qo_len", [1, 3, 5, 7, 9, 11, 13, 15, 17])
@pytest.mark.parametrize("num_heads", [16, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_batch_mla_varlen_page_attention(
    batch_size,
    kv_len_0,
    kv_len_1,
    kv_len_2,
    qo_len,
    num_heads,
    causal,
    page_size,
    backend,
    dtype,
):
    device = device2str("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > min(kv_len_0, kv_len_1, kv_len_2):
        pytest.skip("qo_len > kv_len not supported for causal attention")
    num_different_kv_len = 3
    kv_lens = paddle.to_tensor(data=[kv_len_0, kv_len_1, kv_len_2], dtype="int32")
    paddle.seed(seed=42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = paddle.randn(
        shape=[num_different_kv_len * batch_size * qo_len, num_heads, head_dim_ckv],
        dtype=dtype,
    )
    q_pe = paddle.randn(
        shape=[num_different_kv_len * batch_size * qo_len, num_heads, head_dim_kpe],
        dtype=dtype,
    )
    pages_nums = paddle.to_tensor(
        data=[math.ceil(kv_len / page_size) for kv_len in kv_lens], dtype="int32"
    )
    pages_nums_indptr = paddle.zeros(shape=num_different_kv_len + 1, dtype="int32")
    pages_nums_indptr[1:] = pages_nums.cumsum(axis=0)
    pages_nums_sum = pages_nums_indptr[-1]
    ckv = paddle.randn(
        shape=[batch_size * pages_nums_sum, page_size, head_dim_ckv], dtype=dtype
    )
    kpe = paddle.randn(
        shape=[batch_size * pages_nums_sum, page_size, head_dim_kpe], dtype=dtype
    )
    sm_scale = 1.0 / (128 + 64) ** 0.5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = (
        paddle.arange(start=0, end=num_different_kv_len * batch_size + 1, dtype="int32")
        * qo_len
    )
    kv_indptr = paddle.concat(
        x=[
            (
                paddle.arange(start=0, end=batch_size + 1)
                .unsqueeze(axis=-1)
                .astype(dtype="int32")
                * pages_nums_sum
                + pages_nums_indptr[i]
            )
            for i in range(num_different_kv_len)
        ],
        axis=-1,
    ).flatten()
    kv_indices = paddle.arange(start=0, end=batch_size * pages_nums_sum, dtype="int32")
    kv_lens = paddle.to_tensor(data=kv_lens, dtype="int32", place=device).tile(
        repeat_times=batch_size
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
    q_rows = (
        paddle.arange(start=0, end=num_different_kv_len * qo_len)[None, :]
        + paddle.arange(start=0, end=batch_size)[:, None]
        * num_different_kv_len
        * qo_len
    ).astype(dtype="int32")
    kv_rows = (
        paddle.arange(start=0, end=pages_nums_sum)[None, :]
        + paddle.arange(start=0, end=batch_size)[:, None] * pages_nums_sum
    ).astype(dtype="int32")
    q_rows_arr = [
        q_rows[:, i * qo_len : (i + 1) * qo_len].flatten()
        for i in range(num_different_kv_len)
    ]
    kv_rows_arr = [
        kv_rows[:, pages_nums_indptr[i] : pages_nums_indptr[i + 1]].flatten()
        for i in range(num_different_kv_len)
    ]
    for i in range(num_different_kv_len):
        k, v = generate_kv_from_cache(
            ckv[kv_rows_arr[i]], kpe[kv_rows_arr[i]], kv_lens[i], batch_size, num_heads
        )
        q = paddle.concat(x=[q_nope, q_pe], axis=-1)[q_rows_arr[i]]
        o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
        lse_ref = lse_ref.flatten(start_axis=0, stop_axis=1)
        o_i = o[q_rows_arr[i]]
        assert paddle.allclose(x=o_i, y=o_ref, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7, 157])
@pytest.mark.parametrize("kv_len", [17, 33, 75, 197])
@pytest.mark.parametrize("qo_len", [3, 7, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", ["float16"])
def test_batch_mla_oob_kv_nan(
    batch_size, kv_len, qo_len, num_heads, causal, page_size, backend, dtype
):
    device = device2str("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    paddle.seed(seed=42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = paddle.randn(
        shape=[batch_size * qo_len, num_heads, head_dim_ckv], dtype=dtype
    )
    q_pe = paddle.randn(
        shape=[batch_size * qo_len, num_heads, head_dim_kpe], dtype=dtype
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = paddle.randn(
        shape=[batch_size * pages_num, page_size, head_dim_ckv], dtype=dtype
    )
    kpe = paddle.randn(
        shape=[batch_size * pages_num, page_size, head_dim_kpe], dtype=dtype
    )
    for i in range(batch_size):
        last_page_len = kv_len - (pages_num - 1) * page_size
        ckv[(i + 1) * pages_num - 1, last_page_len:, :] = float("nan")
        kpe[(i + 1) * pages_num - 1, last_page_len:, :] = float("nan")
    sm_scale = 1.0 / (128 + 64) ** 0.5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    kv_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * pages_num
    kv_indices = paddle.arange(start=0, end=batch_size * pages_num, dtype="int32")
    kv_lens = paddle.full(shape=(batch_size,), fill_value=kv_len, dtype="int32")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = paddle.concat(x=[q_nope, q_pe], axis=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(start_axis=0, stop_axis=1)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    if kv_len != 0:
        assert paddle.allclose(x=lse, y=lse_ref, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7, 157])
@pytest.mark.parametrize("kv_len", [0, 17, 33, 96, 97, 114, 514, 1024])
@pytest.mark.parametrize("qo_len", [1, 3, 5, 7, 9, 11, 13, 15, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("use_cuda_graph", [False])
@pytest.mark.parametrize("dtype", ["float16"])
def test_batch_mla_page_attention(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    backend,
    use_cuda_graph,
    dtype,
):
    device = device2str("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    paddle.seed(seed=42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = paddle.randn(
        shape=[batch_size * qo_len, num_heads, head_dim_ckv], dtype=dtype
    )
    q_pe = paddle.randn(
        shape=[batch_size * qo_len, num_heads, head_dim_kpe], dtype=dtype
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = paddle.randn(
        shape=[batch_size * pages_num, page_size, head_dim_ckv], dtype=dtype
    )
    kpe = paddle.randn(
        shape=[batch_size * pages_num, page_size, head_dim_kpe], dtype=dtype
    )
    sm_scale = 1.0 / (128 + 64) ** 0.5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        backend=backend,
        use_cuda_graph=True,
        qo_indptr=paddle.empty(shape=batch_size + 1, dtype="int32"),
        kv_indptr=paddle.empty(shape=batch_size + 1, dtype="int32"),
        kv_indices=paddle.empty(shape=[1048576], dtype="int32"),
        kv_len_arr=paddle.empty(shape=batch_size, dtype="int32"),
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    kv_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * pages_num
    kv_indices = paddle.arange(start=0, end=batch_size * pages_num, dtype="int32")
    kv_lens = paddle.full(shape=(batch_size,), fill_value=kv_len, dtype="int32")
    if use_cuda_graph:
        kv_indptr_warmup = paddle.zeros(shape=batch_size + 1, dtype="int32")
        kv_indices_warmup = paddle.arange(start=0, end=batch_size, dtype="int32")
        kv_lens_warmup = paddle.full(shape=(batch_size,), fill_value=0, dtype="int32")
        wrapper.plan(
            q_indptr,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_lens_warmup,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )
        s = paddle.device.Stream()
        s.wait_stream(paddle.device.current_stream())
        with paddle.device.stream_guard(stream=s):
            for _ in range(3):
                o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
        paddle.device.current_stream().wait_stream(s)
>>>>>>        g = torch.cuda.CUDAGraph()
>>>>>>        with torch.cuda.graph(g):
            o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    if use_cuda_graph:
        o.fill_(value=0)
        lse.fill_(value=0)
        g.replay()
    else:
        o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = paddle.concat(x=[q_nope, q_pe], axis=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(start_axis=0, stop_axis=1)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    if kv_len != 0:
        assert paddle.allclose(x=lse, y=lse_ref, rtol=0.001, atol=0.001).item(), ""
    o_buffer = paddle.empty_like(x=o)
    lse_buffer = paddle.empty_like(x=lse)
    wrapper.run(q_nope, q_pe, ckv, kpe, out=o_buffer, lse=lse_buffer)
    assert paddle.allclose(x=o, y=o_buffer, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=lse, y=lse_buffer, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("page_size", [1, 16, 128])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cutlass_mla(batch_size, max_seq_len, page_size, dtype):
    device = device2str("cuda:0")
    clear_cuda_cache(device)
    if not is_sm100a_supported(device):
        pytest.skip("Cutlass MLA is not supported on this device")
    paddle.seed(seed=42)
    num_local_heads = 128
    head_dim_ckv = 512
    head_dim_kpe = 64
    total_page_num = 8192
    q_nope_pe = (
        paddle.randn(
            shape=[batch_size, num_local_heads, head_dim_ckv + head_dim_kpe],
            dtype=dtype,
        )
        * 100
    )
    ckv_kpe = paddle.randn(
        shape=[total_page_num, page_size, head_dim_ckv + head_dim_kpe], dtype=dtype
    )
    kv_lens = paddle.full(shape=(batch_size,), fill_value=max_seq_len, dtype="int32")
    page_num_per_batch = (max_seq_len + page_size - 1) // page_size
    assert page_num_per_batch % (128 // page_size) == 0
    page_table = paddle.randint(
        low=0,
        high=total_page_num,
        shape=(batch_size, page_num_per_batch),
        dtype="int32",
    )
    mla_ref = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="int8"), backend="fa2"
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32")
    kv_lens = paddle.full(shape=(batch_size,), fill_value=max_seq_len, dtype="int32")
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * page_num_per_batch
    )
    kv_indices = page_table.flatten()
    q_nope = q_nope_pe[..., :head_dim_ckv]
    q_pe = q_nope_pe[..., head_dim_ckv:]
    ckv = ckv_kpe[..., :head_dim_ckv]
    kpe = ckv_kpe[..., head_dim_ckv:]
    sm_scale = 1.0 / (128 + 64) ** 0.5
    mla_ref.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_local_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o_ref = mla_ref.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    mla_ans = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="int8"), backend="cutlass"
    )
    o_ans = mla_ans.run(q_nope, q_pe, ckv, kpe, kv_len=kv_lens, page_table=page_table)
    assert paddle.allclose(x=o_ans, y=o_ref, rtol=0.01, atol=0.01).item(), ""


if __name__ == "__main__":
    test_batch_mla_varlen_page_attention(
        1, 65, 65, 65, 1, 128, True, 64, "fa2", "float16"
    )
