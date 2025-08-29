import sys

sys.path.append("/home/flashinfer_paddle")
import math

import paddle
import pytest
from conftest import VARLEN_INDPTR_PARAMS
from paddle_utils import *

import flashinfer
from flashinfer.utils import is_sm100a_supported


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


def attention_varlen_ref(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    qo_indptr: paddle.Tensor,
    kv_indptr: paddle.Tensor,
    causal: bool,
    sm_scale: float,
) -> paddle.Tensor:
    batch_size = tuple(qo_indptr.shape)[0] - 1
    nnz_qo = qo_indptr[-1].item()
    o = paddle.empty(
        shape=[nnz_qo, *tuple(q.shape)[1:-1], tuple(v.shape)[-1]], dtype=q.dtype
    )
    lse = paddle.empty(shape=[nnz_qo, tuple(q.shape)[1]], dtype="float32")
    for i in range(batch_size):
        o_i, lse_i = attention_ref(
            1,
            q[qo_indptr[i] : qo_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal,
            sm_scale,
        )
        lse_i = lse_i.flatten(start_axis=0, stop_axis=1)
        o[qo_indptr[i] : qo_indptr[i + 1]] = o_i
        lse[qo_indptr[i] : qo_indptr[i + 1]] = lse_i
    return o, lse


@pytest.mark.parametrize("batch_size", [1, 2, 3, 9, 17])
@pytest.mark.parametrize("qo_len", [1, 17, 177, 377, 977])
@pytest.mark.parametrize("kv_len", [1, 17, 544, 977, 1999])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0, 1.0 / math.sqrt(192), 1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_blackwell_cutlass_fmha(
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    causal,
    dtype,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    if not is_sm100a_supported(device2str("cuda")):
        pytest.skip("SM100A is not supported on this device")
    paddle.seed(seed=42)
    q = paddle.randn(
        shape=[batch_size * qo_len, num_qo_heads, head_dim_qk], dtype=dtype
    )
    qo_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * qo_len
    k = paddle.randn(
        shape=[batch_size * kv_len, num_kv_heads, head_dim_qk], dtype=dtype
    )
    v = paddle.randn(
        shape=[batch_size * kv_len, num_kv_heads, head_dim_vo], dtype=dtype
    )
    kv_indptr = paddle.arange(start=0, end=batch_size + 1, dtype="int32") * kv_len
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="uint8"),
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)
    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = paddle.repeat_interleave(x=k, repeats=gqa_group_ratio, axis=1)
    v_repeated = paddle.repeat_interleave(x=v, repeats=gqa_group_ratio, axis=1)
    o_ref, lse_ref = attention_ref(
        batch_size, q, k_repeated, v_repeated, causal, sm_scale
    )
    lse_ref = lse_ref.flatten(start_axis=0, stop_axis=1)
    if dtype == "float16":
        assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    else:
        assert paddle.allclose(x=o, y=o_ref, rtol=0.01, atol=0.01).item(), ""
    assert paddle.allclose(x=lse, y=lse_ref, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("indptr", VARLEN_INDPTR_PARAMS)
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0 / math.sqrt(128)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", ["bfloat16"])
def test_blackwell_cutlass_varlen(
    indptr,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    causal,
    dtype,
):
    if not is_sm100a_supported(device2str("cuda")):
        pytest.skip("SM100A is not supported on this device")
    paddle.seed(seed=42)
    qkv = paddle.randn(
        shape=[
            indptr[-1],
            num_qo_heads * head_dim_qk
            + num_kv_heads * head_dim_qk
            + num_kv_heads * head_dim_vo,
        ],
        dtype=dtype,
    )
    q = qkv[:, : num_qo_heads * head_dim_qk].view(indptr[-1], num_qo_heads, head_dim_qk)
    k = qkv[
        :,
        num_qo_heads * head_dim_qk : num_qo_heads * head_dim_qk
        + num_kv_heads * head_dim_qk,
    ].view(indptr[-1], num_kv_heads, head_dim_qk)
    v = qkv[:, num_qo_heads * head_dim_qk + num_kv_heads * head_dim_qk :].view(
        indptr[-1], num_kv_heads, head_dim_vo
    )
    qo_indptr = paddle.to_tensor(data=indptr, dtype="int32", place="gpu")
    kv_indptr = qo_indptr
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="uint8"),
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)
    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = paddle.repeat_interleave(x=k, repeats=gqa_group_ratio, axis=1)
    v_repeated = paddle.repeat_interleave(x=v, repeats=gqa_group_ratio, axis=1)
    o_ref, lse_ref = attention_varlen_ref(
        q, k_repeated, v_repeated, qo_indptr, kv_indptr, causal, sm_scale
    )
    if dtype == "float16":
        assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    else:
        assert paddle.allclose(x=o, y=o_ref, rtol=0.01, atol=0.01).item(), ""
    assert paddle.allclose(x=lse, y=lse_ref, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("qo_indptr_list", [[0, 10, 20, 30, 40, 50, 60, 100]])
@pytest.mark.parametrize("kv_indptr_list", [[0, 50, 50, 50, 50, 50, 50, 50]])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim_qk", [192, 128])
@pytest.mark.parametrize("head_dim_vo", [128])
@pytest.mark.parametrize("sm_scale", [1.0 / math.sqrt(128)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_blackwell_cutlass_qo_kv_varlen(
    qo_indptr_list,
    kv_indptr_list,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    sm_scale,
    dtype,
):
    causal = False
    if not is_sm100a_supported(device2str("cuda")):
        pytest.skip("SM100A is not supported on this device")
    paddle.seed(seed=42)
    q = paddle.randn(shape=[qo_indptr_list[-1], num_qo_heads, head_dim_qk], dtype=dtype)
    k = paddle.randn(shape=[kv_indptr_list[-1], num_kv_heads, head_dim_qk], dtype=dtype)
    v = paddle.randn(shape=[kv_indptr_list[-1], num_kv_heads, head_dim_vo], dtype=dtype)
    qo_indptr = paddle.to_tensor(data=qo_indptr_list, dtype="int32", place="gpu")
    kv_indptr = paddle.to_tensor(data=kv_indptr_list, dtype="int32", place="gpu")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="uint8"),
        kv_layout="NHD",
        backend="cutlass",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o, lse = wrapper.run(q, k, v, return_lse=True)
    gqa_group_ratio = num_qo_heads // num_kv_heads
    k_repeated = paddle.repeat_interleave(x=k, repeats=gqa_group_ratio, axis=1)
    v_repeated = paddle.repeat_interleave(x=v, repeats=gqa_group_ratio, axis=1)
    o_ref, lse_ref = attention_varlen_ref(
        q, k_repeated, v_repeated, qo_indptr, kv_indptr, causal, sm_scale
    )
    if dtype == "float16":
        assert paddle.allclose(
            x=o[10:60], y=o_ref[10:60], rtol=0.001, atol=0.001
        ).item(), ""
    else:
        assert paddle.allclose(
            x=o[10:60], y=o_ref[10:60], rtol=0.01, atol=0.01
        ).item(), ""
    assert paddle.allclose(x=lse, y=lse_ref, rtol=0.001, atol=0.001).item(), ""


if __name__ == "__main__":
    test_blackwell_cutlass_fmha(9, 377, 977, 1, 1, 192, 128, 1, False, "bfloat16")
    test_blackwell_cutlass_varlen(
        [0, 1274, 2568, 3915, 5194, 6498, 7839, 8192],
        32,
        4,
        128,
        128,
        1,
        True,
        "bfloat16",
    )
    test_blackwell_cutlass_qo_kv_varlen(
        [0, 10, 20, 30, 40, 50, 60, 100],
        [0, 50, 50, 50, 50, 50, 50, 50],
        32,
        8,
        128,
        128,
        1,
        "bfloat16",
    )
