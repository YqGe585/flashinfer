import math

import paddle
import pytest

import flashinfer


def build_causal_mask(qo_len, kv_len):
    i = paddle.arange(end=qo_len).unsqueeze(axis=1).to("gpu:0")
    j = paddle.arange(end=kv_len).unsqueeze(axis=0).to("gpu:0")
    offset = kv_len - qo_len
    mask = (j - offset > i).to("bool")
    return mask


def _repeat_kv(t: paddle.Tensor, num_groups: int) -> paddle.Tensor:
    return t.repeat_interleave(repeats=num_groups, axis=1)


def single_prefill_with_kv_cache_ref(
    q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, causal: bool = False
):
    Lq, Hq, D = tuple(q.shape)
    Lk, Hkv, _ = tuple(k.shape)
    assert (Lk, Hkv, D) == tuple(v.shape)
    assert Hq % Hkv == 0
    groups = Hq // Hkv
    k_states = _repeat_kv(k, groups)
    v_states = _repeat_kv(v, groups)
    q_t = q.transpose(perm=[1, 0, 2])
    k_t = k_states.transpose(perm=[1, 2, 0])
    v_t = v_states.transpose(perm=[1, 0, 2])
    scale = 1.0 / math.sqrt(D)
    attn_scores = paddle.bmm(x=q_t, y=k_t) * scale
    if causal:
        causal_mask = build_causal_mask(Lq, Lk)
        attn_scores = attn_scores.masked_fill(mask=causal_mask, value=float("-inf"))
    attn_weights = paddle.nn.functional.softmax(
        x=attn_scores, axis=-1, dtype="float32"
    ).to(q.dtype)
    attn_output = paddle.bmm(x=attn_weights, y=v_t)
    attn_output = attn_output.transpose(perm=[1, 0, 2]).contiguous()
    return attn_output


@pytest.mark.parametrize("kv_len", [501, 2042, 3771, 4932])
@pytest.mark.parametrize("qo_len", [37, 127, 577, 1024])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [4, 7])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
def test_sinqle_prefill_with_paged_kv_cache(
    kv_len, qo_len, num_kv_heads, num_qo_heads, head_dim, causal, pos_encoding_mode
):
    paddle.seed(seed=0)
    paddle.seed(seed=0)
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    q = paddle.randn(shape=[qo_len, num_qo_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[kv_len, num_kv_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[kv_len, num_kv_heads, head_dim], dtype="float16")
    o = flashinfer.prefill.single_prefill_with_kv_cache(
        q, k, v, causal=causal, pos_encoding_mode=pos_encoding_mode, backend="fa2"
    )
    o_ref = single_prefill_with_kv_cache_ref(q, k, v, causal=causal)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
