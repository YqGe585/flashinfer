import sys

sys.path.append("/home/flashinfer")
import paddle
import pytest
from paddle_utils import *

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [32, 64, 87])
@pytest.mark.parametrize("s_kv", [32, 64, 87])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1, 16])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_prefill_deepseek(
    batch_size, s_qo, s_kv, num_kv_heads, num_qo_heads, causal
):
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")
    head_dim_qk = 192
    head_dim_vo = 128
    return_lse = True
    seed = 0
    paddle.seed(seed=seed)
    device = "cuda:0"
    actual_seq_lens_q = paddle.randint(
        low=1, high=s_qo + 1, shape=(batch_size, 1, 1, 1), dtype="int32"
    )
    actual_seq_lens_kv = paddle.randint(
        low=s_qo, high=s_kv + 1, shape=(batch_size, 1, 1, 1), dtype="int32"
    )
    cumsum_s_qo = paddle.sum(x=actual_seq_lens_q)
    q = paddle.randn(shape=[cumsum_s_qo, num_qo_heads, head_dim_qk], dtype="bfloat16")
    q_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_q.view(-1), axis=0)
            * head_dim_qk
            * num_qo_heads,
        ]
    ).astype(dtype="int32")
    k_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_kv.view(-1), axis=0)
            * head_dim_qk
            * num_kv_heads,
        ]
    ).astype(dtype="int32")
    v_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_kv.view(-1), axis=0)
            * head_dim_vo
            * num_kv_heads,
        ]
    ).astype(dtype="int32")
    o_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_q.view(-1), axis=0)
            * head_dim_vo
            * num_qo_heads,
        ]
    ).astype(dtype="int32")
    batch_offsets_stats = paddle.concat(
        x=[
            paddle.zeros(shape=[1], dtype=actual_seq_lens_q.dtype),
            paddle.cumsum(x=actual_seq_lens_q.flatten(), axis=0) * num_qo_heads,
        ]
    ).cuda()
    k_cache = paddle.randn(
        shape=[batch_size * s_kv, num_kv_heads, head_dim_qk], dtype="bfloat16"
    )
    v_cache = paddle.randn(
        shape=[batch_size * s_kv, num_kv_heads, head_dim_vo], dtype="bfloat16"
    )
    scale = float(1.0 / head_dim_qk**0.5)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    output, lse = flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        causal=causal,
        return_lse=return_lse,
        batch_offsets_q=q_indptr,
        batch_offsets_k=k_indptr,
        batch_offsets_v=v_indptr,
        batch_offsets_o=o_indptr,
        batch_offsets_stats=batch_offsets_stats,
        is_cuda_graph_compatible=True,
    )
    qo_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_q.view(-1), axis=0),
        ]
    ).astype(dtype="int32")
    kv_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_kv.view(-1), axis=0),
        ]
    ).astype(dtype="int32")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="uint8"), kv_layout="NHD"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
        sm_scale=scale,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
    )
    output_ref, lse_ref = wrapper.run(q, k_cache, v_cache, return_lse=True)
    assert paddle.allclose(x=output, y=output_ref, atol=0.01, rtol=0.01).item(), ""
