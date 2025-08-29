import sys

sys.path.append("/home/flashinfer")
import paddle
import pytest
from paddle_utils import *

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")
    seed = 1
    paddle.seed(seed=seed)
    device = "cuda:0"
    actual_seq_lens_q = paddle.randint(
        low=1, high=s_qo + 1, shape=(batch_size, 1, 1, 1), dtype="int32"
    )
    actual_seq_lens_kv = paddle.randint(
        low=s_qo, high=s_kv + 1, shape=(batch_size, 1, 1, 1), dtype="int32"
    )
    cumsum_s_qo = paddle.sum(x=actual_seq_lens_q)
    q = paddle.randn(shape=[cumsum_s_qo, num_qo_heads, head_dim], dtype="bfloat16")
    q_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_q.view(-1), axis=0)
            * head_dim
            * num_qo_heads,
        ]
    ).astype(dtype="int32")
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache_shape = total_num_pages, 2, num_kv_heads, page_size, head_dim
    kv_cache = paddle.randn(shape=kv_cache_shape, dtype="bfloat16").to(device)
    kv_cache = kv_cache.as_strided(
        shape=tuple(kv_cache.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]
    v_cache = v_cache_view.as_strided(
        shape=tuple(v_cache_view.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache = k_cache_view.as_strided(
        shape=tuple(k_cache_view.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    kv_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(
                x=(actual_seq_lens_kv.flatten() + page_size - 1) // page_size, axis=0
            ),
        ]
    ).astype(dtype="int32")
    kv_indices = paddle.zeros(shape=kv_indptr[-1], dtype="int32")
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = paddle.arange(
            start=i * num_pages_per_seq,
            end=i * num_pages_per_seq + (end_idx - start_idx),
        )
    kv_last_page_len = paddle.where(
        condition=actual_seq_lens_kv.flatten() % page_size == 0,
        x=paddle.full(shape=(batch_size,), fill_value=page_size),
        y=actual_seq_lens_kv.flatten() % page_size,
    ).astype(dtype="int32")
    block_tables = paddle.to_tensor(
        data=[
            [(k + i * num_pages_per_seq) for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype="int32",
        place=device,
    )
    scale = float(1.0 / head_dim**0.5)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type="bfloat16",
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )
    output = wrapper_cudnn.run(q, (k_cache, v_cache))
    qo_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_q.view(-1), axis=0),
        ]
    ).astype(dtype="int32")
    workspace_buffer_ref = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND"
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
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type="bfloat16",
    )
    output_ref = wrapper.run(q, kv_cache)
    assert paddle.allclose(x=output, y=output_ref, atol=0.002, rtol=0.01).item(), ""
