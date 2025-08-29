import math

import paddle
import pytest

import flashinfer


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("s_kv", [512, 8192])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True, False])
def test_cudnn_decode(
    batch_size, s_kv, page_size, num_kv_heads, num_qo_heads, is_cuda_graph_compatible
):
    seed = 0
    paddle.seed(seed=seed)
    device = "cuda:0"
    s_qo = 1
    head_dim = 128
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype="bfloat16")
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
    block_tables = paddle.to_tensor(
        data=[
            [(k + i * num_pages_per_seq) for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype="int32",
        place=device,
    )
    scale = float(1.0 / head_dim**0.5)
    actual_seq_lens_kv = paddle.randint(
        low=0, high=s_kv + 1, shape=(batch_size, 1, 1, 1), dtype="int32"
    )
    ragged_q = paddle.arange(start=0, end=batch_size + 1) * (num_qo_heads * head_dim)
    workspace_buffer_size = math.ceil(
        (
            batch_size * s_qo * num_qo_heads * head_dim * 4
            + batch_size * s_qo * num_qo_heads * 4
        )
        / (1024 * 1024)
    ) * (1024 * 1024)
    workspace_buffer_size = max(workspace_buffer_size, 128 * 1024 * 1024)
    workspace_buffer = paddle.empty(shape=workspace_buffer_size, dtype="int8")
    output = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_sequence_kv=s_kv,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        is_cuda_graph_compatible=is_cuda_graph_compatible,
        batch_offsets_q=ragged_q,
        batch_offsets_o=ragged_q,
    )
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    kv_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(
                    x=(actual_seq_lens_kv_device.flatten() + page_size - 1)
                    // page_size,
                    axis=0,
                ),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indices = paddle.zeros(shape=kv_indptr[-1], dtype="int32")
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = paddle.arange(
            start=i * num_pages_per_seq,
            end=i * num_pages_per_seq + (end_idx - start_idx),
        )
    kv_last_page_len = (
        paddle.where(
            condition=actual_seq_lens_kv_device.flatten() % page_size == 0,
            x=paddle.full(shape=(batch_size,), fill_value=page_size),
            y=actual_seq_lens_kv_device.flatten() % page_size,
        )
        .astype(dtype="int32")
        .to(device)
    )
    workspace_buffer_ref = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_ref, "HND")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="bfloat16",
    )
    output_ref = wrapper.run(q, kv_cache)
    assert paddle.allclose(x=output, y=output_ref, rtol=0.01, atol=0.01).item(), ""
