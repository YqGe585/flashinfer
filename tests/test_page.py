import paddle
import pytest

import flashinfer


@pytest.mark.parametrize("contiguous", [True, False])
def test_append_paged_kv_cache(contiguous):
    nnz_kv = 100
    num_kv_heads = 32
    head_dim = 128
    if contiguous:
        k_append = (
            paddle.randn(shape=[nnz_kv, num_kv_heads, head_dim])
            .astype(dtype="float16")
            .to(0)
        )
        v_append = (
            paddle.randn(shape=[nnz_kv, num_kv_heads, head_dim])
            .astype(dtype="float16")
            .to(0)
        )
    else:
        kv_append = (
            paddle.randn(shape=[nnz_kv, 2, num_kv_heads, head_dim])
            .astype(dtype="float16")
            .to(0)
        )
        k_append = kv_append[:, 0]
        v_append = kv_append[:, 1]
    kv_append_length = paddle.to_tensor(
        data=[45, 8, 25, 22], dtype="int32", place="gpu:0"
    )
    kv_append_indptr = paddle.concat(
        x=[
            paddle.zeros(shape=[1]).astype(dtype="int32").to(0),
            paddle.cumsum(x=kv_append_length, axis=0),
        ]
    ).astype(dtype="int32")
    max_num_pages = 1000
    page_size = 16
    paged_kv_cache = (
        paddle.randn(shape=[max_num_pages, 2, page_size, num_kv_heads, head_dim])
        .astype(dtype="float16")
        .to(0)
    )
    num_pages_per_req = paddle.to_tensor(
        data=[3, 1, 2, 2], dtype="int32", place="gpu:0"
    )
    kv_page_indptr = paddle.concat(
        x=[
            paddle.zeros(shape=[1]).astype(dtype="int32").to(0),
            paddle.cumsum(x=num_pages_per_req, axis=0),
        ]
    ).astype(dtype="int32")
    kv_page_indices = paddle.arange(dtype="int32", end=8)
    kv_last_page_len = paddle.to_tensor(
        data=[13, 8, 9, 6], dtype="int32", place="gpu:0"
    )
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )
    flashinfer.append_paged_kv_cache(
        k_append,
        v_append,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )
