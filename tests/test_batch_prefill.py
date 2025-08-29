import paddle
import pytest

from flashinfer import BatchPrefillWithPagedKVCacheWrapper


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_kv_scale_forwarding_effect(dtype):
    paddle.seed(seed=42)
    H_QO, H_KV, N_CTX, HEAD_DIM, PAGE_SIZE = 1, 1, 8, 64, 16
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE
    k_cache = paddle.randn(
        shape=[max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM], dtype=dtype
    )
    v_cache = paddle.randn(
        shape=[max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM], dtype=dtype
    )
    paged_kv_cache = k_cache, v_cache
    q = paddle.randn(shape=[N_CTX, H_QO, HEAD_DIM], dtype=dtype)
    qo_indptr = paddle.to_tensor(data=[0, N_CTX], dtype="int32", place="gpu")
    paged_kv_indptr = paddle.to_tensor(
        data=[0, max_num_pages], dtype="int32", place="gpu"
    )
    paged_kv_indices = paddle.arange(dtype="int32", end=max_num_pages)
    paged_kv_last_page_len = paddle.to_tensor(
        data=[N_CTX % PAGE_SIZE or PAGE_SIZE], dtype="int32", place="gpu"
    )
    workspace_buffer = paddle.empty(shape=16 * 1024 * 1024, dtype="uint8")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=0.1, v_scale=0.1)
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=2.0, v_scale=2.0)
    assert not paddle.allclose(
        x=out1, y=out2, atol=0.001
    ).item(), "Output should change when k_scale/v_scale values are different. This may indicate that the arguments are not passed correctly."


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_kv_scale_forwarding_math_property(dtype: paddle.dtype):
    paddle.seed(seed=0)
    N_CTX, PAGE_SIZE = 128, 16
    H_QO, H_KV, HEAD_DIM = 1, 1, 64
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE
    k_cache = paddle.randn(
        shape=[max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM], dtype=dtype
    )
    v_cache = paddle.randn(shape=k_cache.shape, dtype=k_cache.dtype)
    paged_kv_cache = k_cache, v_cache
    q = paddle.randn(shape=[N_CTX, H_QO, HEAD_DIM], dtype=dtype)
    qo_indptr = paddle.to_tensor(data=[0, N_CTX], dtype="int32", place="gpu")
    paged_kv_indptr = paddle.to_tensor(
        data=[0, max_num_pages], dtype="int32", place="gpu"
    )
    paged_kv_indices = paddle.arange(dtype="int32", end=max_num_pages)
    paged_kv_last_page_len = paddle.to_tensor(
        data=[N_CTX % PAGE_SIZE or PAGE_SIZE], dtype="int32", place="gpu"
    )
    workspace = paddle.empty(shape=16 * 1024 * 1024, dtype="uint8")
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)
    wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    k_scale = paddle.to_tensor(data=0.5, dtype="float32", place="gpu")
    v_scale = paddle.to_tensor(data=2.0, dtype="float32", place="gpu")
    out1, _ = wrapper.forward_return_lse(q, paged_kv_cache, k_scale=k_scale)
    out1_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    assert paddle.allclose(x=out1, y=out1_ref, rtol=0.01, atol=0.001).item(), ""
    out2, _ = wrapper.forward_return_lse(q, paged_kv_cache, v_scale=v_scale)
    out2_ref, _ = wrapper.forward_return_lse(q, paged_kv_cache)
    assert paddle.allclose(
        x=out2, y=out2_ref * v_scale, rtol=0.01, atol=0.001
    ).item(), ""
    out3, _ = wrapper.forward_return_lse(
        q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale
    )
    out3_ref, _ = wrapper.forward_return_lse(q * k_scale, paged_kv_cache)
    assert paddle.allclose(
        x=out3, y=out3_ref * v_scale, rtol=0.01, atol=0.001
    ).item(), ""
