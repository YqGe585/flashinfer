import sys

sys.path.append("/home/flashinfer_paddle")
import math

import paddle
import pytest
from paddle_utils import *

import flashinfer

global_workspace_buffer = None


@pytest.mark.parametrize("batch_size", [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024])
@pytest.mark.parametrize("scale", [1.0, 0.5])
>>>>>>@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, "bfloat16"])
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("q_len_per_request", [1, 2])
@pytest.mark.parametrize("dynamic_scale", [False])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_decode_mla(
    batch_size: int,
    scale: float,
    dtype: paddle.dtype,
    page_size: int,
    q_len_per_request: int,
    dynamic_scale: bool,
    enable_pdl: bool,
):
>>>>>>    if dynamic_scale and dtype != torch.float8_e4m3fn:
        pytest.skip("Dynamic scale is not supported for non-fp8 dtype")
    paddle.seed(seed=42)
    device = "cuda:0"
    MAX_SEQ_LEN = 1024
    num_q_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    query = paddle.randn(
        shape=[
            batch_size,
            q_len_per_request,
            num_q_heads,
            kv_lora_rank + qk_rope_head_dim,
        ]
    ).to(dtype)
    num_tokens = MAX_SEQ_LEN * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    seq_lens = [
        paddle.randint(low=1, high=MAX_SEQ_LEN, shape=(1,)).item()
        for _ in range(batch_size)
    ]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens_tensor = paddle.to_tensor(data=seq_lens, dtype="int32", place=device)
    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq._max().item()
    total_blocks_needed = sum(blocks_per_seq)
    all_block_ids = paddle.randperm(n=total_blocks_needed)
    block_id = 0
    block_tables = paddle.zeros(
        shape=(batch_size, max_num_blocks_per_seq), dtype="int32"
    )
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = blocks_per_seq[i]
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed
    kv_cache = paddle.randn(
        shape=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim)
    ).to(dtype)
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = paddle.zeros(shape=128 * 1024 * 1024, dtype="int8")
    workspace_buffer = global_workspace_buffer
    bmm1_log2_scale_tensor = (
        paddle.to_tensor(
            data=[scale / ((128 + 64) ** 0.5 * math.log2(math.e))],
            dtype="float32",
            place=device,
        )
        if dynamic_scale
        else None
    )
    bmm2_scale_tensor = (
        paddle.to_tensor(data=[1.0], dtype="float32", place=device)
        if dynamic_scale
        else None
    )
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(axis=1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / (128 + 64) ** 0.5,
        bmm2_scale=1.0,
        bmm1_scale_log2_tensor=bmm1_log2_scale_tensor,
        bmm2_scale_tensor=bmm2_scale_tensor,
        enable_pdl=enable_pdl,
    )
    sm_scale = scale / (128 + 64) ** 0.5
    workspace_buffer_ref = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer_ref, backend="fa2"
    )
>>>>>>    if dtype == torch.float8_e4m3fn:
        query = query.to("bfloat16")
        kv_cache = kv_cache.to("bfloat16")
    q_indptr = (
        paddle.arange(start=0, end=batch_size + 1, dtype="int32") * q_len_per_request
    )
    kv_indptr = paddle.zeros_like(x=q_indptr)
    kv_indptr[1:] = paddle.cumsum(x=blocks_per_seq, axis=0)
    kv_indices = all_block_ids.astype(dtype="int32")
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_tensor,
        num_q_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        True,
        sm_scale,
        query.dtype,
        kv_cache.dtype,
    )
    q_nope = query[..., :kv_lora_rank].view(
        batch_size * q_len_per_request, num_q_heads, kv_lora_rank
    )
    q_pe = query[..., kv_lora_rank:].view(
        batch_size * q_len_per_request, num_q_heads, qk_rope_head_dim
    )
    ckv = kv_cache[..., :kv_lora_rank]
    kpe = kv_cache[..., kv_lora_rank:]
    o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    assert not paddle.isnan(x=o_ref).astype("bool").any(), "o_ref is nan"
    assert not paddle.isnan(x=output).astype("bool").any(), "output is nan"
>>>>>>    if dtype == torch.float8_e4m3fn:
        try:
            assert paddle.allclose(
                x=output,
                y=o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
                rtol=0.1,
                atol=0.1,
            ).item(), ""
        except AssertionError as e:
            print("output:", output)
            print("o_ref:", o_ref)
            raise e
    else:
        try:
            assert paddle.allclose(
                x=output,
                y=o_ref.view(batch_size, q_len_per_request, num_q_heads, -1),
                rtol=0.01,
                atol=0.01,
            ).item(), ""
        except AssertionError as e:
            print("output:", output)
            print("o_ref:", o_ref)
            raise e
