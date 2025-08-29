import sys

sys.path.append("/home/flashinfer")
import math

import paddle
import pytest
from paddle_utils import *
from utils_fp4 import cast_from_fp4, recover_swizzled_scales, ref_fp4_quant

import flashinfer
from flashinfer.utils import FP4Tensor, ceil_div, round_up

DTYPE_MAP = {
    "fp16": "float16",
    "bf16": "bfloat16",
>>>>>>    "fp8": paddle.float8_e4m3fn,
    "nvfp4": "nvfp4",
}
GPU_DEVICE = "cuda:0"
global_workspace_buffer = None


def flip_coin(*args, **kwargs):
    param_tuple = args + tuple(sorted(kwargs.items()))
    hash_value = hash(param_tuple)
    return hash_value % 2 == 0


>>>>>>def to_float8(x, dtype=paddle.float8_e4m3fn):
    finfo = paddle.finfo(dtype=dtype)
    min_val, max_val = tuple(
        [
            paddle.amin(x, axis=None, keepdim=False),
            paddle.max(x, axis=None, keepdim=False),
        ]
    )
    amax = paddle.maximum(x=min_val.abs(), y=max_val.abs()).clip(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clip(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.astype(dtype="float32").reciprocal()


def generate_seq_lens(batch_size, max_q_len, max_in_kv_len):
    q_lens = paddle.randint(
        low=1, high=max_q_len + 1, shape=(batch_size,), dtype="int32"
    )
    q_lens[-1] = max_q_len
    in_kv_lens = paddle.randint(
        low=0, high=max_in_kv_len + 1, shape=(batch_size,), dtype="int32"
    )
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    return q_lens, in_kv_lens, seq_lens


def generate_cumsum_lens(lens):
    return paddle.concat(
        x=[
            paddle.to_tensor(data=[0], dtype="int32", place=GPU_DEVICE),
            paddle.cumsum(x=lens.to(GPU_DEVICE), axis=0, dtype="int32"),
        ]
    )


def create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype):
    q = paddle.randn(
        shape=[paddle.sum(x=q_lens).item(), num_qo_heads, head_dim],
        dtype="bfloat16" if q_dtype == "fp8" else DTYPE_MAP[q_dtype],
    )
    if q_dtype == "fp8":
        q, q_scale = to_float8(q)
        ref_q = q.astype(dtype="bfloat16") * q_scale
    else:
        q_scale = 1.0
        ref_q = q
    return q, q_scale, ref_q


def create_kv_cache(
    batch_size, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, ref_kv_dtype
):
    max_seq_len = paddle.max(x=seq_lens).item()
    num_tokens = max_seq_len * batch_size
    num_pages = (num_tokens + page_size - 1) // page_size
    ref_kv_dtype_torch = DTYPE_MAP[ref_kv_dtype]
    if kv_dtype != "fp8":
        assert (
            kv_dtype == ref_kv_dtype
        ), "kv_dtype and ref_kv_dtype must be the same for non-fp8 kv_cache"
    k_cache = paddle.randn(
        shape=[num_pages, num_kv_heads, page_size, head_dim], dtype=ref_kv_dtype_torch
    )
    v_cache = paddle.randn(
        shape=[num_pages, num_kv_heads, page_size, head_dim], dtype=ref_kv_dtype_torch
    )
    if kv_dtype == "fp8":
        k_cache, k_scale = to_float8(k_cache)
        v_cache, v_scale = to_float8(v_cache)
        ref_kv_cache = paddle.stack(
            x=[
                k_cache.to(ref_kv_dtype_torch) * k_scale,
                v_cache.to(ref_kv_dtype_torch) * v_scale,
            ],
            axis=1,
        )
    else:
        k_scale = v_scale = 1.0
        ref_kv_cache = paddle.stack(x=[k_cache, v_cache], axis=1)
    kv_cache = paddle.stack(x=[k_cache, v_cache], axis=1)
    return kv_cache, k_scale, v_scale, ref_kv_cache


def create_page_table(batch_size, seq_lens, page_size):
    page_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_pages_per_seq = paddle.max(x=page_per_seq).item()
    total_pages_needed = paddle.sum(x=page_per_seq).item()
    all_page_ids = paddle.randperm(n=total_pages_needed, dtype="int32")
    page_tables = paddle.zeros(shape=(batch_size, max_num_pages_per_seq), dtype="int32")
    page_id = 0
    for i in range(batch_size):
        num_pages_needed = page_per_seq[i]
        page_tables[i, :num_pages_needed] = all_page_ids[
            page_id : page_id + num_pages_needed
        ]
        page_id += num_pages_needed
    return page_tables, all_page_ids, page_per_seq


def create_output(q, o_dtype, create_out_tensor):
    if o_dtype == "fp8":
        o_scale = paddle.rand(shape=[1]).item() * 0.5 + 0.5
    else:
        o_scale = 1.0
    o_sf_scale = 300 if o_dtype == "nvfp4" else None
    o_sf_vec_size = 16 if o_dtype == "nvfp4" else None
    if create_out_tensor:
        if o_dtype == "nvfp4":
            fp4_out_shape = tuple(q.shape)[:-1] + (ceil_div(tuple(q.shape)[-1], 2),)
            extra_size = paddle.randint(low=0, high=256, shape=(1,)).item()
            fp4_out_scale_shape = round_up(
                tuple(q.shape)[0] + extra_size, 128
            ), round_up(tuple(q.shape)[1] * tuple(q.shape)[2] // o_sf_vec_size, 4)
            out_scale_factor = paddle.empty(
>>>>>>                shape=fp4_out_scale_shape, dtype=paddle.float8_e4m3fn
            )
            rounded_extra_size = fp4_out_scale_shape[0] - tuple(q.shape)[0]
            o_sf_start_index = (
                paddle.randint(low=0, high=rounded_extra_size, shape=(1,)).item()
                if rounded_extra_size > 0
                else 0
            )
            out_data = paddle.empty(shape=fp4_out_shape, dtype="uint8")
            out = FP4Tensor(out_data, out_scale_factor, o_sf_start_index)
        else:
            out = paddle.empty_like(x=q, dtype=DTYPE_MAP[o_dtype])
    else:
        out = None
    return out, o_scale, o_sf_scale, o_sf_vec_size


def get_last_page_len(seq_lens, page_size):
    kv_last_page_len = seq_lens % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size
    return kv_last_page_len


def unpack_compare_nvfp4(
    output: FP4Tensor,
    output_ref,
    o_sf_scale,
    o_sf_vec_size,
    sf_rtol=0.2,
    sf_atol=0.2,
    rmse_tol=0.3,
):
    output_ref, out_scale_factor_ref = ref_fp4_quant(
        output_ref, o_sf_scale, o_sf_vec_size
    )
    output_unpacked = cast_from_fp4(output.data)
    out_scale_factor = recover_swizzled_scales(
        output.scale,
        tuple(output_unpacked.shape)[0],
        math.prod(list(tuple(output_unpacked.shape)[1:])),
        o_sf_vec_size,
        output.scale_start_index,
    )
    assert paddle.allclose(
        x=out_scale_factor.astype(dtype="float32").reshape(
            tuple(out_scale_factor_ref.shape)
        ),
        y=out_scale_factor_ref.astype(dtype="float32"),
        rtol=sf_rtol,
        atol=sf_atol,
    ).item(), ""
    rmse = paddle.sqrt(
        x=paddle.mean(
            x=(
                output_unpacked.astype(dtype="float32")
                - output_ref.astype(dtype="float32")
            )
            ** 2
        )
    )
    assert rmse.item() < rmse_tol
    return output_unpacked, output_ref


@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_prefill(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
):
    paddle.seed(seed=0)
    head_dim = 128
    MAX_Q_LEN = 511
    MAX_IN_KV_LEN = 2047
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens(
        batch_size, MAX_Q_LEN, MAX_IN_KV_LEN
    )
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    out, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor
    )
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = paddle.zeros(shape=128 * 1024 * 1024, dtype="int8")
    workspace_buffer = global_workspace_buffer
    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    plan_params = {
        "qo_indptr": q_indptr,
        "paged_kv_indptr": kv_indptr,
        "paged_kv_indices": all_page_ids,
        "paged_kv_last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim,
        "page_size": page_size,
        "causal": True,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": 0.0,
        "q_data_type": ref_q.dtype,
        "kv_data_type": ref_kv_cache.dtype,
        "window_left": window_left,
    }
    wrapper_ref.plan(**plan_params)
    output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
    sm_scale = float(1.0 / head_dim**0.5)
    output = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        paddle.max(x=q_lens).item(),
        paddle.max(x=seq_lens).item(),
        q_scale * k_scale * sm_scale,
        v_scale / o_scale,
        batch_size,
        q_indptr,
        kv_indptr,
        window_left,
        out=out,
        out_dtype=DTYPE_MAP[o_dtype],
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        enable_pdl=enable_pdl,
    )
    if o_dtype == "nvfp4":
        output, output_ref = unpack_compare_nvfp4(
            output, output_ref, o_sf_scale, o_sf_vec_size
        )
        assert o_scale == 1.0
        rtol, atol = 0.4, 1.0
    elif q_dtype == "fp8" and o_dtype == "fp8":
        rtol, atol = 0.05, 0.07
    elif q_dtype == "fp8" and o_dtype in ["bf16", "fp16"]:
        rtol, atol = 0.04, 0.06
    else:
        rtol, atol = 0.01, 0.01
    assert paddle.allclose(
        x=output.astype(dtype="float32") * o_scale,
        y=output_ref.astype(dtype="float32"),
        rtol=rtol,
        atol=atol,
    ).item(), ""
    if o_dtype != "nvfp4":
        wrapper_trtllm_gen = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        plan_params["q_data_type"] = q.dtype
        plan_params["kv_data_type"] = kv_cache.dtype
        wrapper_trtllm_gen.plan(**plan_params)
        output_wrapper = wrapper_trtllm_gen.run(
            q.contiguous(),
            kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale / o_scale,
            enable_pdl=enable_pdl,
        )
        if v_scale == o_scale == 1.0:
            assert (output_wrapper == output).astype("bool").all()
        else:
            assert paddle.allclose(
                x=output.astype(dtype="float32"),
                y=output_wrapper.astype(dtype="float32"),
                rtol=0.1,
                atol=0.1,
            ).item(), ""


@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("num_kv_heads", [2, 4])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("bf16", "fp8", "bf16"),
        ("fp16", "fp8", "fp16"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [True, False, None])
def test_trtllm_batch_decode(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
):
    paddle.seed(seed=0)
    head_dim = 128
    MAX_Q_LEN = 1
    MAX_IN_KV_LEN = 110
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, in_kv_lens, seq_lens = generate_seq_lens(
        batch_size, MAX_Q_LEN, MAX_IN_KV_LEN
    )
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    kv_cache, k_scale, v_scale, ref_kv_cache = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    out, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor
    )
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = paddle.zeros(shape=128 * 1024 * 1024, dtype="int8")
    workspace_buffer = global_workspace_buffer
    wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    plan_params = {
        "indptr": kv_indptr,
        "indices": all_page_ids,
        "last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "page_size": page_size,
        "pos_encoding_mode": "NONE",
        "kv_data_type": ref_kv_cache.dtype,
        "q_data_type": ref_q.dtype,
        "window_left": window_left,
    }
    wrapper_ref.plan(**plan_params)
    output_ref = wrapper_ref.run(ref_q, ref_kv_cache)
    sm_scale = float(1.0 / head_dim**0.5)
    output = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        q.contiguous(),
        kv_cache,
        workspace_buffer,
        page_table,
        seq_lens.to(GPU_DEVICE),
        paddle.max(x=seq_lens).item(),
        q_scale * k_scale * sm_scale,
        v_scale / o_scale,
        window_left,
        out=out,
        out_dtype=DTYPE_MAP[o_dtype],
        o_sf_scale=o_sf_scale,
        o_sf_vec_size=o_sf_vec_size,
        enable_pdl=enable_pdl,
    )
    if o_dtype == "nvfp4":
        output, output_ref = unpack_compare_nvfp4(
            output, output_ref, o_sf_scale, o_sf_vec_size
        )
        assert o_scale == 1.0
        rtol, atol = 0.3, 1.0
    elif q_dtype == "fp8" and o_dtype == "fp8":
        rtol, atol = 0.05, 0.07
    elif q_dtype == "fp8" and o_dtype in ["bf16", "fp16"]:
        rtol, atol = 0.04, 0.06
    else:
        rtol, atol = 0.01, 0.01
    assert paddle.allclose(
        x=output.astype(dtype="float32") * o_scale,
        y=output_ref.astype(dtype="float32"),
        rtol=rtol,
        atol=atol,
    ).item(), ""
    if o_dtype != "nvfp4":
        wrapper_trtllm_gen = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="trtllm-gen"
        )
        plan_params["q_data_type"] = q.dtype
        plan_params["kv_data_type"] = kv_cache.dtype
        wrapper_trtllm_gen.plan(**plan_params)
        output_wrapper = wrapper_trtllm_gen.run(
            q.contiguous(),
            kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale / o_scale,
            enable_pdl=enable_pdl,
        )
        if v_scale == o_scale == 1.0:
            assert (output_wrapper == output).astype("bool").all()
        else:
            assert paddle.allclose(
                x=output.astype(dtype="float32"),
                y=output_wrapper.astype(dtype="float32"),
                rtol=0.1,
                atol=0.1,
            ).item(), ""


@pytest.mark.parametrize("batch_size", [4, 128, 256])
@pytest.mark.parametrize("s_qo", [32, 64, 87])
@pytest.mark.parametrize("s_kv", [32, 64, 87])
@pytest.mark.parametrize("num_kv_heads", [16, 32])
@pytest.mark.parametrize("head_grp_size", [1, 5, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_gen_prefill_deepseek(
    batch_size, s_qo, s_kv, num_kv_heads, head_grp_size, causal
):
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")
    num_qo_heads = num_kv_heads * head_grp_size
    head_dim_qk = 192
    head_dim_vo = 128
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
    cumsum_s_kv = paddle.sum(x=actual_seq_lens_kv)
    q = paddle.randn(shape=[cumsum_s_qo, num_qo_heads, head_dim_qk], dtype="bfloat16")
    k_cache = paddle.randn(
        shape=(cumsum_s_kv, num_kv_heads, head_dim_qk), dtype="bfloat16"
    )
    v_cache = paddle.randn(
        shape=(cumsum_s_kv, num_kv_heads, head_dim_vo), dtype="bfloat16"
    )
    scale = float(1.0 / head_dim_qk**0.5)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
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
        sm_scale=scale,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
    )
    output_ref, lse_ref = wrapper.run(q, k_cache, v_cache, return_lse=True)
    output = paddle.empty_like(x=output_ref)
    bmm1_scale = scale
    bmm2_scale = 1.0
    output_trtllm, lse_trtllm = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q,
        k_cache,
        v_cache,
        workspace_buffer,
        actual_seq_lens_kv,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        out=output,
    )
    assert paddle.allclose(
        x=output_trtllm, y=output_ref, atol=0.01, rtol=0.01
    ).item(), ""
    assert paddle.allclose(x=lse_trtllm, y=lse_ref, atol=0.001, rtol=0.001).item(), ""


if __name__ == "__main__":
    test_trtllm_batch_prefill("HND", 128, 32, 2, 5, -1, "half", "half", "half", False)
    test_trtllm_batch_decode("HND", 128, 32, 2, 5, -1, "half", "half", "half", False)
