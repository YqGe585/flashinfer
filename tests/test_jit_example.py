import sys

sys.path.append("/home/flashinfer")
import functools
import math

import paddle
import pytest
from paddle_utils import *

import flashinfer
from flashinfer.decode import single_decode_with_kv_cache_with_jit_module
from flashinfer.jit.attention import (gen_customize_single_decode_module,
                                      gen_customize_single_prefill_module)
from flashinfer.prefill import single_prefill_with_kv_cache_with_jit_module
from flashinfer.utils import MaskMode, is_sm90a_supported


def test_single_decode_mask():
    paddle.seed(seed=42)
    variant_decl = """
struct SingleDecodeWithCustomMask : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint8_t* custom_mask_ptr;
  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ SingleDecodeWithCustomMask(const Params& params, uint32_t batch_idx,
                                          uint8_t* smem_ptr) {
    custom_mask_ptr = params.custom_mask;
    qo_len = 1;
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    const uint32_t offset = kv_idx;
    return ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
  })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    return output;
  })
};
"""
    jit_module = gen_customize_single_decode_module(
        "single_decode_custom_mask",
        "float16",
        "float16",
        "float16",
        128,
        128,
        ["custom_mask"],
        ["uint8_t"],
        ["sm_scale"],
        ["double"],
        "SingleDecodeWithCustomMask",
        variant_decl,
    ).build_and_load()
    f = functools.partial(single_decode_with_kv_cache_with_jit_module, jit_module)
    q = paddle.randn(shape=[32, 128], dtype="float16")
    k = paddle.randn(shape=[254, 32, 128], dtype="float16")
    v = paddle.randn(shape=[254, 32, 128], dtype="float16")
    sm_scale = 1.0 / math.sqrt(128)
    custom_mask = paddle.randint(low=0, high=2, shape=(254,), dtype="uint8")
    packed_custom_mask = flashinfer.packbits(custom_mask, bitorder="little")
    o = f(q, k, v, packed_custom_mask, sm_scale)
    p = (
        paddle.einsum(
            "hd,nhd->hn", q.astype(dtype="float32"), k.astype(dtype="float32")
        )
        * sm_scale
    )
    p[:, paddle.nonzero(x=paddle.logical_not(x=custom_mask)).squeeze()] = -float("inf")
    o_ref = paddle.einsum(
        "hn,nhd->hd",
        paddle.nn.functional.softmax(x=p, axis=-1),
        v.astype(dtype="float32"),
    ).astype(dtype="float16")
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""


flash_sigmoid_sm80_decl = """
struct FlashSigmoid : AttentionVariantBase {
  static constexpr bool use_softmax = false;

  uint32_t window_left, qo_len, kv_len;
  float sigmoid_scale_log2;
  float sigmoid_bias_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ FlashSigmoid(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sigmoid_bias_log2 = params.sigmoid_bias * math::log2e;
    sigmoid_scale_log2 = params.logits_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits * sigmoid_scale_log2 + sigmoid_bias_log2)));
  });

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    return output;
  })
};
"""
flash_sigmoid_sm90_decl = """
struct FlashSigmoid : AttentionVariantBase {
  float logits_scale_log2, sigmoid_bias_log2e;
  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ FlashSigmoid(const MainloopParams& params, const BlockCoord& block_coord) {
    logits_scale_log2 = params.additional_params.logits_scale * math::log2e;
    sigmoid_bias_log2e = params.additional_params.sigmoid_bias * math::log2e;
  }


  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return DefaultUpdater<NUM_ROWS_PER_THREAD>();
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return math::ptx_rcp(1.f + math::ptx_exp2(-float(logits * logits_scale_log2 + sigmoid_bias_log2e)));
  });
};
"""


def test_flash_sigmoid():
    paddle.seed(seed=42)
    variant_decl = flash_sigmoid_sm80_decl
    jit_module = gen_customize_single_prefill_module(
        "fa2",
        "single_prefill_flash_sigmoid",
        "float16",
        "float16",
        "float16",
        128,
        128,
        [],
        [],
        ["logits_scale", "sigmoid_bias"],
        ["double", "double"],
        "FlashSigmoid",
        variant_decl,
    ).build_and_load()
    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)
    q = paddle.randn(shape=[128, 8, 128], dtype="float16")
    k = paddle.randn(shape=[1027, 8, 128], dtype="float16")
    v = paddle.randn(shape=[1027, 8, 128], dtype="float16")
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25
    o = f(q, k, v, logits_scale, sigmoid_bias, mask_mode=MaskMode.NON_CAUSAL.value)
    p = paddle.nn.functional.sigmoid(
        x=paddle.einsum(
            "mhd,nhd->hmn", q.astype(dtype="float32"), k.astype(dtype="float32")
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = paddle.einsum("hmn,nhd->mhd", p, v.astype(dtype="float32")).astype(
        dtype="float16"
    )
    assert paddle.allclose(x=o, y=o_ref, rtol=0.02, atol=0.02).item(), ""


def test_dump_logits():
    paddle.seed(seed=42)
    variant_decl = """
struct DumpLogits : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ DumpLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (qo_idx < qo_len && kv_idx < kv_len) {
      params.output_logits[qo_head_idx * (qo_len * kv_len) + qo_idx * kv_len + kv_idx] = logits * params.sm_scale;
    }
    return logits;
  });
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa2",
        "single_prefill_dump_logits",
        "float16",
        "float16",
        "float16",
        128,
        128,
        ["output_logits"],
        ["float"],
        ["sm_scale"],
        ["double"],
        "DumpLogits",
        variant_decl,
    ).build_and_load()
    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)
    q = paddle.randn(shape=[128, 32, 128], dtype="float16")
    k = paddle.randn(shape=[1023, 32, 128], dtype="float16")
    v = paddle.randn(shape=[1023, 32, 128], dtype="float16")
    logits = paddle.empty(shape=[32, 128, 1023], dtype="float32")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, logits, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)
    p = (
        paddle.einsum(
            "mhd,nhd->hmn", q.astype(dtype="float32"), k.astype(dtype="float32")
        )
        * sm_scale
    )
    o_ref = paddle.einsum(
        "hmn,nhd->mhd",
        paddle.nn.functional.softmax(x=p, axis=-1),
        v.astype(dtype="float32"),
    ).astype(dtype="float16")
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=logits, y=p, rtol=0.02, atol=0.02).item(), ""


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_flash_sigmoid(use_tensor_cores):
    paddle.seed(seed=42)
    variant_decl = flash_sigmoid_sm80_decl
    jit_args = (
        f"batch_decode_flash_sigmoid_sm80_{use_tensor_cores}",
        "float16",
        "float16",
        "float16",
        "int32",
        128,
        128,
        [],
        [],
        ["logits_scale", "sigmoid_bias"],
        ["double", "double"],
        "FlashSigmoid",
        variant_decl,
    )
    float_workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="uint8")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=use_tensor_cores,
        jit_args=jit_args,
    )
    batch_size = 128
    seq_len_per_request = 1024
    kv_indptr_host = paddle.arange(
        start=0,
        end=batch_size * seq_len_per_request + 1,
        step=seq_len_per_request,
        dtype="int32",
    )
    page_size = 1
    kv_indices_host = paddle.arange(
        start=0, end=batch_size * seq_len_per_request, dtype="int32"
    )
    last_page_len_host = paddle.full(shape=(batch_size,), fill_value=1, dtype="int32")
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128
    wrapper.plan(
        kv_indptr_host,
        kv_indices_host,
        last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type="float16",
        kv_data_type="float16",
    )
    q = paddle.randn(shape=[batch_size, num_qo_heads, head_dim], dtype="float16")
    k_cache = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_kv_heads, head_dim],
        dtype="float16",
    )
    v_cache = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_kv_heads, head_dim],
        dtype="float16",
    )
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25
    o = wrapper.run(q, (k_cache, v_cache), logits_scale, sigmoid_bias)
    p = paddle.nn.functional.sigmoid(
        x=paddle.einsum(
            "bhd,bnhd->bhn",
            q.view(batch_size, num_qo_heads, head_dim).astype(dtype="float32"),
            k_cache.view(
                batch_size, seq_len_per_request, num_kv_heads, head_dim
            ).astype(dtype="float32"),
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = (
        paddle.einsum(
            "bhn,bnhd->bhd",
            p,
            v_cache.view(
                batch_size, seq_len_per_request, num_kv_heads, head_dim
            ).astype(dtype="float32"),
        )
        .astype(dtype="float16")
        .reshape(batch_size, num_qo_heads, head_dim)
    )
    assert paddle.allclose(x=o, y=o_ref, rtol=0.02, atol=0.02).item(), ""


def test_batch_prefill_flash_sigmoid():
    paddle.seed(seed=42)
    variant_decl = flash_sigmoid_sm80_decl
    jit_args = (
        "batch_prefill_flash_sigmoid_sm80",
        "float16",
        "float16",
        "float16",
        "int32",
        128,
        128,
        [],
        [],
        ["logits_scale", "sigmoid_bias"],
        ["double", "double"],
        "FlashSigmoid",
        variant_decl,
    )
    float_workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="uint8")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    batch_size = 128
    seq_len_per_request = 1024
    qo_indptr_host = paddle.arange(
        start=0,
        end=batch_size * seq_len_per_request + 1,
        step=seq_len_per_request,
        dtype="int32",
    )
    kv_indptr_host = paddle.arange(
        start=0,
        end=batch_size * seq_len_per_request + 1,
        step=seq_len_per_request,
        dtype="int32",
    )
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128
    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=False,
        q_data_type="float16",
        kv_data_type="float16",
    )
    q = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_qo_heads, head_dim],
        dtype="float16",
    )
    k = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_kv_heads, head_dim],
        dtype="float16",
    )
    v = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_kv_heads, head_dim],
        dtype="float16",
    )
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25
    o = wrapper.run(q, k, v, logits_scale, sigmoid_bias)
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    kv_indices_host = paddle.arange(
        start=0, end=batch_size * seq_len_per_request, dtype="int32"
    )
    paged_kv_last_page_len_host = paddle.full(
        shape=(batch_size,), fill_value=1, dtype="int32"
    )
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )
    o_paged = wrapper_paged.run(q, (k, v), logits_scale, sigmoid_bias)
    p = paddle.nn.functional.sigmoid(
        x=paddle.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, seq_len_per_request, num_qo_heads, head_dim).astype(
                dtype="float32"
            ),
            k.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).astype(
                dtype="float32"
            ),
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = (
        paddle.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).astype(
                dtype="float32"
            ),
        )
        .astype(dtype="float16")
        .reshape(batch_size * seq_len_per_request, num_qo_heads, head_dim)
    )
    assert paddle.allclose(x=o, y=o_ref, rtol=0.02, atol=0.02).item(), ""
    assert paddle.allclose(x=o_paged, y=o_ref, rtol=0.02, atol=0.02).item(), ""


def test_batch_prefill_sm90_flash_sigmoid():
    if not is_sm90a_supported(device2str("cuda")):
        pytest.skip("SM90A is not supported")
    paddle.seed(seed=42)
    variant_decl = flash_sigmoid_sm90_decl
    jit_args = (
        "batch_prefill_flash_sigmoid",
        "float16",
        "float16",
        "float16",
        "int32",
        128,
        128,
        [],
        [],
        ["logits_scale", "sigmoid_bias"],
        ["double", "double"],
        "FlashSigmoid",
        variant_decl,
    )
    float_workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="uint8")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa3", jit_args=jit_args
    )
    batch_size = 128
    seq_len_per_request = 1024
    qo_indptr_host = paddle.arange(
        start=0,
        end=batch_size * seq_len_per_request + 1,
        step=seq_len_per_request,
        dtype="int32",
    )
    kv_indptr_host = paddle.arange(
        start=0,
        end=batch_size * seq_len_per_request + 1,
        step=seq_len_per_request,
        dtype="int32",
    )
    num_qo_heads = 32
    num_kv_heads = 32
    head_dim = 128
    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=False,
        q_data_type="float16",
        kv_data_type="float16",
    )
    q = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_qo_heads, head_dim],
        dtype="float16",
    )
    k = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_kv_heads, head_dim],
        dtype="float16",
    )
    v = paddle.randn(
        shape=[batch_size * seq_len_per_request, num_kv_heads, head_dim],
        dtype="float16",
    )
    logits_scale = 1.0 / math.sqrt(128)
    sigmoid_bias = 0.25
    o = wrapper.run(q, k, v, logits_scale, sigmoid_bias)
    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa3", jit_args=jit_args
    )
    kv_indices_host = paddle.arange(
        start=0, end=batch_size * seq_len_per_request, dtype="int32"
    )
    paged_kv_last_page_len_host = paddle.full(
        shape=(batch_size,), fill_value=1, dtype="int32"
    )
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
    )
    o_paged = wrapper_paged.run(q, (k, v), logits_scale, sigmoid_bias)
    p = paddle.nn.functional.sigmoid(
        x=paddle.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, seq_len_per_request, num_qo_heads, head_dim).astype(
                dtype="float32"
            ),
            k.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).astype(
                dtype="float32"
            ),
        )
        * logits_scale
        + sigmoid_bias
    )
    o_ref = (
        paddle.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, seq_len_per_request, num_kv_heads, head_dim).astype(
                dtype="float32"
            ),
        )
        .astype(dtype="float16")
        .reshape(batch_size * seq_len_per_request, num_qo_heads, head_dim)
    )
    assert paddle.allclose(x=o, y=o_ref, rtol=0.02, atol=0.02).item(), ""
    assert paddle.allclose(x=o_paged, y=o_ref, rtol=0.02, atol=0.02).item(), ""


def test_debug_print_logits():
    paddle.seed(seed=42)
    variant_decl = """
struct DebugPrintLogits : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ DebugPrintLogits(const Params& params, uint32_t batch_idx,
                                 uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (logits >= 5) {
      printf("Large logits at qo_idx=%d, kv_idx=%d, qo_head_idx=%d, kv_head_idx=%d: %.3f\\n",
              qo_idx, kv_idx, qo_head_idx, kv_head_idx, float(logits));
    }
    return logits;
  });
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa2",
        "batch_prefill_debug_print_logits",
        "float16",
        "float16",
        "float16",
        128,
        128,
        [],
        [],
        ["sm_scale"],
        ["double"],
        "DebugPrintLogits",
        variant_decl,
    ).build_and_load()
    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)
    q = paddle.randn(shape=[128, 32, 128], dtype="float16")
    k = paddle.randn(shape=[1023, 32, 128], dtype="float16")
    v = paddle.randn(shape=[1023, 32, 128], dtype="float16")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)
    p = (
        paddle.einsum(
            "mhd,nhd->hmn", q.astype(dtype="float32"), k.astype(dtype="float32")
        )
        * sm_scale
    )
    o_ref = paddle.einsum(
        "hmn,nhd->mhd",
        paddle.nn.functional.softmax(x=p, axis=-1),
        v.astype(dtype="float32"),
    ).astype(dtype="float16")
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""


def test_sm90_debug_print_logits():
    if not is_sm90a_supported(device2str("cuda")):
        pytest.skip("SM90A is not supported")
    paddle.seed(seed=42)
    variant_decl = """
struct DebugPrintLogits : AttentionVariantBase {
  float sm_scale_log2;
  int qo_len, kv_len;

  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ DebugPrintLogits(const MainloopParams& params, const BlockCoord& block_coord) {
    sm_scale_log2 = params.additional_params.sm_scale * math::log2e;
    auto [_, __, ___, ____, _____, qo_len_, kv_len_, batch_idx] =
        block_coord;

    qo_len = qo_len_;
    kv_len = kv_len_;
  }


  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmax<NUM_ROWS_PER_THREAD, /*WITH_SCALE*/false>(sm_scale_log2);
  }


  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    if (qo_idx < qo_len && kv_idx < kv_len) {
        printf(
            "---> LOGITS DEBUG: "
            "qo_idx=%-5d "
            "kv_idx=%-5d "
            "sm_scale_log2=%-12.5f "
            "logits=%-12.5f "
            "\\n",
            qo_idx,
            kv_idx,
            sm_scale_log2,
            static_cast<float>(logits));
    }
    logits *= sm_scale_log2;
    return logits;
  })
};
"""
    jit_module = gen_customize_single_prefill_module(
        "fa3",
        "debug_print_logits",
        "float16",
        "float16",
        "float16",
        128,
        128,
        [],
        [],
        ["sm_scale"],
        ["double"],
        "DebugPrintLogits",
        variant_decl,
    ).build_and_load()
    f = functools.partial(single_prefill_with_kv_cache_with_jit_module, jit_module)
    q = paddle.randn(shape=[16, 2, 128], dtype="float16")
    k = paddle.randn(shape=[16, 1, 128], dtype="float16")
    v = paddle.randn(shape=[16, 1, 128], dtype="float16")
    sm_scale = 1.0 / math.sqrt(128)
    o = f(q, k, v, sm_scale, mask_mode=MaskMode.NON_CAUSAL.value)
    p = (
        paddle.einsum(
            "mhd,nhd->hmn", q.astype(dtype="float32"), k.astype(dtype="float32")
        )
        * sm_scale
    )
    o_ref = paddle.einsum(
        "hmn,nhd->mhd",
        paddle.nn.functional.softmax(x=p, axis=-1),
        v.astype(dtype="float32"),
    ).astype(dtype="float16")
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""


if __name__ == "__main__":
    test_single_decode_mask()
    test_flash_sigmoid()
    test_dump_logits()
    test_debug_print_logits()
    test_sm90_debug_print_logits()
    test_batch_decode_flash_sigmoid(False)
    test_batch_decode_flash_sigmoid(True)
    test_batch_prefill_flash_sigmoid()
    test_batch_prefill_sm90_flash_sigmoid()
