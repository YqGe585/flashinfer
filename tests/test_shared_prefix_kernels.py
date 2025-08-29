import paddle

"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pytest
from jit_utils import (gen_decode_attention_modules,
                       gen_prefill_attention_modules)

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            ["float16"], ["float16"], [128, 256], [0], [False], [False]
        )
        + gen_prefill_attention_modules(
            ["float16"], ["float16"], [128, 256], [0], [False], [False], [False]
        ),
        verbose=False,
    )
    yield


def ceil_div(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("stage", ["decode", "append"])
@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("unique_kv_len", [37, 17])
@pytest.mark.parametrize("shared_kv_len", [128, 512, 2048])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_attention_with_shared_prefix_paged_kv_cache(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
):
    if stage == "decode" and causal:
        pytest.skip("Causal attention is not required in decode stage")
    assert shared_kv_len % page_size == 0
    kv_layout = "NHD"
    if stage == "append":
        q = (
            paddle.randn(shape=[batch_size * unique_kv_len, num_heads, head_dim])
            .to(0)
            .astype(dtype="float16")
        )
        q_indptr = (
            paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
            * unique_kv_len
        )
    else:
        q = (
            paddle.randn(shape=[batch_size, num_heads, head_dim])
            .to(0)
            .astype(dtype="float16")
        )
        q_indptr = (
            paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
        )
    k_shared = (
        paddle.randn(shape=[shared_kv_len, num_heads, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    v_shared = (
        paddle.randn(shape=[shared_kv_len, num_heads, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    k_unique = (
        paddle.randn(shape=[batch_size * unique_kv_len, num_heads, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    v_unique = (
        paddle.randn(shape=[batch_size * unique_kv_len, num_heads, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    kv_data = (
        paddle.zeros(
            shape=[
                ceil_div(shared_kv_len, page_size)
                + batch_size * ceil_div(unique_kv_len, page_size),
                2,
                page_size,
                num_heads,
                head_dim,
            ]
        )
        .to(0)
        .astype(dtype="float16")
    )
    shared_kv_indices = (
        paddle.arange(start=0, end=ceil_div(shared_kv_len, page_size))
        .to(0)
        .astype(dtype="int32")
    )
    shared_append_indptr = (
        paddle.arange(start=0, end=2).to(0).astype(dtype="int32") * shared_kv_len
    )
    shared_kv_indptr = paddle.arange(start=0, end=2).to(0).astype(
        dtype="int32"
    ) * ceil_div(shared_kv_len, page_size)
    shared_last_page_len = paddle.full(
        shape=(1,), fill_value=(shared_kv_len - 1) % page_size + 1, dtype="int32"
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            tuple(k_shared.shape)[0],
        ),
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout
    )
    unique_kv_indices = paddle.arange(
        start=0, end=batch_size * ceil_div(unique_kv_len, page_size)
    ).to(0).astype(dtype="int32") + ceil_div(shared_kv_len, page_size)
    unique_append_indptr = (
        paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
        * unique_kv_len
    )
    unique_kv_indptr = paddle.arange(start=0, end=batch_size + 1).to(0).astype(
        dtype="int32"
    ) * ceil_div(unique_kv_len, page_size)
    unique_last_page_len = paddle.full(
        shape=(batch_size,),
        fill_value=(unique_kv_len - 1) % page_size + 1,
        dtype="int32",
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            tuple(k_unique.shape)[0],
        ),
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout
    )
    if stage == "decode":
        multi_level_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
            2, paddle.empty(shape=32 * 1024 * 1024, dtype="int8").to(0), kv_layout
        )
        shared_prefix_decode_wrapper = (
            flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
                paddle.empty(shape=32 * 1024 * 1024, dtype="int8").to(0), kv_layout
            )
        )
    else:
        multi_level_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
            2, paddle.empty(shape=32 * 1024 * 1024, dtype="int8").to(0), kv_layout
        )
        shared_prefix_prefill_wrapper = (
            flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
                paddle.empty(shape=32 * 1024 * 1024, dtype="int8").to(0), kv_layout
            )
        )
    qo_indptr_top = paddle.to_tensor(data=[0, tuple(q.shape)[0]], dtype="int32").to(0)
    if stage == "decode":
        qo_indptr_bottom = paddle.arange(start=0, end=batch_size + 1, dtype="int32").to(
            0
        )
        multi_level_wrapper.plan(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_multi_level = multi_level_wrapper.run(q, kv_data)
    else:
        qo_indptr_bottom = (
            paddle.arange(start=0, end=batch_size + 1, dtype="int32").to(0)
            * unique_kv_len
        )
        multi_level_wrapper.plan(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
            causal=causal,
        )
        o_multi_level = multi_level_wrapper.run(q, kv_data)
    if stage == "decode":
        shared_prefix_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_decode_wrapper.forward(
            q, k_shared, v_shared, kv_data
        )
    else:
        shared_prefix_prefill_wrapper.begin_forward(
            q_indptr,
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_prefill_wrapper.forward(
            q, k_shared, v_shared, kv_data, causal=causal
        )
    assert paddle.allclose(
        x=o_multi_level, y=o_two_level, rtol=0.001, atol=0.001
    ).item(), ""


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_tries", [50])
def test_merge_state_in_place_with_mask(seed, num_tries):
    seq_len = 512
    num_heads = 32
    head_dim = 128
    va = (
        paddle.randn(shape=[seq_len, num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    sa = paddle.randn(shape=[seq_len, num_heads], dtype="float32").to("gpu:0")
    vb = (
        paddle.randn(shape=[seq_len, num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    sb = paddle.randn(shape=[seq_len, num_heads], dtype="float32").to("gpu:0")
    va_orginal = va.clone()
    sa_original = sa.clone()
    flashinfer.merge_state_in_place(va, sa, vb, sb)
    va_merged_ref = va.clone()
    sa_merged_ref = sa.clone()
    assert not paddle.allclose(x=va_merged_ref, y=va_orginal).item()
    assert not paddle.allclose(x=sa_merged_ref, y=sa_original).item()
    mask = paddle.ones(shape=seq_len, dtype="bool").to("gpu:0")
    va = va_orginal.clone()
    sa = sa_original.clone()
    flashinfer.merge_state_in_place(va, sa, vb, sb, mask=mask)
    va_merged = va
    sa_merged = sa
    assert paddle.allclose(
        x=va_merged, y=va_merged_ref, rtol=0.001, atol=0.001
    ).item(), ""
    assert paddle.allclose(
        x=sa_merged, y=sa_merged_ref, rtol=0.001, atol=0.001
    ).item(), ""
    mask = paddle.zeros(shape=seq_len, dtype="bool").to("gpu:0")
    va = va_orginal.clone()
    sa = sa_original.clone()
    flashinfer.merge_state_in_place(va, sa, vb, sb, mask=mask)
    va_merged = va
    sa_merged = sa
    assert paddle.allclose(x=va_merged, y=va_orginal, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(
        x=sa_merged, y=sa_original, rtol=0.001, atol=0.001
    ).item(), ""
    randgen = paddle.framework.core.default_cpu_generator()
    randgen.manual_seed(seed)
    for _ in range(num_tries):
        rand_mask = (paddle.rand(shape=seq_len, dtype="float32") > 0.5).to(dtype="bool")
        true_indices = rand_mask.nonzero()
        false_indices = (rand_mask == 0).nonzero()
        va = va_orginal.clone()
        sa = sa_original.clone()
        flashinfer.merge_state_in_place(va, sa, vb, sb, mask=rand_mask)
        va_merged = va
        sa_merged = sa
        assert paddle.allclose(
            x=va_merged[false_indices],
            y=va_orginal[false_indices],
            rtol=0.001,
            atol=0.001,
        ).item(), ""
        assert paddle.allclose(
            x=sa_merged[false_indices],
            y=sa_original[false_indices],
            rtol=0.001,
            atol=0.001,
        ).item(), ""
        assert paddle.allclose(
            x=va_merged[true_indices],
            y=va_merged_ref[true_indices],
            rtol=0.001,
            atol=0.001,
        ).item(), ""
        assert paddle.allclose(
            x=sa_merged[true_indices],
            y=sa_merged_ref[true_indices],
            rtol=0.001,
            atol=0.001,
        ).item(), ""


if __name__ == "__main__":
    test_batch_attention_with_shared_prefix_paged_kv_cache(
        "decode", 12, 37, 128, 8, False, 128, 16
    )
    test_batch_attention_with_shared_prefix_paged_kv_cache(
        "append", 12, 37, 128, 8, True, 128, 16
    )
