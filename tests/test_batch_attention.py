import sys


import paddle
from flashinfer.paddle_utils import *

"""
Copyright (c) 2025 by FlashInfer team.

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
import numpy as np
import pytest
from jit_utils import (gen_persistent_batch_attention_modules,
                       gen_prefill_attention_modules)

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_persistent_batch_attention_modules(
            ["float16", "bfloat16"],
            ["float16", "bfloat16"],
            [64, 128, 256],
            [False, True],
        )
        + gen_prefill_attention_modules(
            ["float16", "bfloat16"],
            ["float16", "bfloat16"],
            [64, 128, 256],
            [0],
            [False],
            [False, True],
            [False],
        ),
        verbose=False,
    )


def _build_seq_len_configs():
    """
    Reproduce the sequence length configurations from the original benchmark (including random cases).
    Returns: List[List[Tuple[int,int]]]  -> Each element is a list of (kv_len, qo_len) pairs.
    """
    np.random.seed(42)
    paddle.seed(seed=42)
    seq_len_configs = [
        [(8190, 7939)],
        [(2, 235)] + [(1, 13353)],
        [(67, 1)],
        [(182, 1)],
        [(2011, 1)],
        [(2048, 1)] * 77,
        [(4099, 129)] * 2,
        [(600, 1)] * 132 * 2 + [(5000, 3)] * 128,
        [(1024, 1)] * 100 + [(8192, 17)] * 8,
        [(766, 2)] * 99 + [(1024, 512)] * 1,
    ]
    bsz, stride, sparsity = 256, 16, 0.05
    full_kv_len = np.random.randint(1000, 11000, size=bsz)
    seq_len = []
    for i in range(bsz):
        if i % stride == 0:
            kv_len, qo_len = full_kv_len[i], stride + 1
        else:
            kv_len, qo_len = int(full_kv_len[i] * sparsity), 1
        seq_len.append((kv_len, qo_len))
    seq_len_configs.append(seq_len)
    return seq_len_configs


def _run_attention(
    kv_lens,
    qo_lens,
    page_block_size=1,
    num_kv_heads=1,
    num_qo_heads=1,
    head_dim=128,
    layout="NHD",
    test_dtype="bfloat16",
    logits_soft_cap=0.0,
    device="cuda",
    causal=True,
):
    """
    Run both implementations and return (output_old, lse_old, output_new, lse_new)
    """
    dev = device2str(device)
    seq_lens = paddle.to_tensor(data=kv_lens, dtype="int32", place=dev)
    q_lens = paddle.to_tensor(data=qo_lens, dtype="int32", place=dev)
    seq_lens_blocks = paddle.ceil(x=seq_lens / page_block_size).astype(dtype="int32")
    q_indptr = paddle.concat(
        x=[paddle.to_tensor(data=[0], place=dev), paddle.cumsum(x=q_lens, axis=0)],
        axis=0,
    ).astype(dtype="int32")
    kv_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=dev),
            paddle.cumsum(x=seq_lens_blocks, axis=0),
        ],
        axis=0,
    ).astype(dtype="int32")
    num_blocks = kv_indptr[-1].item()
    q = paddle.rand(
        shape=[q_indptr[-1].item(), num_qo_heads, head_dim], dtype=test_dtype
    )
    if layout == "NHD":
        kv_data = paddle.randn(
            shape=[num_blocks, 2, page_block_size, num_kv_heads, head_dim],
            dtype=test_dtype,
        )
    elif layout == "HND":
        kv_data = paddle.randn(
            shape=[num_blocks, 2, num_kv_heads, page_block_size, head_dim],
            dtype=test_dtype,
        )
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        paddle.empty(shape=128 * 1024 * 1024, dtype="uint8"),
        kv_layout=layout,
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr,
        kv_indptr,
        paddle.arange(end=num_blocks).astype(dtype="int32"),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out_old, lse_old = wrapper_old.run(q, kv_data, return_lse=True)
    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        paddle.arange(end=num_blocks).astype(dtype="int32"),
        seq_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out_new, lse_new = wrapper.run(q, kv_data, logits_soft_cap=logits_soft_cap)
    paddle.device.synchronize()
    assert paddle.allclose(x=out_old, y=out_new, rtol=0.01, atol=0.01).item(), ""
    assert paddle.allclose(x=lse_old, y=lse_new, rtol=0.01, atol=0.01).item(), ""


@pytest.mark.parametrize("seq_len_pairs", _build_seq_len_configs())
@pytest.mark.parametrize("page_block_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [8, 1, 4])
@pytest.mark.parametrize("gqa_group_size", [1, 4, 7])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("test_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
def test_batch_attention_correctness(
    seq_len_pairs,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    causal,
    layout,
    test_dtype,
    logits_soft_cap,
):
    num_qo_heads = num_kv_heads * gqa_group_size
    kv_lens = [p[0] for p in seq_len_pairs]
    qo_lens = [p[1] for p in seq_len_pairs]
    _run_attention(
        kv_lens=kv_lens,
        qo_lens=qo_lens,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        causal=causal,
        layout=layout,
        test_dtype=test_dtype,
        logits_soft_cap=logits_soft_cap,
        device="cuda",
    )
