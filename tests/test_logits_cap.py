import paddle

"""
Copyright (c) 2024 by FlashInfer team.

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
import math

import pytest
from jit_utils import (gen_decode_attention_modules,
                       gen_prefill_attention_modules)

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            ["float16"], ["float16"], [128, 256], [0], [False], [False, True]
        )
        + gen_prefill_attention_modules(
            ["float16"], ["float16"], [128, 256], [0], [False], [False, True], [False]
        ),
        verbose=False,
    )
    yield


def attention_logits_soft_cap_torch(q, k, v, soft_cap):
    q_len, num_heads, head_dim = tuple(q.shape)
    scores = paddle.einsum(
        "qhd,khd->qkh", q.astype(dtype="float32"), k.astype(dtype="float32")
    )
    scores *= 1.0 / math.sqrt(head_dim)
    scores = soft_cap * paddle.nn.functional.tanh(x=scores / soft_cap)
    attn = paddle.nn.functional.softmax(x=scores, axis=1)
    return paddle.einsum("ovh,vhd->ohd", attn, v.astype(dtype="float32")).to(q)


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729, 33001])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("soft_cap", [1.0, 30.0, 50.0])
def test_single_decode_logits_soft_cap(seq_len, num_heads, head_dim, soft_cap):
    q = paddle.randn(shape=[num_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_heads, head_dim], dtype="float16")
    o = flashinfer.single_decode_with_kv_cache(q, k, v, logits_soft_cap=soft_cap)
    o_ref = attention_logits_soft_cap_torch(
        q.unsqueeze(axis=0), k, v, soft_cap
    ).squeeze(axis=0)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("q_len", [1, 17, 81, 987])
@pytest.mark.parametrize("kv_len", [1, 17, 81, 987, 31111])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("soft_cap", [1.0, 30.0, 50.0])
def test_single_prefill_logits_soft_cap(q_len, kv_len, num_heads, head_dim, soft_cap):
    q = paddle.randn(shape=[q_len, num_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[kv_len, num_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[kv_len, num_heads, head_dim], dtype="float16")
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, logits_soft_cap=soft_cap)
    o_ref = attention_logits_soft_cap_torch(q, k, v, soft_cap)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.01, atol=0.01).item(), ""


if __name__ == "__main__":
    test_single_decode_logits_soft_cap(9, 32, 128, 30.0)
    test_single_prefill_logits_soft_cap(64, 64, 1, 128, 30.0)
