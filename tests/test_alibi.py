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
from alibi_reference import alibi_attention
from jit_utils import (gen_decode_attention_modules,
                       gen_prefill_attention_modules)

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            ["float16"], ["float16"], [128, 256], [0, 2], [False], [False]
        )
        + gen_prefill_attention_modules(
            ["float16"], ["float16"], [128, 256], [0, 2], [False], [False], [False]
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("seq_len", [1, 9, 81, 729])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_single_decode_alibi(seq_len, num_heads, head_dim):
    q = paddle.randn(shape=[num_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_heads, head_dim], dtype="float16")
    o = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ALIBI")
    mask = paddle.ones(shape=[1, seq_len], dtype="bool")
    o_ref = alibi_attention(q.unsqueeze(axis=0), k, v, mask).squeeze(0)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("q_len", [1, 17, 81, 987])
@pytest.mark.parametrize("kv_len", [1, 17, 81, 987])
@pytest.mark.parametrize("num_heads", [4, 8, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_alibi(q_len, kv_len, num_heads, head_dim, causal):
    if causal and q_len > kv_len:
        pytest.skip("Causal attention requires q_len <= kv_len")
    q = paddle.randn(shape=[q_len, num_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[kv_len, num_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[kv_len, num_heads, head_dim], dtype="float16")
    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, pos_encoding_mode="ALIBI"
    )
    mask = paddle.ones(shape=[q_len, kv_len], dtype="bool")
    if causal:
        mask = paddle.tril(x=mask, diagonal=kv_len - q_len)
    o_ref = alibi_attention(q, k, v, mask)
    assert paddle.allclose(x=o, y=o_ref, rtol=0.01, atol=0.01).item(), ""


if __name__ == "__main__":
    test_single_decode_alibi(4096, 32, 128)
    test_single_prefill_alibi(128, 128, 8, 128, False)
