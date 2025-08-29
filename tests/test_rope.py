import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

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
import pytest
from rope_reference import *

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("llama_version", ["llama", "llama31"])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("inplace", [False, True])
def test_rope(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
    llama_version,
    partial_rotary_factor,
    inplace,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    nnz = batch_size * qkv_len
    qkv_packed = paddle.randn(
        shape=[nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim], dtype="float16"
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    indptr = paddle.to_tensor(
        data=[(i * qkv_len) for i in range(batch_size + 1)],
        dtype="int32",
        place="gpu:0",
    )
    offsets = paddle.full(shape=(batch_size,), fill_value=offset, dtype="int32")
    if llama_version == "llama":
        freqs_cis = precompute_freqs_cis(
            rotary_dim, qkv_len + offset, 10000.0, use_scaled=False, device="cuda:0"
        ).to("gpu:0")
    else:
        freqs_cis = precompute_freqs_cis(
            rotary_dim, qkv_len + offset, 500000.0, use_scaled=True, device="cuda:0"
        ).to("gpu:0")
    q_rot_ref, k_rot_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim)[..., :rotary_dim],
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim)[..., :rotary_dim],
        freqs_cis[offset : offset + qkv_len],
    )
    q_pass_ref = q.reshape(batch_size, qkv_len, num_qo_heads, head_dim)[
        ..., rotary_dim:
    ]
    k_pass_ref = k.reshape(batch_size, qkv_len, num_kv_heads, head_dim)[
        ..., rotary_dim:
    ]
    q_rope_ref = paddle.concat(x=[q_rot_ref, q_pass_ref], axis=-1).reshape(
        nnz, num_qo_heads, head_dim
    )
    k_rope_ref = paddle.concat(x=[k_rot_ref, k_pass_ref], axis=-1).reshape(
        nnz, num_kv_heads, head_dim
    )
    if llama_version == "llama":
        if inplace:
            flashinfer.apply_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=10000.0,
            )
            q_rope, k_rope = q, k
        else:
            q_rope, k_rope = flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=10000.0,
            )
    elif inplace:
        flashinfer.apply_llama31_rope_inplace(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=True,
            rope_theta=500000.0,
        )
        q_rope, k_rope = q, k
    else:
        q_rope, k_rope = flashinfer.apply_llama31_rope(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=True,
            rope_theta=500000.0,
        )
    assert paddle.allclose(x=q_rope_ref, y=q_rope, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=k_rope_ref, y=k_rope, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("llama_version", ["llama", "llama31"])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("idtype", ["int32", "int64"])
def test_rope_pos_ids(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
    llama_version,
    partial_rotary_factor,
    inplace,
    interleave,
    idtype,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    nnz = batch_size * qkv_len
    qkv_packed = paddle.randn(
        shape=[nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim], dtype="float16"
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    indptr = paddle.to_tensor(
        data=[(i * qkv_len) for i in range(batch_size + 1)], dtype=idtype, place="gpu:0"
    )
    offsets = paddle.full(shape=(batch_size,), fill_value=offset, dtype=idtype)
    pos_ids = paddle.concat(
        x=[
            paddle.arange(start=offset, end=qkv_len + offset, dtype=idtype)
            for _ in range(batch_size)
        ]
    ).to("gpu:0")
    if llama_version == "llama":
        if inplace:
            q_clone, k_clone = q.clone(), k.clone()
            flashinfer.apply_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=10000.0,
            )
            q_rope, k_rope = q, k
            flashinfer.apply_rope_pos_ids_inplace(
                q_clone,
                k_clone,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=10000.0,
            )
            q_rope_pos_ids, k_rope_pos_ids = q_clone, k_clone
        else:
            q_rope, k_rope = flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=10000.0,
            )
            q_rope_pos_ids, k_rope_pos_ids = flashinfer.apply_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=10000.0,
            )
    elif inplace:
        q_clone, k_clone = q.clone(), k.clone()
        flashinfer.apply_llama31_rope_inplace(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_theta=500000.0,
        )
        q_rope, k_rope = q, k
        flashinfer.apply_llama31_rope_pos_ids_inplace(
            q_clone,
            k_clone,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_theta=500000.0,
        )
        q_rope_pos_ids, k_rope_pos_ids = q_clone, k_clone
    else:
        q_rope, k_rope = flashinfer.apply_llama31_rope(
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_theta=500000.0,
        )
        q_rope_pos_ids, k_rope_pos_ids = flashinfer.apply_llama31_rope_pos_ids(
            q,
            k,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_theta=500000.0,
        )
    assert paddle.allclose(
        x=q_rope_pos_ids, y=q_rope, rtol=0.001, atol=0.001
    ).item(), ""
    assert paddle.allclose(
        x=k_rope_pos_ids, y=k_rope, rtol=0.001, atol=0.001
    ).item(), ""


class FlashInferRotaryEmbedding(RotaryEmbedding):
    def forward_cuda(
        self,
        positions: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
        offsets: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        flashinfer.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query, key


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        (64, 64, 32, 8000, True, "bfloat16", "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, "bfloat16", "cuda", 2, 512, 4, 2),
        (64, 32, 2048, 8432, True, "bfloat16", "cuda", 2, 199, 4, 1),
        (64, 64, 32, 8000, False, "bfloat16", "cuda", 32, 32, 1, 1),
        (64, 64, 32, 8000, False, "bfloat16", "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 9231, False, "bfloat16", "cuda", 3, 231, 4, 2),
    ],
)
def test_rope_cos_sin_cache(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: paddle.dtype,
    device: str,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    rope_ref = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )
    rope_flashinfer = FlashInferRotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )
    pos_ids = paddle.arange(end=seq_len).tile(repeat_times=batch_size)
    query = paddle.randn(
        shape=[batch_size * seq_len, num_q_heads * head_size], dtype=dtype
    )
    key = paddle.randn(
        shape=[batch_size * seq_len, num_kv_heads * head_size], dtype=dtype
    )
    query_ref, key_ref = query.clone(), key.clone()
    query_flashinfer, key_flashinfer = query.clone(), key.clone()
    query_ref_out, key_ref_out = rope_ref.forward_native(pos_ids, query_ref, key_ref)
    query_flashinfer_out, key_flashinfer_out = rope_flashinfer.forward_cuda(
        pos_ids, query_flashinfer, key_flashinfer
    )
    assert paddle.allclose(
        x=query_ref_out, y=query_flashinfer_out, atol=0.01, rtol=0.01
    ).item(), ""
    assert paddle.allclose(
        x=key_ref_out, y=key_flashinfer_out, atol=0.01, rtol=0.01
    ).item(), ""


@pytest.mark.parametrize("num_tokens", [1, 19, 128, 199, 899, 2047])
@pytest.mark.parametrize("input_dtype", ["float16", "bfloat16"])
>>>>>>@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_mla_rope_quantize(num_tokens, input_dtype, quant_dtype):
    device = "cuda:0"
    num_qo_heads = 128
    q_in = paddle.randn(shape=[num_tokens, num_qo_heads, 576], dtype=input_dtype)
    k_in = paddle.randn(shape=[num_tokens, 576], dtype=input_dtype)
    pos_ids = paddle.arange(end=num_tokens)
    rope_flashinfer = FlashInferRotaryEmbedding(
        576, 64, 4096, 10000, False, input_dtype, device
    )
    q_out_f16_ref, k_out_f16_ref = rope_flashinfer.forward_native(pos_ids, q_in, k_in)
    q_out_f8_ref, k_out_f8_ref = map(
        lambda x: x.to(quant_dtype), (q_out_f16_ref, k_out_f16_ref)
    )
    q_out = paddle.empty_like(x=q_in, dtype=quant_dtype)
    k_out = paddle.empty_like(x=k_in, dtype=quant_dtype)
    flashinfer.rope.mla_rope_quantize_fp8(
        q_in[..., :64],
        k_in[..., :64],
        q_in[..., 64:],
        k_in[..., 64:],
        rope_flashinfer.cos_sin_cache,
        pos_ids,
        is_neox=False,
        q_rope_out=q_out[..., :64],
        k_rope_out=k_out[..., :64],
        q_nope_out=q_out[..., 64:],
        k_nope_out=k_out[..., 64:],
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
    )
    assert paddle.allclose(
        x=q_out_f8_ref.astype(dtype="float32"),
        y=q_out.astype(dtype="float32"),
        atol=0.01,
        rtol=0.2,
    ).item(), ""
    assert paddle.allclose(
        x=k_out_f8_ref.astype(dtype="float32"),
        y=k_out.astype(dtype="float32"),
        atol=0.01,
        rtol=0.2,
    ).item(), ""


if __name__ == "__main__":
>>>>>>    test_mla_rope_quantize(1, 1, "float16", torch.float8_e4m3fn)
