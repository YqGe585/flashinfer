import sys

sys.path.append("/home/flashinfer")
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
import numpy as np
import pytest
import scipy as sp
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


def bsr_attention_ref(q, k, v, indptr, indices, mask_data):
    M = tuple(q.shape)[0]
    N = tuple(k.shape)[0]
    bsr = sp.sparse.bsr_matrix(
        (mask_data.cpu().numpy(), indices.cpu().numpy(), indptr.cpu().numpy()),
        shape=(M, N),
    )
    dense_mask = paddle.to_tensor(data=bsr.toarray(), dtype=bool, place=q.place)
    o = flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, custom_mask=dense_mask)
    return o


def set_seed(seed: int = 42):
    paddle.seed(seed=seed)
    paddle.seed(seed=seed)
    np.random.seed(seed)


@pytest.mark.parametrize("R", [1, 4, 16])
@pytest.mark.parametrize("C", [1, 4, 16])
@pytest.mark.parametrize("M", [64, 128, 256])
@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("mask_inside_block", [True, False])
def test_block_sparse_attention(
    R, C, M, N, num_qo_heads, num_kv_heads, head_dim, mask_inside_block
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    set_seed(33)
    rng = np.random.default_rng()
    MB = M // R
    NB = N // C
    S = sp.sparse.random(MB, NB, density=0.25, random_state=rng).tocsr()
    indptr = paddle.to_tensor(data=S.indptr).to(0)
    indices = paddle.to_tensor(data=S.indices).to(0)
    nnz = S.nnz
    if mask_inside_block:
        data_mask = (paddle.rand(shape=(nnz, R, C)) > 0.5).to(0)
    else:
        data_mask = paddle.full(shape=(nnz, R, C), fill_value=True, dtype=bool)
    q = paddle.randn(shape=(M, num_qo_heads, head_dim), dtype="float16")
    k = paddle.randn(shape=(N, num_kv_heads, head_dim), dtype="float16")
    v = paddle.randn(shape=(N, num_kv_heads, head_dim), dtype="float16")
    o_ref = bsr_attention_ref(q, k, v, indptr, indices, data_mask)
    workspace_buffer = paddle.zeros(shape=128 * 1024 * 1024, dtype="uint8")
    sparse_attention_wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(
        workspace_buffer
    )
    sparse_attention_wrapper.plan(
        indptr,
        indices,
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        mask=data_mask if mask_inside_block else None,
    )
    o = sparse_attention_wrapper.run(q, k, v)
    assert paddle.allclose(x=o_ref, y=o, atol=0.01, rtol=0.001).item(), ""
    o_buffer = paddle.empty_like(x=o)
    sparse_attention_wrapper.run(q, k, v, out=o_buffer)
    assert paddle.allclose(x=o_ref, y=o_buffer, atol=0.01, rtol=0.001).item(), ""


def _ref_attention(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    block_mask_map: paddle.Tensor,
    block_row_sz: paddle.Tensor,
    block_col_sz: paddle.Tensor,
) -> paddle.Tensor:
    def _block_mask_to_element_mask(
        block_mask_map: paddle.Tensor,
        block_row_sz: paddle.Tensor,
        block_col_sz: paddle.Tensor,
    ) -> paddle.Tensor:
        block_row_sz = block_row_sz.to(block_mask_map.place, dtype="int64")
        block_col_sz = block_col_sz.to(block_mask_map.place, dtype="int64")
        expanded_rows = paddle.repeat_interleave(
            x=block_mask_map, repeats=block_row_sz, axis=0
        )
        element_mask = paddle.repeat_interleave(
            x=expanded_rows, repeats=block_col_sz, axis=1
        )
        return element_mask

    dense_mask = _block_mask_to_element_mask(
        block_mask_map, block_row_sz, block_col_sz
    ).to(dtype="bool", device=q.place)
    q = q.transpose(perm=dim2perm(q.ndim, 0, 1)).contiguous()
    k = k.transpose(perm=dim2perm(k.ndim, 0, 1)).contiguous()
    v = v.transpose(perm=dim2perm(v.ndim, 0, 1)).contiguous()
    o = flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, custom_mask=dense_mask)
    o = o.transpose(perm=dim2perm(o.ndim, 0, 1)).contiguous()
    return o


@pytest.mark.parametrize("num_qo_heads", [1, 4, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [256, 4096, 8192])
@pytest.mark.parametrize("num_blocks_row", [10, 20])
@pytest.mark.parametrize("num_blocks_col", [50, 100])
@pytest.mark.parametrize("block_density", [0.2, 0.7, 0.9])
def test_variable_block_sparse_attention_wrapper(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    num_blocks_row: int,
    num_blocks_col: int,
    block_density: float,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if seq_len // num_blocks_row < 1:
        pytest.skip("seq_len must be greater than num_blocks_row")
    if seq_len // num_blocks_col < 1:
        pytest.skip("seq_len must be greater than num_blocks_col")
    set_seed(330)

    def random_partition_batch(
        seq_len: int,
        num_blocks: int,
        bsz: int,
        device: (str | str) = "cpu",
        dtype: paddle.dtype = "int32",
    ) -> paddle.Tensor:
        assert seq_len >= num_blocks
        sizes = paddle.empty(shape=(bsz, num_blocks), dtype=dtype)
        for i in range(bsz):
            cut_pts = paddle.randperm(n=seq_len - 1)[: num_blocks - 1] + 1
            cut_pts, _ = paddle.sort(x=cut_pts), paddle.argsort(x=cut_pts)
            row_sizes = paddle.diff(
                x=paddle.concat(
                    x=(
                        paddle.to_tensor(data=[0], place=device),
                        cut_pts,
                        paddle.to_tensor(data=[seq_len], place=device),
                    )
                )
            )
            sizes[i] = row_sizes
        assert sizes._min() >= 1
        assert sizes._max() <= seq_len
        assert paddle.all(x=sizes.sum(axis=-1) == seq_len)
        return sizes.to(device=device)

    def _test_variable_block_sparse_attention(
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_mask_map: paddle.Tensor,
        block_row_sz: paddle.Tensor,
        block_col_sz: paddle.Tensor,
        device: str = "cuda:0",
        dtype: paddle.dtype = "float16",
    ):
        qo_len = block_row_sz.sum(axis=1)[0].item()
        kv_len = block_col_sz.sum(axis=1)[0].item()
        assert paddle.all(x=block_col_sz.sum(axis=1) == block_col_sz.sum(axis=1)[0])
        assert paddle.all(x=block_row_sz.sum(axis=1) == block_row_sz.sum(axis=1)[0])
        q = paddle.randn(shape=[num_qo_heads, qo_len, head_dim], dtype=dtype)
        k = paddle.randn(shape=[num_kv_heads, kv_len, head_dim], dtype=dtype)
        v = paddle.randn(shape=[num_kv_heads, kv_len, head_dim], dtype=dtype)
        float_workspace_buffer = paddle.empty(shape=128 * 1024 * 1024)
        wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
            float_workspace_buffer, backend="auto"
        )
        wrapper.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_data_type=dtype,
        )
        o: paddle.Tensor = wrapper.run(q, k, v)
        o = o.reshape(num_kv_heads, -1, *tuple(o.shape)[-2:])
        q = q.reshape(num_kv_heads, -1, *tuple(q.shape)[-2:])
        for kv_head_idx in range(num_kv_heads):
            o_ref = _ref_attention(
                q[kv_head_idx],
                k[kv_head_idx : kv_head_idx + 1, :, :],
                v[kv_head_idx : kv_head_idx + 1, :, :],
                block_mask_map[kv_head_idx],
                block_row_sz[kv_head_idx],
                block_col_sz[kv_head_idx],
            )
            assert paddle.allclose(
                x=o[kv_head_idx], y=o_ref, atol=0.01, rtol=0.01
            ).item(), ""

    block_row_sz = random_partition_batch(
        seq_len, num_blocks_row, num_kv_heads, device="cuda:0"
    )
    block_col_sz = random_partition_batch(
        seq_len, num_blocks_col, num_kv_heads, device="cuda:0"
    )
    block_mask_map = (
        paddle.rand(shape=[num_kv_heads, num_blocks_row, num_blocks_col])
        > block_density
    ).to(device="gpu:0")
    _test_variable_block_sparse_attention(
        num_qo_heads, num_kv_heads, head_dim, block_mask_map, block_row_sz, block_col_sz
    )


if __name__ == "__main__":
    for seq_len in [16 * 1024, 32 * 1024, 40 * 1024, 48 * 1024, 64 * 1024]:
        test_block_sparse_attention(128, 128, seq_len, seq_len, 1, 1, 128, False)
