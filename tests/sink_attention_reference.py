import sys


import einops
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
from typing import Optional


def sink_softmax(logits, sink):
    sink = einops.tile(repeat_times=[sink, "h -> b h m 1"])
    logits = paddle.concat(x=[logits, sink], axis=-1)
    score = paddle.nn.functional.softmax(x=logits, axis=-1)[..., :-1].contiguous()
    return score


def sink_attention_unified(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    sink: paddle.Tensor,
    window_left: int,
    causal: bool,
    sm_scale: float,
    batch_size: Optional[int] = None,
    mode: str = "auto",
    qo_indptr: Optional[paddle.Tensor] = None,
    kv_indptr: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    """
    Unified sink attention implementation supporting prefill, incremental, chunk prefill, and variable-length scenarios.

    Args:
        q: Query tensor. Format depends on mode:
           - Regular Prefill: [total_q_len, num_qo_heads, head_dim] where q_len == kv_len
           - Incremental: [batch_size, num_qo_heads, head_dim] where q_len == 1
           - Chunk Prefill: [total_q_len, num_qo_heads, head_dim] where q_len != kv_len and q_len > 1
           - Variable Length: [total_q_len, num_qo_heads, head_dim] with different q_len per request
        k: Key tensor. Format depends on mode:
           - Regular Prefill: [total_kv_len, num_kv_heads, head_dim]
           - Incremental: [batch_size, kv_len, num_kv_heads, head_dim]
           - Chunk Prefill: [total_kv_len, num_kv_heads, head_dim]
           - Variable Length: [total_kv_len, num_kv_heads, head_dim]
        v: Value tensor, same format as k
        sink: Sink values [num_qo_heads]
        window_left: Sliding window size (-1 for no window)
        causal: Whether to apply causal masking
        sm_scale: Scaling factor for attention
        batch_size: Required for prefill/chunk modes, auto-detected for incremental
        mode: Processing mode:
            - "auto": Auto-detect based on tensor shapes and dimensions
            - "prefill": Regular prefill (q_len == kv_len)
            - "incremental": Incremental generation (q_len == 1)
            - "chunk": Chunk prefill (q_len != kv_len and q_len > 1)
            - "varlen": Variable length sequences within batch
        qo_indptr: Optional[torch.Tensor] - Query sequence length pointers for variable length mode.
                  Shape: [batch_size + 1]. qo_indptr[i+1] - qo_indptr[i] gives the query length for request i.
                  Only used when mode="varlen".
        kv_indptr: Optional[torch.Tensor] - Key/Value sequence length pointers for variable length mode.
                  Shape: [batch_size + 1]. kv_indptr[i+1] - kv_indptr[i] gives the kv length for request i.
                  Only used when mode="varlen".

    Returns:
        Output tensor. Format depends on mode:
        - Regular Prefill: [total_q_len, num_qo_heads, head_dim]
        - Incremental: [batch_size, num_qo_heads, head_dim]
        - Chunk Prefill: [total_q_len, num_qo_heads, head_dim]
        - Variable Length: [total_q_len, num_qo_heads, head_dim]
    """
    if mode == "auto":
        if qo_indptr is not None or kv_indptr is not None:
            mode = "varlen"
        elif len(tuple(q.shape)) == 3 and len(tuple(k.shape)) == 4:
            mode = "incremental"
        elif len(tuple(q.shape)) == 3 and len(tuple(k.shape)) == 3:
            if batch_size is None:
                raise ValueError(
                    "batch_size is required for auto-detection in prefill/chunk modes"
                )
            qo_len = tuple(q.shape)[0] // batch_size
            kv_len = tuple(k.shape)[0] // batch_size
            if qo_len == kv_len:
                mode = "prefill"
            elif qo_len == 1:
                mode = "incremental"
            elif qo_len > 1 and qo_len != kv_len:
                mode = "chunk"
            else:
                raise ValueError(
                    f"Cannot auto-detect mode: qo_len={qo_len}, kv_len={kv_len}"
                )
        else:
            raise ValueError(
                f"Cannot auto-detect mode from tensor shapes: q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
    if mode == "incremental":
        batch_size = tuple(q.shape)[0]
        qo_len = 1
        kv_len = tuple(k.shape)[1]
        num_qo_heads = tuple(q.shape)[1]
        num_kv_heads = tuple(k.shape)[2]
        if num_qo_heads != num_kv_heads:
            k = paddle.repeat_interleave(
                x=k, repeats=num_qo_heads // num_kv_heads, axis=2
            ).contiguous()
            v = paddle.repeat_interleave(
                x=v, repeats=num_qo_heads // num_kv_heads, axis=2
            ).contiguous()
            num_kv_heads = num_qo_heads
        head_dim_qk = tuple(q.shape)[2]
        head_dim_vo = tuple(v.shape)[3]
        logits = (
            paddle.einsum(
                "bhd,blhd->bhl", q.astype(dtype="float32"), k.astype(dtype="float32")
            ).unsqueeze(axis=2)
            * sm_scale
        )
    elif mode in ["prefill", "chunk"]:
        if batch_size is None:
            raise ValueError(f"batch_size is required for {mode} mode")
        qo_len = tuple(q.shape)[0] // batch_size
        kv_len = tuple(k.shape)[0] // batch_size
        num_qo_heads = tuple(q.shape)[1]
        num_kv_heads = tuple(k.shape)[1]
        if num_qo_heads != num_kv_heads:
            k = paddle.repeat_interleave(
                x=k, repeats=num_qo_heads // num_kv_heads, axis=1
            ).contiguous()
            v = paddle.repeat_interleave(
                x=v, repeats=num_qo_heads // num_kv_heads, axis=1
            ).contiguous()
        head_dim_qk = tuple(q.shape)[2]
        head_dim_vo = tuple(v.shape)[2]
        logits = (
            paddle.einsum(
                "bmhd,bnhd->bhmn",
                q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).astype(
                    dtype="float32"
                ),
                k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).astype(
                    dtype="float32"
                ),
            )
            * sm_scale
        )
    elif mode == "varlen":
        if qo_indptr is None or kv_indptr is None:
            raise ValueError("qo_indptr and kv_indptr are required for varlen mode")
        batch_size = tuple(qo_indptr.shape)[0] - 1
        num_qo_heads = tuple(q.shape)[1]
        num_kv_heads = tuple(k.shape)[1]
        head_dim_qk = tuple(q.shape)[2]
        head_dim_vo = tuple(v.shape)[2]
        if num_qo_heads != num_kv_heads:
            k = paddle.repeat_interleave(
                x=k, repeats=num_qo_heads // num_kv_heads, axis=1
            ).contiguous()
            v = paddle.repeat_interleave(
                x=v, repeats=num_qo_heads // num_kv_heads, axis=1
            ).contiguous()
            num_kv_heads = num_qo_heads
        output_list = []
        for i in range(batch_size):
            qo_start, qo_end = qo_indptr[i].item(), qo_indptr[i + 1].item()
            kv_start, kv_end = kv_indptr[i].item(), kv_indptr[i + 1].item()
            q_i = q[qo_start:qo_end]
            k_i = k[kv_start:kv_end]
            v_i = v[kv_start:kv_end]
            qo_len_i = qo_end - qo_start
            kv_len_i = kv_end - kv_start
            logits_i = (
                paddle.einsum(
                    "qhd,khd->hqk",
                    q_i.astype(dtype="float32"),
                    k_i.astype(dtype="float32"),
                ).unsqueeze(axis=0)
                * sm_scale
            )
            if causal:
                row_idx = paddle.arange(dtype="int32", end=qo_len_i)[:, None]
                col_idx = paddle.arange(dtype="int32", end=kv_len_i)[None, :]
                query_positions = kv_len_i - qo_len_i + row_idx
                mask_i = query_positions >= col_idx
                if window_left >= 0:
                    mask_i &= query_positions - window_left <= col_idx
            else:
                mask_i = paddle.ones(shape=[qo_len_i, kv_len_i], dtype="bool")
                if window_left >= 0:
                    row_idx = paddle.arange(dtype="int32", end=qo_len_i)[:, None]
                    col_idx = paddle.arange(dtype="int32", end=kv_len_i)[None, :]
                    query_positions = kv_len_i - qo_len_i + row_idx
                    mask_i = query_positions - window_left <= col_idx
            logits_i = logits_i.masked_fill(
                mask=mask_i.unsqueeze(axis=0).unsqueeze(axis=0) == 0,
                value=float("-inf"),
            )
            p_i = sink_softmax(logits_i, sink)
            o_i = (
                paddle.einsum("bhmn,nhd->bmhd", p_i, v_i.astype(dtype="float32"))
                .contiguous()
                .view(qo_len_i, num_qo_heads, head_dim_vo)
                .to(q)
            )
            output_list.append(o_i)
        o_ref = paddle.concat(x=output_list, axis=0)
        return o_ref
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Supported modes: 'auto', 'prefill', 'incremental', 'chunk', 'varlen'"
        )
    if causal:
        if mode == "incremental":
            mask = paddle.ones(shape=[1, kv_len], dtype="bool")
            if window_left >= 0:
                col_idx = paddle.arange(dtype="int32", end=kv_len)
                mask = kv_len - 1 - window_left <= col_idx
        elif mode == "prefill":
            mask = paddle.arange(start=kv_len - qo_len, end=kv_len).unsqueeze(
                axis=1
            ) >= paddle.arange(start=0, end=kv_len).unsqueeze(axis=0)
            if window_left >= 0:
                row_idx = paddle.arange(dtype="int32", end=qo_len)[:, None]
                col_idx = paddle.arange(dtype="int32", end=kv_len)[None, :]
                mask &= row_idx - window_left <= col_idx
        elif mode == "chunk":
            current_chunk_start = kv_len - qo_len
            row_idx = paddle.arange(dtype="int32", end=qo_len)[:, None]
            col_idx = paddle.arange(dtype="int32", end=kv_len)[None, :]
            abs_row_positions = current_chunk_start + row_idx
            mask = abs_row_positions >= col_idx
            if window_left >= 0:
                mask &= abs_row_positions - window_left <= col_idx
    elif mode == "incremental":
        mask = paddle.ones(shape=[1, kv_len], dtype="bool")
        if window_left >= 0:
            col_idx = paddle.arange(dtype="int32", end=kv_len)
            mask = kv_len - 1 - window_left <= col_idx
    else:
        mask = paddle.ones(shape=[qo_len, kv_len], dtype="bool")
        if window_left >= 0:
            if mode == "chunk":
                current_chunk_start = kv_len - qo_len
                row_idx = paddle.arange(dtype="int32", end=qo_len)[:, None]
                col_idx = paddle.arange(dtype="int32", end=kv_len)[None, :]
                abs_row_positions = current_chunk_start + row_idx
                mask = abs_row_positions - window_left <= col_idx
            else:
                row_idx = paddle.arange(dtype="int32", end=qo_len)[:, None]
                col_idx = paddle.arange(dtype="int32", end=kv_len)[None, :]
                mask = row_idx - window_left <= col_idx
    logits = logits.masked_fill(
        mask=mask.unsqueeze(axis=0).unsqueeze(axis=0) == 0, value=float("-inf")
    )
    p = sink_softmax(logits, sink)
    if mode == "incremental":
        o_ref = (
            paddle.einsum("bhml,blhd->bhd", p, v.astype(dtype="float32"))
            .contiguous()
            .to(q)
        )
    else:
        o_ref = (
            paddle.einsum(
                "bhmn,bnhd->bmhd",
                p,
                v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).astype(
                    dtype="float32"
                ),
            )
            .contiguous()
            .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
            .to(q)
        )
    return o_ref
