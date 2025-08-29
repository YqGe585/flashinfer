import sys


from enum import Enum
from typing import Optional

import paddle
from flashinfer.paddle_utils import *

from ..jit import get_cudnn_fmha_gen_module

try:
    import cudnn

    CUDNN_AVAILABLE = True
except Exception:
    cudnn = None
    CUDNN_AVAILABLE = False
_cudnn_handle = None


def _create_cudnn_handle(stream: paddle.device.Stream):
    global _cudnn_handle
    if _cudnn_handle is None:
        _cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(_cudnn_handle, stream.cuda_stream)
    return _cudnn_handle


class UIDs(Enum):
    RESERVED_INVALID_UID = 0
    Q_UID = 1
    K_UID = 2
    V_UID = 3
    ACTUAL_SEQ_LENS_Q_UID = 100
    ACTUAL_SEQ_LENS_KV_UID = 101
    BLOCK_TABLES_UID = 200
    BLOCK_TABLES_K_UID = 201
    BLOCK_TABLES_V_UID = 202
    RAGGED_Q_UID = 50
    RAGGED_O_UID = 51
    RAGGED_STATS_UID = 52
    RAGGED_K_UID = 53
    RAGGED_V_UID = 54
    O_UID = 1000
    STATS_UID = 1001


def _sdpa_prefill_key_fn(
    q: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    scale: float,
    *,
    max_token_seq_q: Optional[int] = None,
    max_sequence_kv: Optional[int] = None,
    actual_seq_lens_q: Optional[paddle.Tensor] = None,
    actual_seq_lens_kv: paddle.Tensor,
    block_tables: Optional[paddle.Tensor] = None,
    page_size: Optional[int] = None,
    bottom_right_causal_mask: Optional[bool] = None,
    return_lse: Optional[bool] = False,
    batch_offsets_q: Optional[paddle.Tensor] = None,
    batch_offsets_o: Optional[paddle.Tensor] = None,
    batch_offsets_k: Optional[paddle.Tensor] = None,
    batch_offsets_v: Optional[paddle.Tensor] = None,
    batch_offsets_stats: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
    lse: Optional[paddle.Tensor] = None,
):
    graph_b = tuple(actual_seq_lens_q.shape)[0]
    if q.dim() == 3:
        h_qo, d_qk = tuple(q.shape)[1], tuple(q.shape)[2]
    elif q.dim() == 4:
        h_qo, d_qk = tuple(q.shape)[1], tuple(q.shape)[3]
    if v_cache.dim() == 3:
        h_kv, d_vo = tuple(k_cache.shape)[1], tuple(k_cache.shape)[2]
    elif k_cache.dim() == 4:
        h_kv, d_vo = tuple(k_cache.shape)[1], tuple(k_cache.shape)[3]
    if block_tables is not None:
        page_size = tuple(k_cache.shape)[2]
    key = (
        graph_b,
        q.dim(),
        k_cache.dim(),
        max_token_seq_q,
        max_sequence_kv,
        h_qo,
        d_qk,
        h_kv,
        d_vo,
        block_tables is not None,
        return_lse,
        bottom_right_causal_mask,
        page_size,
    )
    return key


if CUDNN_AVAILABLE:

    @cudnn.jit(heur_modes=[cudnn.heur_mode.A])
    @cudnn.graph_cache(key_fn=_sdpa_prefill_key_fn)
    def _build_prefill_graph(
        q: paddle.Tensor,
        k_cache: paddle.Tensor,
        v_cache: paddle.Tensor,
        scale: float,
        *,
        max_token_seq_q: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        actual_seq_lens_q: Optional[paddle.Tensor] = None,
        actual_seq_lens_kv: Optional[paddle.Tensor] = None,
        block_tables: Optional[paddle.Tensor] = None,
        bottom_right_causal_mask: Optional[bool] = True,
        return_lse: Optional[bool] = False,
        batch_offsets_q: Optional[paddle.Tensor] = None,
        batch_offsets_o: Optional[paddle.Tensor] = None,
        batch_offsets_k: Optional[paddle.Tensor] = None,
        batch_offsets_v: Optional[paddle.Tensor] = None,
        batch_offsets_stats: Optional[paddle.Tensor] = None,
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
    ):
        handle = _create_cudnn_handle(
            paddle.device.current_stream(device=device2str(q.place))
        )
        graph_b = tuple(actual_seq_lens_q.shape)[0]
        graph_s_qo = max_token_seq_q
        graph_s_kv = max_sequence_kv
        with cudnn.graph(handle) as (g, _):
            if q.dim() == 3:
                h_qo, d_qk = tuple(q.shape)[1], tuple(q.shape)[2]
            elif q.dim() == 4:
                h_qo, d_qk = tuple(q.shape)[2], tuple(q.shape)[3]
            else:
                raise ValueError(f"Invalid query tensor shape: {tuple(q.shape)}")
            cudnn_q = g.tensor(
                name="q",
                dim=(graph_b, h_qo, graph_s_qo, d_qk),
                stride=(h_qo * d_qk, d_qk, d_qk * h_qo, 1),
                data_type=cudnn.data_type.BFLOAT16,
            )
            if batch_offsets_q is not None:
                ragged_q = g.tensor_like(batch_offsets_q)
                ragged_q.set_uid(UIDs.RAGGED_Q_UID.value)
                cudnn_q.set_ragged_offset(ragged_q)
            if v_cache.dim() == 3:
                assert (
                    block_tables is None
                ), "block_tables needs 4 dimensions of kv cache"
                h_kv, d_vo = tuple(v_cache.shape)[1], tuple(v_cache.shape)[2]
            elif v_cache.dim() == 4:
                h_kv, d_vo = tuple(v_cache.shape)[1], tuple(v_cache.shape)[3]
            else:
                raise ValueError(
                    f"Invalid kv cache tensor shape: {tuple(k_cache.shape)}"
                )
            if k_cache.dim() == 3:
                cudnn_k_cache = g.tensor(
                    name="k_cache",
                    dim=(graph_b, h_kv, graph_s_kv, d_qk),
                    stride=(h_kv * d_qk * graph_s_kv, d_qk, d_qk * h_kv, 1),
                    data_type=cudnn.data_type.BFLOAT16,
                )
                if batch_offsets_k is not None:
                    ragged_k = g.tensor_like(batch_offsets_k)
                    ragged_k.set_uid(UIDs.RAGGED_K_UID.value)
                    cudnn_k_cache.set_ragged_offset(ragged_k)
                cudnn_v_cache = g.tensor(
                    name="v_cache",
                    dim=(graph_b, h_kv, graph_s_kv, d_vo),
                    stride=(h_kv * d_vo * graph_s_kv, d_vo, d_vo * h_kv, 1),
                    data_type=cudnn.data_type.BFLOAT16,
                )
                if batch_offsets_v is not None:
                    ragged_v = g.tensor_like(batch_offsets_v)
                    ragged_v.set_uid(UIDs.RAGGED_V_UID.value)
                    cudnn_v_cache.set_ragged_offset(ragged_v)
            elif k_cache.dim() == 4:
                cudnn_k_cache = g.tensor(
                    name="k_cache",
                    dim=tuple(k_cache.shape),
                    stride=k_cache.get_strides(),
                    data_type=cudnn.data_type.BFLOAT16,
                )
                cudnn_v_cache = g.tensor(
                    name="v_cache",
                    dim=tuple(v_cache.shape),
                    stride=v_cache.get_strides(),
                    data_type=cudnn.data_type.BFLOAT16,
                )
            cudnn_q.set_uid(UIDs.Q_UID.value)
            cudnn_k_cache.set_uid(UIDs.K_UID.value)
            cudnn_v_cache.set_uid(UIDs.V_UID.value)
            if block_tables is not None:
                nd_block_tables = block_tables.reshape(
                    tuple(block_tables.shape)[0], 1, tuple(block_tables.shape)[1], 1
                )
                cudnn_k_block_tables = g.tensor_like(nd_block_tables)
                cudnn_k_block_tables.set_uid(UIDs.BLOCK_TABLES_K_UID.value)
                cudnn_v_block_tables = g.tensor_like(nd_block_tables)
                cudnn_v_block_tables.set_uid(UIDs.BLOCK_TABLES_V_UID.value)
            if actual_seq_lens_q is not None:
                cudnn_actual_seq_lens_q = g.tensor_like(actual_seq_lens_q)
                cudnn_actual_seq_lens_q.set_name("actual_seq_lens_q")
                cudnn_actual_seq_lens_q.set_uid(UIDs.ACTUAL_SEQ_LENS_Q_UID.value)
            if actual_seq_lens_kv is not None:
                cudnn_actual_seq_lens_kv = g.tensor_like(actual_seq_lens_kv)
                cudnn_actual_seq_lens_kv.set_name("actual_seq_lens_kv")
                cudnn_actual_seq_lens_kv.set_uid(UIDs.ACTUAL_SEQ_LENS_KV_UID.value)
            padding_mask = (
                actual_seq_lens_q is not None and actual_seq_lens_kv is not None
            )
            O, Stats = g.sdpa(
                name="sdpa",
                q=cudnn_q,
                k=cudnn_k_cache,
                v=cudnn_v_cache,
                seq_len_q=cudnn_actual_seq_lens_q
                if actual_seq_lens_q is not None
                else None,
                seq_len_kv=cudnn_actual_seq_lens_kv
                if actual_seq_lens_kv is not None
                else None,
                use_padding_mask=padding_mask,
                attn_scale=scale,
                generate_stats=return_lse,
                use_causal_mask_bottom_right=bottom_right_causal_mask,
                paged_attention_k_table=cudnn_k_block_tables
                if block_tables is not None
                else None,
                paged_attention_v_table=cudnn_v_block_tables
                if block_tables is not None
                else None,
                paged_attention_max_seq_len_kv=graph_s_kv
                if block_tables is not None
                else None,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            if batch_offsets_o is not None:
                ragged_o = g.tensor_like(batch_offsets_o)
                ragged_o.set_uid(UIDs.RAGGED_O_UID.value)
                O.set_ragged_offset(ragged_o)
            if batch_offsets_stats is not None:
                ragged_stats = g.tensor_like(batch_offsets_stats)
                ragged_stats.set_uid(UIDs.RAGGED_STATS_UID.value)
                Stats.set_ragged_offset(ragged_stats)
            O.set_uid(UIDs.O_UID.value).set_output(True).set_dim(
                [graph_b, h_qo, graph_s_qo, d_vo]
            ).set_stride(
                [graph_s_qo * d_vo * h_qo, d_vo, d_vo * h_qo, 1]
            ).set_data_type(
                cudnn.data_type.BFLOAT16
            )
            if return_lse:
                Stats.set_uid(UIDs.STATS_UID.value).set_output(
                    return_lse
                ).set_data_type(cudnn.data_type.FLOAT).set_dim(
                    [graph_b, h_qo, graph_s_qo, 1]
                ).set_stride(
                    [graph_s_qo * h_qo, 1, h_qo, 1]
                )
            tensors_to_return = [cudnn_q, cudnn_k_cache, cudnn_v_cache, O]
            if return_lse:
                tensors_to_return.append(Stats)
            if actual_seq_lens_q is not None:
                tensors_to_return.append(cudnn_actual_seq_lens_q)
            if actual_seq_lens_kv is not None:
                tensors_to_return.append(cudnn_actual_seq_lens_kv)
            return g, tensors_to_return


def _batch_prefill_with_kv_cache(
    q: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    scale: float,
    workspace_buffer: paddle.Tensor,
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: paddle.Tensor,
    actual_seq_lens_kv: paddle.Tensor,
    block_tables: Optional[paddle.Tensor] = None,
    causal: bool,
    return_lse: bool,
    batch_offsets_q: Optional[paddle.Tensor] = None,
    batch_offsets_o: Optional[paddle.Tensor] = None,
    batch_offsets_k: Optional[paddle.Tensor] = None,
    batch_offsets_v: Optional[paddle.Tensor] = None,
    batch_offsets_stats: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
    lse: Optional[paddle.Tensor] = None,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    graph, tensors = _build_prefill_graph(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        scale=scale,
        max_token_seq_q=max_token_per_sequence,
        max_sequence_kv=max_sequence_kv,
        actual_seq_lens_q=actual_seq_lens_q,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        bottom_right_causal_mask=causal,
        return_lse=return_lse,
        batch_offsets_q=batch_offsets_q,
        batch_offsets_o=batch_offsets_o,
        batch_offsets_k=batch_offsets_k,
        batch_offsets_v=batch_offsets_v,
        batch_offsets_stats=batch_offsets_stats,
        out=out,
        lse=lse,
    )
    var_map = {
        UIDs.Q_UID.value: q,
        UIDs.K_UID.value: k_cache,
        UIDs.V_UID.value: v_cache,
        UIDs.O_UID.value: out,
    }
    if actual_seq_lens_q is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_Q_UID.value] = actual_seq_lens_q
    if actual_seq_lens_kv is not None:
        var_map[UIDs.ACTUAL_SEQ_LENS_KV_UID.value] = actual_seq_lens_kv
    if batch_offsets_q is not None:
        var_map[UIDs.RAGGED_Q_UID.value] = batch_offsets_q
    if batch_offsets_o is not None:
        var_map[UIDs.RAGGED_O_UID.value] = batch_offsets_o
    if batch_offsets_k is not None:
        var_map[UIDs.RAGGED_K_UID.value] = batch_offsets_k
    if batch_offsets_v is not None:
        var_map[UIDs.RAGGED_V_UID.value] = batch_offsets_v
    if block_tables is not None:
        var_map[UIDs.BLOCK_TABLES_K_UID.value] = block_tables
        var_map[UIDs.BLOCK_TABLES_V_UID.value] = block_tables
    if return_lse:
        var_map[UIDs.STATS_UID.value] = lse
        if batch_offsets_stats is not None:
            var_map[UIDs.RAGGED_STATS_UID.value] = batch_offsets_stats
    handle = _create_cudnn_handle(
        paddle.device.current_stream(device=device2str(q.place))
    )
    graph.execute(var_map, workspace=workspace_buffer, handle=handle)
    if return_lse:
        return out, lse
    else:
        return out, None


def cudnn_batch_prefill_with_kv_cache(
    q: paddle.Tensor,
    k_cache: paddle.Tensor,
    v_cache: paddle.Tensor,
    scale: float,
    workspace_buffer: paddle.Tensor,
    *,
    max_token_per_sequence: int,
    max_sequence_kv: int,
    actual_seq_lens_q: paddle.Tensor,
    actual_seq_lens_kv: paddle.Tensor,
    block_tables: Optional[paddle.Tensor] = None,
    causal: bool,
    return_lse: bool,
    batch_offsets_q: Optional[paddle.Tensor] = None,
    batch_offsets_o: Optional[paddle.Tensor] = None,
    batch_offsets_k: Optional[paddle.Tensor] = None,
    batch_offsets_v: Optional[paddle.Tensor] = None,
    batch_offsets_stats: Optional[paddle.Tensor] = None,
    out: Optional[paddle.Tensor] = None,
    lse: Optional[paddle.Tensor] = None,
    is_cuda_graph_compatible: bool = False,
    backend: Optional[str] = None,
) -> tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """Performs batched prefill attention with paged KV cache using cuDNN.

    Args:
        q: Query tensor of shape (Total number of tokens, num_heads_qo, head_dim)
        k_cache: Key cache tensor of shape   (total_num_pages, num_heads_kv, page_size, head_dim) if paged kv cache is enabled else (Total sequence length of kv, num_heads_kv, d_qk)
        v_cache: Value cache tensor of shape (total_num_pages, num_heads_kv, page_size, head_dim) if paged kv cache is enabled else (Total sequence length of kv, num_heads_kv, d_vo)
        scale: Scaling factor for attention scores, typically 1/sqrt(head_dim)
        workspace_buffer: Workspace buffer for cuDNN operations. Scales with batch size. 128 MB should be sufficient for most cases
        max_token_per_sequence: Maximum number of tokens per query sequence (s_qo_max)
        max_sequence_kv: Maximum number of tokens per key/value sequence (s_kv_max)
        actual_seq_lens_q:  Actual number of tokens per query sequence shape (batch_size,) on cpu or device (cpu if cuda_graph is False)
        actual_seq_lens_kv: Actual sequence lengths for key/values per batch, shape (batch_size,) on CPU or device (cpu if cuda_graph is False)
        block_tables: Page table mapping for KV cache, shape (batch_size, num_pages_per_seq) on GPU
        causal: Whether to apply causal masking
        return_lse: Whether to return log-sum-exp values (must be True)
        out: Optional pre-allocated output tensor
        lse: Optional pre-allocated tensor for log-sum-exp values if return_lse is True else returns None
        is_cuda_graph_compatible: Whether the prefill operation is compatible with CUDA graph
        batch_offsets_q: Optional batch offsets for query tensor of shape (batch_size,) on GPU
        batch_offsets_o: Optional batch offsets for output tensor of shape (batch_size,) on GPU
        batch_offsets_k: Optional batch offsets for key tensor of shape (batch_size,) on GPU
        batch_offsets_v: Optional batch offsets for value tensor of shape (batch_size,) on GPU

    Returns:
        Output tensor of shape (batch_size * seq_len_q, num_heads_qo, head_dim)
        If return_lse is True, also returns log-sum-exp tensor of shape (batch_size, seq_len_q, num_heads_qo)

    Note:
        Query and KV heads can have different sizes (num_heads_qo >= num_heads_kv)
        When using cuda graph, actual_seq_lens_q and actual_seq_lens_kv must be on the same device as q
        Head dimension of query and key must be 128 or 192
        Head dimension of value and output must be 128
    """
    num_tokens = tuple(q.shape)[0]
    num_sequences = tuple(actual_seq_lens_q.shape)[0]
    if q.dim() == 3:
        h_qo, d_qk = tuple(q.shape)[1], tuple(q.shape)[2]
    elif q.dim() == 4:
        h_qo, d_qk = tuple(q.shape)[1], tuple(q.shape)[3]
    if v_cache.dim() == 3:
        d_vo = tuple(v_cache.shape)[2]
    elif v_cache.dim() == 4:
        d_vo = tuple(v_cache.shape)[3]
    if return_lse:
        if lse is None:
            lse = paddle.empty(
                shape=[num_sequences, max_token_per_sequence, h_qo], dtype="float32"
            )
    if lse is not None and tuple(lse.shape) != (
        num_sequences,
        max_token_per_sequence,
        h_qo,
    ):
        raise ValueError(
            "lse must have shape (num_sequences, max_token_per_sequence, h_qo)"
        )
    if out is None:
        out_shape = num_tokens, h_qo, d_vo
        out = paddle.empty(shape=out_shape, dtype=q.dtype)
    if CUDNN_AVAILABLE and backend != "cubin":
        return _batch_prefill_with_kv_cache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            scale=scale,
            workspace_buffer=workspace_buffer,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            causal=causal,
            return_lse=return_lse,
            batch_offsets_q=batch_offsets_q,
            batch_offsets_o=batch_offsets_o,
            batch_offsets_k=batch_offsets_k,
            batch_offsets_v=batch_offsets_v,
            batch_offsets_stats=batch_offsets_stats,
            out=out,
            lse=lse,
        )
    else:
        assert return_lse, "Currently only supports return_lse = True"
        assert (
            d_qk == 192
            and block_tables is None
            or d_qk == 128
            and block_tables is not None
        ), "Currently only supports if d_qk = 192 and block_tables is None or d_qk = 128 and block_tables is not None"
        if max_sequence_kv is None:
            max_sequence_kv = max_token_per_sequence
        actual_seq_lens_q_gpu = actual_seq_lens_q.to(q.place, blocking=not True)
        actual_seq_lens_kv_gpu = actual_seq_lens_kv.to(q.place, blocking=not True)
        run_func = get_cudnn_fmha_gen_module().prefill
        run_func(
            num_sequences,
            max_token_per_sequence,
            max_sequence_kv,
            q,
            k_cache,
            v_cache,
            scale,
            workspace_buffer,
            actual_seq_lens_q,
            actual_seq_lens_kv,
            actual_seq_lens_q_gpu,
            actual_seq_lens_kv_gpu,
            block_tables,
            causal,
            return_lse,
            out,
            lse,
            None,
            None,
            None,
            None,
            is_cuda_graph_compatible,
        )
    return out, lse
