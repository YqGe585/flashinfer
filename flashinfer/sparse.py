import sys


import einops
import paddle
from flashinfer.paddle_utils import *

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
from typing import Optional, Tuple, Union

from .decode import get_batch_decode_module
from .page import block_sparse_indices_to_vector_sparse_offsets
from .prefill import _compute_page_mask_indptr, get_batch_prefill_module
from .quantization import segment_packbits
from .utils import (MaskMode, PosEncodingMode, TensorLayout,
                    _check_pos_encoding_mode, _get_cache_alibi_slopes_buf,
                    canonicalize_torch_dtype, check_shape_dtype_device,
                    determine_attention_backend, device_support_pdl, is_float8)


def convert_bsr_mask_layout(
    mask: paddle.Tensor, indptr: paddle.Tensor
) -> paddle.Tensor:
    """Convert mask from BSR data layout to flashinfer's flattened mask layout.

    Parameters
    ----------
    mask : torch.Tensor
        A boolean mask tensor with shape ``(nnz, R, C)``.
    indptr : torch.Tensor
        The indptr tensor in BSR format.

    Returns
    -------
    flattened_mask : torch.Tensor
        A flattenedd mask tensor with shape ``(nnz * R * C,)``.
    """
    nnz, R, C = tuple(mask.shape)
    MB = len(indptr) - 1
    mask_flashinfer = paddle.empty(shape=(nnz * R * C,), dtype=mask.dtype)
    for i in range(MB):
        mask_flashinfer[indptr[i] * R * C : indptr[i + 1] * R * C] = (
            mask[indptr[i] : indptr[i + 1]]
            .transpose(perm=dim2perm(mask[indptr[i] : indptr[i + 1]].ndim, 0, 1))
            .reshape(-1)
        )
    return mask_flashinfer


class BlockSparseAttentionWrapper:
    """Wrapper class for attention computation with a block-sparse matrix as attention mask.
    The definition of block sparse matrix can be found at
    `bsr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html>`_
    in SciPy.

    This API supports any block size ``(R, C)``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 32
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)
    >>> # sparse mask: [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    >>> M = 3
    >>> N = 3
    >>> indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda:0")
    >>> indices = torch.tensor([2, 0, 2, 1, 2], dtype=torch.int32, device="cuda:0")
    >>> bsr_wrapper.plan(
    ...     indptr,
    ...     indices,
    ...     M,
    ...     N,
    ...     1, # R(block_rows)=1
    ...     1, # C(block_columns)=1
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = bsr_wrapper.run(q, k, v)
    >>> # use dense implementation with attention mask for comparison
    >>> mask = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.bool, device="cuda:0")
    >>> o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_ref)
    True
    """

    def __init__(
        self, float_workspace_buffer: paddle.Tensor, backend: str = "auto"
    ) -> None:
        """Constructs of :class:`BlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.place
        self._int_workspace_buffer = paddle.empty(
            shape=(8 * 1024 * 1024,), dtype="uint8"
        )
        if backend in ["fa3", "auto"]:
            self._vector_sparse_indices_buffer = paddle.empty(
                shape=(128 * 1024 * 1024,), dtype="int32"
            )
            self._vector_sparse_indptr_buffer = paddle.empty(
                shape=(32768,), dtype="int32"
            )
        self._kv_lens_buffer = paddle.empty(shape=(32768,), dtype="int32")
        self._pin_memory_int_workspace_buffer = paddle.empty(
            shape=tuple(self._int_workspace_buffer.shape), dtype="uint8"
        ).pin_memory()
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[paddle.Tensor] = None
        self._paged_kv_indptr_buf: Optional[paddle.Tensor] = None
        self._paged_kv_indices_buf: Optional[paddle.Tensor] = None
        self._paged_kv_last_page_len: Optional[paddle.Tensor] = None
        self._packed_mask_buf: Optional[paddle.Tensor] = None
        self._mask_indptr_buf: Optional[paddle.Tensor] = None
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self._backend = backend

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: paddle.Tensor,
        int_workspace_buffer: paddle.Tensor,
        vector_sparse_indices_buffer: Optional[paddle.Tensor] = None,
        vector_sparse_indptr_buffer: Optional[paddle.Tensor] = None,
    ) -> None:
        """Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = paddle.empty(
            shape=tuple(self._int_workspace_buffer.shape),
            dtype=self._int_workspace_buffer.dtype,
        ).pin_memory()
        if vector_sparse_indices_buffer is not None:
            self._vector_sparse_indices_buffer = vector_sparse_indices_buffer
        if vector_sparse_indptr_buffer is not None:
            self._vector_sparse_indptr_buffer = vector_sparse_indptr_buffer

    def plan(
        self,
        indptr: paddle.Tensor,
        indices: paddle.Tensor,
        M: int,
        N: int,
        R: int,
        C: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mask: Optional[paddle.Tensor] = None,
        packed_mask: Optional[paddle.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, paddle.dtype] = "float16",
        kv_data_type: Optional[Union[str, paddle.dtype]] = None,
        o_data_type: Union[str, paddle.dtype] = "float16",
        non_blocking: bool = True,
    ) -> None:
        """Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        indptr : torch.Tensor
            The block index pointer of the block-sparse matrix on row dimension, shape ``(MB + 1,)``,
            where ``MB`` is the number of blocks in the row dimension.
        indices: torch.Tensor
            The block indices of the block-sparse matrix on column dimension, shape ``(nnz,)``, where
            ``nnz`` is the number of non-zero blocks. The elements in ``indices`` array should be less then ``NB``:
            the number of blocks in the column dimension.
        M : int
            The number of rows of the block-sparse matrix, ``MB = ceil_div(M, R)``.
        N : int
            The number of columns of the block-sparse matrix, ``NB = N // C``, ``N`` should be divisible by ``C``.
        R : int
            The number of rows in each block.
        C : int
            The number of columns in each block.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor.
        head_dim : int
            The dimension of each head.
        mask : torch.Tensor, optional
            The mask tensor with shape ``(nnz, R, C,)``, where nnz is the number of non-zero blocks.
            If every block is full, then we don't need to provide the mask tensor.
        packed_mask : torch.Tensor, optional
            The 1D packed mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\\texttt{logits_soft_cap} \\times \\mathrm{tanh}(x / \\texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : str, optional
            The data type of the query tensor.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        o_data_type : str, optional
            The data type of the output tensor. Default is ``half``. As output dtype cannot
            be inferred by input dtype in quantization
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = canonicalize_torch_dtype(o_data_type)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        num_blocks_row = len(indptr) - 1
        qo_indptr_host = R * paddle.arange(dtype="int32", end=num_blocks_row + 1)
        qo_indptr_host[-1] = M
        qo_indptr = qo_indptr_host.to(indptr.place, blocking=not non_blocking)
        if indices._max().item() * C > N:
            raise ValueError("indices out of bound")
        last_block_len = paddle.full(
            shape=(num_blocks_row,), fill_value=C, dtype="int32"
        )
        if mask is not None or packed_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr, indptr, last_block_len, C
            )
        if packed_mask is None and mask is not None:
            mask = convert_bsr_mask_layout(mask, indptr)
            packed_mask, mask_indptr = segment_packbits(
                mask.contiguous().view(-1), mask_indptr, bitorder="little"
            )
        self._qo_indptr = qo_indptr.to(self.device, blocking=not non_blocking)
        self._paged_kv_indptr_buf = indptr.to(self.device, blocking=not non_blocking)
        self._paged_kv_indices_buf = indices.to(self.device, blocking=not non_blocking)
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, blocking=not non_blocking
        )
        if packed_mask is not None:
            self._packed_mask_buf = packed_mask.to(
                self.device, blocking=not non_blocking
            )
            self._mask_indptr_buf = mask_indptr.to(
                self.device, blocking=not non_blocking
            )
            mask_mode = MaskMode.CUSTOM.value
        else:
            self._packed_mask_buf = None
            self._mask_indptr_buf = None
            mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._mask_mode = mask_mode
        self.M = M
        self.N = N
        self.R = R
        self.C = C
        kv_indptr_host = indptr.to("cpu")
        if (
            R * (num_qo_heads // num_kv_heads) < 4
            and mask_mode != MaskMode.CUSTOM.value
>>>>>>            and q_data_type not in [paddle.float8_e4m3fn, paddle.float8_e5m2]
        ):
            self._use_tensor_cores = False
            self._cached_module = get_batch_decode_module(
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,
                logits_soft_cap > 0,
            )
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                kv_indptr_host,
                num_blocks_row,
                num_qo_heads,
                num_kv_heads,
                C,
                False,
                -1,
                logits_soft_cap,
                head_dim,
                head_dim,
                paddle.empty(shape=[0], dtype=q_data_type),
                paddle.empty(shape=[0], dtype=kv_data_type),
            )
        else:
            self._use_tensor_cores = True
            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    mask_mode == MaskMode.CUSTOM.value,
                    q_data_type,
                    kv_data_type,
                )
            get_module_args = (
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,
                logits_soft_cap > 0,
                use_fp16_qk_reduction,
            )
            self._cached_module = get_batch_prefill_module(
                self._backend, *get_module_args
            )
            kv_lens_arr_host = (kv_indptr_host[1:] - kv_indptr_host[:-1]) * self.C
            paddle.assign(
                kv_lens_arr_host, output=self._kv_lens_buffer[: len(kv_lens_arr_host)]
            )
            if self._backend == "fa3":
                if self.C != 1:
                    vector_sparse_indptr_host = paddle.concat(
                        x=[
                            paddle.to_tensor(data=[0], dtype="int32"),
                            paddle.cumsum(x=kv_lens_arr_host, axis=0, dtype="int32"),
                        ],
                        axis=0,
                    )
                    paddle.assign(
                        vector_sparse_indptr_host,
                        output=self._vector_sparse_indptr_buffer[
                            : len(vector_sparse_indptr_host)
                        ],
                    )
                    kv_indptr_host = vector_sparse_indptr_host
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                kv_indptr_host,
                kv_lens_arr_host,
                M,
                num_blocks_row,
                num_qo_heads,
                num_kv_heads,
                self.C,
                False,
                head_dim,
                head_dim,
                causal,
            )
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        scale_q: Optional[paddle.Tensor] = None,
        scale_k: Optional[paddle.Tensor] = None,
        scale_v: Optional[paddle.Tensor] = None,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> paddle.Tensor:
        """Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v, scale_q, scale_k, scale_v)

    def run(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        scale_q: Optional[paddle.Tensor] = None,
        scale_k: Optional[paddle.Tensor] = None,
        scale_v: Optional[paddle.Tensor] = None,
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(M, num_qo_heads, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(N, num_kv_heads, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(N, num_kv_heads, head_dim)``.
        scale_q : Optional[torch.Tensor]
            The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_k : Optional[torch.Tensor]
            The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_v : Optional[torch.Tensor]
            The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.place)
        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 10000.0
        k = k.reshape(-1, self.C, *tuple(k.shape)[-2:])
        v = v.reshape(-1, self.C, *tuple(v.shape)[-2:])
        stride_block = k.get_strides()[0]
        stride_n = k.get_strides()[1]
        if return_lse:
            if lse is None:
                lse = paddle.empty(shape=(q.shape[0], q.shape[1]), dtype="float32")
            else:
                check_shape_dtype_device(
                    lse, (q.shape[0], q.shape[1]), "float32", q.place, "lse"
                )
        if out is None:
            out = paddle.empty_like(x=q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, tuple(q.shape), self._o_dtype, q.place, "out")
        if is_float8(q):
            assert q.dtype == k.dtype == v.dtype
            assert tuple(q.shape)[-1] == tuple(k.shape)[-1] == tuple(v.shape)[-1]
            assert self._backend == "fa3" and self._use_tensor_cores
            if scale_q is None:
                scale_q = paddle.ones(shape=tuple(q.shape)[1], dtype="float32")
            if scale_k is None:
                scale_k = paddle.ones(shape=tuple(k.shape)[1], dtype="float32")
            if scale_v is None:
                scale_v = paddle.ones(shape=tuple(v.shape)[1], dtype="float32")
        if self._use_tensor_cores:
            if self._backend == "fa3":
                if (
                    self._vector_sparse_indices_buffer.size
                    <= self._paged_kv_indices_buf.size * self.C
                ):
                    raise ValueError(
                        "_vector_sparse_indices_buffer is not large enough. Please increase the size."
                    )
                sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                    self._paged_kv_indices_buf,
                    self._paged_kv_indptr_buf,
                    self._vector_sparse_indices_buffer,
                    self._vector_sparse_indptr_buffer,
                    self._kv_lens_buffer,
                    stride_block // stride_n,
                    1,
                    self.C,
                )
                sparse_indptr = self._vector_sparse_indptr_buffer
            else:
                sparse_indices = self._paged_kv_indices_buf
                sparse_indptr = self._paged_kv_indptr_buf
            self._cached_module.paged_run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._qo_indptr,
                sparse_indptr,
                sparse_indices,
                self._paged_kv_last_page_len,
                out,
                lse,
                self._mask_mode,
                TensorLayout[self._kv_layout].value,
                -1,
                enable_pdl,
                self._packed_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(tuple(q.shape)[1], self.device),
                None,
                None,
                None,
                logits_soft_cap,
                sm_scale,
                scale_q,
                scale_k,
                scale_v,
                rope_scale,
                rope_theta,
                0,
            )
        else:
            self._cached_module.run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                -1,
                enable_pdl,
                _get_cache_alibi_slopes_buf(tuple(q.shape)[1], self.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
            )
        return (out, lse) if return_lse else out

    def end_forward(self) -> None:
        """Warning: This method is deprecated and has no effect."""
        pass


class VariableBlockSparseAttentionWrapper:
    """Wrapper class for attention computation with a block-sparse matrix as attention mask.
    This API supports variable block sizes provided by ``block_row_sz`` and ``block_col_sz``.
    Besides, each ``kv_head_idx`` can specify its own sparse patterns without using the same mask.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 1
    >>> num_kv_heads = 1
    >>> head_dim = 128
    >>> seq_len = 6 # This corresponds to the `block_row_sz` and `block_col_sz`
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.VariableBlockSparseAttentionWrapper(workspace_buffer)
    >>> block_mask_map = torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], dtype=torch.bool, device="cuda:0")
    >>> block_row_sz = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda:0")
    >>> block_col_sz = torch.tensor([[3, 1, 2]], dtype=torch.int32, device="cuda:0")
    >>> wrapper.plan(
    ...     block_mask_map,
    ...     block_row_sz,
    ...     block_col_sz,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((num_qo_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = wrapper.run(q, k, v)
    """

    def __init__(
        self, float_workspace_buffer: paddle.Tensor, backend: str = "auto"
    ) -> None:
        """Constructs of :class:`VariableBlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.place
        self._int_workspace_buffer = paddle.empty(
            shape=(8 * 1024 * 1024,), dtype="uint8"
        )
        if backend in ["fa3", "auto"]:
            self._vector_sparse_indices_buffer = paddle.empty(
                shape=(128 * 1024 * 1024,), dtype="int32"
            )
            self._vector_sparse_indptr_buffer = paddle.empty(
                shape=(32768,), dtype="int32"
            )
        self._kv_lens_buffer = paddle.empty(shape=(32768,), dtype="int32")
        self._pin_memory_int_workspace_buffer = paddle.empty(
            shape=tuple(self._int_workspace_buffer.shape), dtype="uint8"
        ).pin_memory()
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[paddle.Tensor] = None
        self._paged_kv_indptr_buf: Optional[paddle.Tensor] = None
        self._paged_kv_indices_buf: Optional[paddle.Tensor] = None
        self._paged_kv_last_page_len: Optional[paddle.Tensor] = None
        self._backend = backend

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: paddle.Tensor,
        int_workspace_buffer: paddle.Tensor,
        vector_sparse_indices_buffer: Optional[paddle.Tensor] = None,
        vector_sparse_indptr_buffer: Optional[paddle.Tensor] = None,
    ) -> None:
        """Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = paddle.empty(
            shape=tuple(self._int_workspace_buffer.shape),
            dtype=self._int_workspace_buffer.dtype,
        ).pin_memory()
        if vector_sparse_indices_buffer is not None:
            self._vector_sparse_indices_buffer = vector_sparse_indices_buffer
        if vector_sparse_indptr_buffer is not None:
            self._vector_sparse_indptr_buffer = vector_sparse_indptr_buffer

    def plan(
        self,
        block_mask_map: paddle.Tensor,
        block_row_sz: paddle.Tensor,
        block_col_sz: paddle.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        q_data_type: Union[str, paddle.dtype] = "float16",
        kv_data_type: Optional[Union[str, paddle.dtype]] = None,
    ) -> None:
        """Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        block_mask_map : torch.Tensor
            The block mask map (boolean), shape ``(num_kv_heads, MB, NB)``, where ``MB`` is the number of blocks in the row dimension,
            ``NB`` is the number of blocks in the column dimension.
        block_row_sz : torch.Tensor
            The block row size, shape ``(num_kv_heads, MB,)``.
        block_col_sz : torch.Tensor
            The block column size, shape ``(num_kv_heads, NB,)``.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor. Note that a group of ``qo_heads`` shares the same sparse pattern of ``kv_heads``.
        head_dim : int
            The dimension of each head.
        causal : bool
            Whether to apply causal mask to the attention matrix.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\\texttt{logits_soft_cap} \\times \\mathrm{tanh}(x / \\texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.


        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = q_data_type
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        num_blocks_row = tuple(block_row_sz.shape)[-1]
        num_blocks_col = tuple(block_col_sz.shape)[-1]
        qo_indptr = paddle.concat(
            x=[
                paddle.zeros(shape=[1], dtype="int32"),
                paddle.cumsum(x=block_row_sz.flatten(), axis=0, dtype="int32"),
            ],
            axis=0,
        )
        qo_indptr_host = qo_indptr.to("cpu", blocking=not non_blocking)
        last_block_len = paddle.full(
            shape=(num_blocks_row * num_kv_heads,), fill_value=1, dtype="int32"
        )

        def _block_mask_map_to_expanded_indices(
            block_mask_map: paddle.Tensor, block_col_sz: paddle.Tensor
        ) -> Tuple[paddle.Tensor, paddle.Tensor]:
            """
            Args:
                block_mask_map:  bool/int  [num_kv_heads, num_blocks_row, num_blocks_col]
                block_col_sz:    int32/64  [num_kv_heads, num_blocks_col]
            Returns:
                kv_indptr:  [H*R + 1]  int32  —  CSR indptr
                kv_indices: [nnz]      int32  —  token indices per (head, row)
            """
            device = block_mask_map.place
            dtype_i = "int32"
            row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(axis=-1)
            kv_indptr = paddle.concat(
                x=[
                    paddle.zeros(shape=[1], dtype=dtype_i),
                    paddle.cumsum(x=row_lengths.flatten(), axis=0),
                ],
                axis=0,
            )
            col_offset = (
                paddle.cumsum(x=block_col_sz.to(dtype_i), axis=1) - block_col_sz
            )
            head_len = block_col_sz.sum(axis=1, dtype=dtype_i)
            head_offset = paddle.cumsum(x=head_len, axis=0) - head_len
            h_idx, r_idx, c_idx = block_mask_map.nonzero(as_tuple=True)
            lengths = block_col_sz[h_idx, c_idx].to(dtype_i)
            base = head_offset[h_idx] + col_offset[h_idx, c_idx]
            cum = paddle.cumsum(x=lengths, axis=0)
            starts = paddle.repeat_interleave(x=cum - lengths, repeats=lengths)
            offsets_within = paddle.arange(end=cum[-1]) - starts
            kv_indices = (
                paddle.repeat_interleave(x=base, repeats=lengths) + offsets_within
            )
            return kv_indptr.to(dtype=dtype_i, device=device), kv_indices.to(
                dtype=dtype_i, device=device
            )

        kv_indptr, kv_indices = _block_mask_map_to_expanded_indices(
            block_mask_map, block_col_sz
        )
        kv_indptr_host = kv_indptr.to("cpu", blocking=not non_blocking)
        kv_indices_host = kv_indices.to("cpu", blocking=not non_blocking)
        self._qo_indptr = qo_indptr.to(self.device, blocking=not non_blocking)
        self._paged_kv_indptr_buf = kv_indptr.to(self.device, blocking=not non_blocking)
        self._paged_kv_indices_buf = kv_indices.to(
            self.device, blocking=not non_blocking
        )
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, blocking=not non_blocking
        )
        paddle.device.synchronize()
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        assert (
            num_qo_heads % num_kv_heads == 0
        ), "num_qo_heads must be a multiple of num_kv_heads"
        assert num_blocks_row * num_kv_heads + 1 == tuple(kv_indptr_host.shape)[0]
        assert (
            kv_indptr_host[-1].item() == tuple(kv_indices_host.shape)[0]
        ), f"{kv_indptr_host[-1].item()} != {tuple(kv_indices_host.shape)[0]}"
        assert num_kv_heads == tuple(block_mask_map.shape)[0]
        assert num_kv_heads == tuple(block_row_sz.shape)[0]
        assert num_kv_heads == tuple(block_col_sz.shape)[0]
        assert num_blocks_row == tuple(block_mask_map.shape)[1]
        assert num_blocks_col == tuple(block_mask_map.shape)[2]
        if self._backend == "auto":
            self._backend = determine_attention_backend(
                self.device,
                PosEncodingMode[pos_encoding_mode].value,
                use_fp16_qk_reduction,
                self._mask_mode == MaskMode.CUSTOM.value,
                q_data_type,
                kv_data_type,
            )
        get_module_args = (
            q_data_type,
            kv_data_type,
            self._o_dtype,
            kv_indptr_host.dtype,
            head_dim,
            head_dim,
            PosEncodingMode[pos_encoding_mode].value,
            False,
            logits_soft_cap > 0,
            use_fp16_qk_reduction,
        )
        self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)
        kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]
        paddle.assign(
            kv_lens_arr_host, output=self._kv_lens_buffer[: len(kv_lens_arr_host)]
        )
        if self._backend == "fa3":
            if self._vector_sparse_indptr_buffer.size <= kv_indptr.size:
                raise ValueError(
                    "_vector_sparse_indptr_buffer is not large enough. Please increase the buffer size."
                )
            paddle.assign(
                kv_indptr, output=self._vector_sparse_indptr_buffer[: len(kv_indptr)]
            )
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_lens_arr_host,
            qo_indptr_host[-1].item(),
            num_blocks_row * num_kv_heads,
            num_qo_heads // num_kv_heads,
            1,
            1,
            False,
            head_dim,
            head_dim,
            causal,
        )
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._num_kv_heads = num_kv_heads
        self._gqa_group_size = num_qo_heads // num_kv_heads

    def forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> paddle.Tensor:
        """Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v)

    def run(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(num_qo_heads, qo_len, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(num_kv_heads, kv_len, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(num_kv_heads, kv_len, head_dim)``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.place)
        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 10000.0
        q = einops.rearrange(
            q,
            "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()
        k = einops.rearrange(
            k, "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim"
        ).contiguous()
        v = einops.rearrange(
            v, "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim"
        ).contiguous()
        stride_block = k.get_strides()[0]
        stride_n = k.get_strides()[1]
        if return_lse:
            if lse is None:
                lse = paddle.empty(shape=(q.shape[0], q.shape[1]), dtype="float32")
            else:
                check_shape_dtype_device(
                    lse, (q.shape[0], q.shape[1]), "float32", q.place, "lse"
                )
        if out is None:
            out = paddle.empty_like(x=q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, tuple(q.shape), self._o_dtype, q.place, "out")
        if self._backend == "fa3":
            if (
                self._vector_sparse_indices_buffer.size
                <= self._paged_kv_indices_buf.size
            ):
                raise ValueError(
                    "_vector_sparse_indices_buffer is not large enough. Please increase the buffer size."
                )
            sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                self._paged_kv_indices_buf,
                self._paged_kv_indptr_buf,
                self._vector_sparse_indices_buffer,
                self._vector_sparse_indptr_buffer,
                self._kv_lens_buffer,
                stride_block // stride_n,
                1,
                1,
            )
            sparse_indptr = self._vector_sparse_indptr_buffer
        else:
            sparse_indices = self._paged_kv_indices_buf
            sparse_indptr = self._paged_kv_indptr_buf
        self._cached_module.paged_run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr,
            sparse_indptr,
            sparse_indices,
            self._paged_kv_last_page_len,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            -1,
            enable_pdl,
            None,
            None,
            None,
            None,
            None,
            None,
            logits_soft_cap,
            sm_scale,
            None,
            None,
            None,
            rope_scale,
            rope_theta,
            0,
        )
        out = einops.rearrange(
            out,
            "(num_kv_heads qo_len) gqa_group_size head_dim -> (num_kv_heads gqa_group_size) qo_len head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()
        if return_lse:
            lse = einops.rearrange(
                lse,
                "(num_kv_heads qo_len) gqa_group_size -> (num_kv_heads gqa_group_size) qo_len",
                num_kv_heads=self._num_kv_heads,
            ).contiguous()
        return (out, lse) if return_lse else out
