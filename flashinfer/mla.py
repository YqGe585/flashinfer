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
import functools
from typing import Literal, Optional, Tuple, Union, overload

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_batch_mla_module, gen_jit_spec, sm100a_nvcc_flags
from .utils import MaskMode, check_shape_dtype_device, determine_mla_backend


def _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table):
    if q_nope_pe.ndim != 3:
        raise ValueError(f"Expected q_nope_pe.ndim == 3, got {q_nope_pe.ndim}")
    if ckv_kpe_cache.ndim != 3:
        raise ValueError(f"Expected ckv_kpe_cache.ndim == 3, got {ckv_kpe_cache.ndim}")
    if kv_len.ndim != 1:
        raise ValueError(f"Expected kv_len.ndim == 1, got {kv_len.ndim}")
    if page_table.ndim != 2:
        raise ValueError(f"Expected page_table.ndim == 2, got {page_table.ndim}")
    B_q, H, D_q = tuple(q_nope_pe.shape)
    D_ckv = tuple(ckv_kpe_cache.shape)[2]
    if H != 128:
        raise ValueError(f"Expected 128 heads for q_nope_pe, got {H}")
    if D_q != D_ckv or D_q != 576:
        raise ValueError(
            f"Expected head dim 576 for q_nope_pe and ckv_kpe_cache, got {D_q} and {D_ckv}"
        )
    B_block_table, block_num = tuple(page_table.shape)
    block_size = tuple(ckv_kpe_cache.shape)[1]
    if B_q != B_block_table:
        raise ValueError(
            f"Expected batch size {B_q} for q_nope_pe and block_table, got {B_q} and {B_block_table}"
        )
    if block_num % (128 / block_size) != 0:
        raise ValueError(
            f"Expected block_num % (128 / block_size) == 0, got block_num={block_num!r} and block_size={block_size!r}"
        )


def gen_mla_module() -> JitSpec:
    return gen_jit_spec(
        "mla",
        [
            jit_env.FLASHINFER_CSRC_DIR / "cutlass_mla.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_mla_ops.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


@functools.cache
def get_mla_module():
    return gen_mla_module().build_and_load()


@functools.cache
def get_batch_mla_module(backend, *args):
    return gen_batch_mla_module(backend, *args).build_and_load()


class BatchMLAPagedAttentionWrapper:
    """Wrapper class for MLA (`Multi-head Latent Attention <https://arxiv.org/abs/2405.04434>`_)
    PagedAttention on DeepSeek models. This kernel can be used in decode, and incremental prefill
    and should be used together with `Matrix Absorption trick
    <https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md>`_:
    where :math:`W_{UQ}` is absorbed with :math:`W_{UK}`, and :math:`W_{UV}` is
    absorbed with :math:`W_{O}`.
    For MLA attention without Matrix Absorption (``head_dim_qk=192`` and ``head_dim_vo=128``, which is
    used in prefilling self-attention stage), please use
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`.

    More information about The Paged KV-Cache layout in MLA is explained in our tutorial
    :ref:`MLA Page Layout <mla-page-layout>`.

    For more details about the MLA computation, Matrix Absorption and FlashInfer's MLA implementation,
    please refer to our `blog post <http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html>`_.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_local_heads = 128
    >>> batch_size = 114
    >>> head_dim_ckv = 512
    >>> head_dim_kpe = 64
    >>> page_size = 1
    >>> mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    ...     torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    ...     backend="fa2"
    ... )
    >>> q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    >>> kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
    >>> kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
    >>> kv_indices = torch.arange(0, batch_size * 999).to(0).int()
    >>> q_nope = torch.randn(
    ...     batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> q_pe = torch.zeros(
    ...     batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> ckv = torch.randn(
    ...     batch_size * 999, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> kpe = torch.zeros(
    ...     batch_size * 999, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    >>> mla_wrapper.plan(
    ...     q_indptr,
    ...     kv_indptr,
    ...     kv_indices,
    ...     kv_lens,
    ...     num_local_heads,
    ...     head_dim_ckv,
    ...     head_dim_kpe,
    ...     page_size,
    ...     False,  # causal
    ...     sm_scale,
    ...     q_nope.dtype,
    ...     ckv.dtype,
    ... )
    >>> o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    >>> o.shape
    torch.Size([114, 128, 512])
    """

    def __init__(
        self,
        float_workspace_buffer: paddle.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[paddle.Tensor] = None,
        kv_indptr: Optional[paddle.Tensor] = None,
        kv_indices: Optional[paddle.Tensor] = None,
        kv_len_arr: Optional[paddle.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        """Constructor for BatchMLAPagedAttentionWrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors.
        use_cuda_graph : bool, optional
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.
        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indices`` array.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_len_arr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_len_arr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability. If ``cutlass`` is provided, the MLA
            kernels will be generated by CUTLASS and only float_workspace_buffer is required and
            other arguments are ignored.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.place
        if backend == "cutlass":
            self._backend = backend
            return
        self._int_workspace_buffer = paddle.empty(
            shape=(8 * 1024 * 1024,), dtype="uint8"
        )
        self._pin_memory_int_workspace_buffer = paddle.empty(
            shape=tuple(self._int_workspace_buffer.shape),
            dtype=self._int_workspace_buffer.dtype,
        ).pin_memory()
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        if backend == "auto":
            self._backend = determine_mla_backend(self.device)
        else:
            self._backend = backend

    def plan(
        self,
        qo_indptr: paddle.Tensor,
        kv_indptr: paddle.Tensor,
        kv_indices: paddle.Tensor,
        kv_len_arr: paddle.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: paddle.dtype,
        kv_data_type: paddle.dtype,
        use_profiler: bool = False,
    ) -> None:
        """Plan the MLA attention computation.

        Parameters
        ----------
        qo_indptr : torch.IntTensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
            For decoding attention, the length of each query is 1, and the content
            of the tensor should be ``[0, 1, 2, ..., batch_size]``.
        kv_indptr : torch.IntTensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices : torch.IntTensor
            The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger.
        kv_len_arr : torch.IntTensor
            The query length of each request, shape: ``[batch_size]``.
        num_heads : int
            The number of heads in query/output tensor.
        head_dim_ckv : int
            The head dimension of compressed-kv.
        head_dim_kpe : int
            The head dimension for rope k-cache.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to use causal attention.
        sm_scale : float
            The scale factor for softmax operation.
        q_data_type : torch.dtype
            The data type of the query tensor.
        kv_data_type : torch.dtype
            The data type of the kv-cache tensor.
        use_profiler : bool, optional
            Whether to enable intra-kernel profiler, default is False.
        """
        for tensor, name in [
            (kv_len_arr, "kv_len_arr"),
            (kv_indptr, "kv_indptr"),
            (qo_indptr, "qo_indptr"),
            (kv_indices, "kv_indices"),
        ]:
            if tensor.dtype != "int32":
                raise ValueError(
                    f"Expected {name}.dtype == torch.int32, got {tensor.dtype}"
                )
        self._cached_module = get_batch_mla_module(
            self._backend,
            q_data_type,
            kv_data_type,
            q_data_type,
            qo_indptr.dtype,
            head_dim_ckv,
            head_dim_kpe,
            use_profiler,
        )
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")
        if self._use_cuda_graph:
            paddle.assign(qo_indptr, output=self._qo_indptr_buf)
            paddle.assign(kv_indptr, output=self._kv_indptr_buf)
            paddle.assign(kv_indices, output=self._kv_indices_buf[: len(kv_indices)])
            paddle.assign(kv_len_arr, output=self._kv_len_arr_buf)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, blocking=not True)
            self._kv_indptr_buf = kv_indptr.to(self.device, blocking=not True)
            self._kv_indices_buf = kv_indices.to(self.device, blocking=not True)
            self._kv_len_arr_buf = kv_len_arr.to(self.device, blocking=not True)
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler
        self._plan_info = self._cached_module.plan.default(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            num_heads,
            head_dim_ckv,
            causal,
        )

    @overload
    def run(
        self,
        q_nope: paddle.Tensor,
        q_pe: paddle.Tensor,
        ckv_cache: paddle.Tensor,
        kpe_cache: paddle.Tensor,
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
        return_lse: Literal[False] = False,
        profiler_buffer: Optional[paddle.Tensor] = None,
        kv_len: Optional[paddle.Tensor] = None,
        page_table: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        ...

    @overload
    def run(
        self,
        q_nope: paddle.Tensor,
        q_pe: paddle.Tensor,
        ckv_cache: paddle.Tensor,
        kpe_cache: paddle.Tensor,
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
        return_lse: Literal[True] = True,
        profiler_buffer: Optional[paddle.Tensor] = None,
        kv_len: Optional[paddle.Tensor] = None,
        page_table: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        ...

    def run(
        self,
        q_nope: paddle.Tensor,
        q_pe: paddle.Tensor,
        ckv_cache: paddle.Tensor,
        kpe_cache: paddle.Tensor,
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[paddle.Tensor] = None,
        kv_len: Optional[paddle.Tensor] = None,
        page_table: Optional[paddle.Tensor] = None,
    ) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
        """Run the MLA attention computation.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor without rope, shape: ``[batch_size, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            The rope part of the query tensor, shape: ``[batch_size, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            The compressed kv-cache tensor (without rope), shape: ``[num_pages, page_size, head_dim_ckv]``.
            ``head_dim_ckv`` is 512 in DeepSeek v2/v3 models.
        kpe_cache : torch.Tensor
            The rope part of the kv-cache tensor, shape: ``[num_pages, page_size, head_dim_kpe]``.
            ``head_dim_kpe`` is 64 in DeepSeek v2/v3 models.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool, optional
            Whether to return the log-sum-exp value, default is False.
        profiler_buffer : Optional[torch.Tensor]
            The buffer to store the profiler data.
        kv_len : Optional[torch.Tensor]
            The query length of each request, shape: ``[batch_size]``. Required when ``backend`` is ``cutlass``.
        page_table : Optional[torch.Tensor]
            The page table of the paged kv-cache, shape: ``[batch_size, num_pages]``. Required when ``backend`` is ``cutlass``.
        """
        if self._backend == "cutlass":
            if return_lse:
                raise ValueError("return_lse does not support cutlass backend for now.")
            if profiler_buffer is not None:
                raise ValueError(
                    "profiler_buffer does not support cutlass backend for now."
                )
            self._cached_module = get_mla_module()
            if out is None:
                out = paddle.empty_like(x=q_nope)
            else:
                check_shape_dtype_device(
                    out, tuple(q_nope.shape), q_nope.dtype, q_nope.place, "out"
                )
            q_nope_pe = paddle.concat(x=[q_nope, q_pe], axis=-1)
            ckv_kpe_cache = paddle.concat(x=[ckv_cache, kpe_cache], axis=-1)
            _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table)
            lse = paddle.empty(shape=[0], dtype="float32")
            self._cached_module.cutlass_mla_paged_attention.default(
                self._float_workspace_buffer,
                out,
                lse,
                q_nope_pe,
                ckv_kpe_cache,
                kv_len,
                page_table,
            )
            return out
        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        num_heads = tuple(q_nope.shape)[1]
        page_size = self._page_size
        sm_scale = self._sm_scale
        causal = self._causal
        mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        device = self.device
        if out is None:
            out = paddle.empty_like(x=q_nope)
        else:
            check_shape_dtype_device(
                out, tuple(q_nope.shape), q_nope.dtype, q_nope.place, "out"
            )
        if return_lse:
            if lse is None:
                lse = paddle.empty(shape=tuple(q_nope.shape)[:2], dtype="float32")
            else:
                check_shape_dtype_device(
                    lse, tuple(q_nope.shape)[:2], "float32", q_nope.place, "lse"
                )
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self._cached_module.run.default(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            self._kv_indices_buf,
            out,
            lse,
            mask_mode,
            num_heads,
            page_size,
            sm_scale,
            *profiler_args,
        )
        return (out, lse) if return_lse else out
