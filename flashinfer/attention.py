import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

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
import functools
import math
from typing import Optional, Tuple, Union

from .jit import gen_batch_attention_module
from .utils import (MaskMode, PosEncodingMode, TensorLayout, _check_kv_layout,
                    _unpack_paged_kv_cache)


@functools.cache
def get_holistic_attention_module(*args):
    return gen_batch_attention_module(*args).build_and_load()


class BatchAttention:
    def __init__(self, kv_layout: str = "NHD", device: str = "cuda"):
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout
        self.float_workspace_buffer = paddle.empty(
            shape=384 * 1024 * 1024, dtype="uint8"
        )
        self.int_workspace_buffer = paddle.empty(shape=8 * 1024 * 1024, dtype="uint8")
        self.page_locked_int_workspace_buffer = paddle.empty(
            shape=8 * 1024 * 1024, dtype="uint8"
        ).pin_memory()

    def plan(
        self,
        qo_indptr: paddle.Tensor,
        kv_indptr: paddle.Tensor,
        kv_indices: paddle.Tensor,
        kv_len_arr: paddle.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool = False,
        sm_scale: float = None,
        logits_soft_cap: Optional[float] = None,
        q_data_type: paddle.dtype = "bfloat16",
        kv_data_type: paddle.dtype = "bfloat16",
        use_profiler: bool = False,
    ) -> None:
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self._logits_soft_cap = logits_soft_cap
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode["NONE"].value,
            logits_soft_cap > 0.0,
            use_profiler,
        )
        self.module = get_holistic_attention_module(*get_module_args)
        qo_indptr_host = qo_indptr.to(device2str("cpu"), blocking=not True)
        kv_indptr_host = kv_indptr.to(device2str("cpu"), blocking=not True)
        kv_len_arr_host = kv_len_arr.to(device2str("cpu"), blocking=not True)
        paddle.device.synchronize()
        batch_size = tuple(kv_len_arr.shape)[0]
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler
        self._kv_indices = kv_indices
        self._plan_info = self.module.plan(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self.page_locked_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )

    def run(
        self,
        q: paddle.Tensor,
        kv_cache: Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]],
        out: Optional[paddle.Tensor] = None,
        lse: Optional[paddle.Tensor] = None,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        if logits_soft_cap > 0.0 and self._logits_soft_cap <= 0.0:
            raise ValueError(
                "logits_soft_cap used in kernel run but not provided in plan(). This will cause template deduction error."
            )
        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = paddle.empty_like(x=q)
        if lse is None:
            lse = paddle.empty(
                shape=[tuple(q.shape)[0], tuple(q.shape)[1]], dtype="float32"
            )
        head_dim_qk = tuple(q.shape)[2]
        if self._sm_scale is None:
            self._sm_scale = 1.0 / math.sqrt(head_dim_qk)
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            self._sm_scale,
            logits_soft_cap,
            *profiler_args
        )
        return out, lse
