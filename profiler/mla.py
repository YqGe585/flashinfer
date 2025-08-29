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
import argparse

import flashinfer
from flashinfer.profiler import export_to_perfetto_trace


def profile_deepseek_mla_decode(
    batch_size, seq_len, num_heads, profiler_buffer_size, backend
):
    head_dim_ckv = 512
    head_dim_kpe = 64
    page_size = 1
    q_nope = paddle.randn(
        shape=[batch_size * 1, num_heads, head_dim_ckv], dtype="float16"
    )
    q_pe = paddle.zeros(
        shape=[batch_size * 1, num_heads, head_dim_kpe], dtype="float16"
    )
    ckv = paddle.randn(shape=[batch_size * seq_len, 1, head_dim_ckv], dtype="float16")
    kpe = paddle.zeros(shape=[batch_size * seq_len, 1, head_dim_kpe], dtype="float16")
    sm_scale = 1.0 / (head_dim_ckv + head_dim_kpe) ** 0.5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8").to(0)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32")
    kv_indptr = (
        paddle.arange(start=0, end=batch_size + 1).to(0).astype(dtype="int32") * seq_len
    )
    kv_indices = (
        paddle.arange(start=0, end=batch_size * seq_len).to(0).astype(dtype="int32")
    )
    kv_lens = paddle.full(shape=(batch_size,), fill_value=seq_len, dtype="int32").to(0)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
        use_profiler=True,
    )
>>>>>>    profiler_buffer = paddle.zeros(shape=(profiler_buffer_size,), dtype=torch.uint64)
    _o = wrapper.run(
        q_nope, q_pe, ckv, kpe, return_lse=False, profiler_buffer=profiler_buffer
    )
    profiler_buffer.zero_()
    wrapper.run(
        q_nope, q_pe, ckv, kpe, return_lse=False, profiler_buffer=profiler_buffer
    )
    export_to_perfetto_trace(
        profiler_buffer,
        [
            "issue-load-q",
            "issue-load-kv",
            "write-o",
            "softmax-update",
            "gemm-qk",
            "gemm-pv",
            "rescale-o",
            "write-p-smem",
            "split-k",
        ],
        f"mla-{backend}-{batch_size}-{seq_len}-{num_heads}.perfetto-trace",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Intra-kernel profiling for FlashInfer MLA kernels"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=128)
    parser.add_argument("--profiler-buffer-size", type=int, default=1024 * 1024)
    args = parser.parse_args()
    profile_deepseek_mla_decode(
        args.batch_size, args.seq_len, args.num_heads, args.profiler_buffer_size, "fa3"
    )
