import paddle

"""
Benchmark RoPE for flashinfer and vLLM. vLLM installation is required to run this benchmark.

Usage:
$ pip install vllm
$ python bench_rope.py
"""
from typing import Optional, Tuple, Union

import numpy as np
import triton
from vllm.model_executor.layers.rotary_embedding import \
    RotaryEmbedding as vLLMRotaryEmbedding

from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.testing.utils import bench_gpu_time


class FlashInferRotaryEmbedding(paddle.nn.Layer):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: paddle.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache: paddle.Tensor
        self.register_buffer(name="cos_sin_cache", tensor=cache, persistable=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> paddle.Tensor:
        inv_freq = 1.0 / base ** (
            paddle.arange(start=0, end=self.rotary_dim, step=2, dtype="float32")
            / self.rotary_dim
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> paddle.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = paddle.arange(dtype="float32", end=self.max_position_embeddings)
        freqs = paddle.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = paddle.concat(x=(cos, sin), axis=-1)
        return cache

    def _apply_rotary_emb(
        self,
        x: paddle.Tensor,
        cos: paddle.Tensor,
        sin: paddle.Tensor,
        is_neox_style: bool,
    ) -> paddle.Tensor:
        """
        Args:
            x: [num_tokens, num_heads, head_size]
            cos: [num_tokens, head_size // 2]
            sin: [num_tokens, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
                positional embeddings.
        """
        cos = cos.unsqueeze(axis=-2).to(x.dtype)
        sin = sin.unsqueeze(axis=-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = paddle.chunk(x=x, chunks=2, axis=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return paddle.concat(x=(o1, o2), axis=-1)
        else:
            return paddle.stack(x=(o1, o2), axis=-1).flatten(start_axis=-2)

    def forward_cuda(
        self,
        positions: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
        offsets: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query, key


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ],
        line_arg="provider",
        line_vals=["flashinfer", "native", "vllm"],
        line_names=["FlashInfer", "Native", "vLLM"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="rope-latency",
        args={
            "head_size": 4096 // 32,
            "rotary_dim": 4096 // 32,
            "max_position_embeddings": 65536,
            "base": 500000,
            "is_neox_style": True,
            "dtype": "bfloat16",
            "device": "cuda",
            "batch_size": 2,
            "num_q_heads": 32,
            "num_kv_heads": 8,
        },
    )
)
def benchmark(
    provider,
    head_size,
    rotary_dim,
    max_position_embeddings,
    base,
    is_neox_style,
    dtype,
    device,
    batch_size,
    seq_len,
    num_q_heads,
    num_kv_heads,
):
    print(
        f"provider: {provider}, head_size: {head_size}, rotary_dim: {rotary_dim}, max_position_embeddings: {max_position_embeddings}, base: {base}, is_neox_style: {is_neox_style}, dtype: {dtype}, device: {device}, batch_size: {batch_size}, seq_len: {seq_len}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}"
    )
    rope_forward = None
    if provider == "vllm":
        rope = vLLMRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_cuda
    elif provider == "flashinfer":
        rope = FlashInferRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_cuda
    elif provider == "native":
        rope = vLLMRotaryEmbedding(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        ).to(device)
        rope_forward = rope.forward_native
    pos_ids = paddle.arange(end=seq_len).tile(repeat_times=batch_size)
    query = paddle.randn(
        shape=[batch_size * seq_len, num_q_heads * head_size], dtype=dtype
    )
    key = paddle.randn(
        shape=[batch_size * seq_len, num_kv_heads * head_size], dtype=dtype
    )
    measurements = bench_gpu_time(lambda: rope_forward(pos_ids, query, key))
    ms = np.median(measurements)
    min_ms = np.percentile(measurements, 20)
    max_ms = np.percentile(measurements, 80)
    return ms, min_ms, max_ms


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="rope_benchmark.png")
