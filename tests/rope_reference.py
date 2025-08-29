import sys

sys.path.append("/home/flashinfer")
import math
from typing import Optional, Tuple, Union

import paddle
from paddle_utils import *


def apply_scaling(freqs: paddle.Tensor):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return paddle.to_tensor(data=new_freqs, dtype=freqs.dtype, place=freqs.place)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    use_scaled: bool = False,
    device: str = "cuda:0",
):
    freqs = 1.0 / theta ** (
        paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(dtype="float32")
        / dim
    )
    t = paddle.arange(dtype="float32", end=end)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = paddle.outer(x=t, y=freqs)
    freqs_cis = paddle.polar(abs=paddle.ones_like(x=freqs), angle=freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: paddle.Tensor, x: paddle.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert tuple(freqs_cis.shape) == (tuple(x.shape)[1], tuple(x.shape)[-1])
    shape = [
        (d if i == 1 or i == ndim - 1 else 1) for i, d in enumerate(tuple(x.shape))
    ]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: paddle.Tensor, xk: paddle.Tensor, freqs_cis: paddle.Tensor
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    xq_ = paddle.as_complex(
        x=xq.astype(dtype="float32").reshape(*tuple(xq.shape)[:-1], -1, 2)
    )
    xk_ = paddle.as_complex(
        x=xk.astype(dtype="float32").reshape(*tuple(xk.shape)[:-1], -1, 2)
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = paddle.as_real(x=xq_ * freqs_cis).flatten(start_axis=3)
    xk_out = paddle.as_real(x=xk_ * freqs_cis).flatten(start_axis=3)
    return xq_out.astype(dtype=xq.dtype), xk_out.astype(dtype=xk.dtype)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(axis=1)
    sin = sin.unsqueeze(axis=1)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def rotate_half(x):
    x1 = x[..., : tuple(x.shape)[-1] // 2]
    x2 = x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def generate_cos_sin_f32_cache(
    max_seq_len,
    head_dim,
    theta=10000.0,
    use_scaled: bool = False,
    device: str = "cuda:0",
):
    position = paddle.arange(dtype="float32", end=max_seq_len).unsqueeze(axis=1)
    freqs = 1.0 / theta ** (
        paddle.arange(start=0, end=head_dim, step=2, dtype="float32") / head_dim
    )
    freqs = paddle.concat(x=[freqs, freqs], axis=-1).contiguous()
    if use_scaled:
        freqs = apply_scaling(freqs)
    args = position * freqs
    sin_cache = paddle.sin(x=args)
    cos_cache = paddle.cos(x=args)
    return cos_cache, sin_cache


class RotaryEmbedding(paddle.nn.Layer):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: paddle.dtype,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.device = device
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

    def forward_native(
        self,
        positions: paddle.Tensor,
        query: paddle.Tensor,
        key: paddle.Tensor,
        offsets: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = tuple(positions.shape)[0]
        cos_sin = self.cos_sin_cache.index_select(axis=0, index=positions)
        query = query.to("float32")
        key = key.to("float32")
        cos, sin = cos_sin.chunk(chunks=2, axis=-1)
        query_shape = tuple(query.shape)
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self._apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = paddle.concat(x=(query_rot, query_pass), axis=-1).reshape(query_shape)
        key_shape = tuple(key.shape)
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = paddle.concat(x=(key_rot, key_pass), axis=-1).reshape(key_shape)
        query = query.to(self.dtype)
        key = key.to(self.dtype)
        return query, key
