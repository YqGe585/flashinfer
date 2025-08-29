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
import pytest

import flashinfer
from flashinfer.utils import get_compute_capability


def llama_rms_norm(x, w, eps=1e-06):
    orig_dtype = x.dtype
    x = x.astype(dtype="float32")
    variance = x.pow(y=2).mean(axis=-1, keepdim=True)
    x = x * paddle.rsqrt(x=variance + eps)
    x = x * w.astype(dtype="float32")
    x = x.to(orig_dtype)
    return x


def gemma_rms_norm(x, w, eps=1e-06):
    orig_dtype = x.dtype
    x = x.astype(dtype="float32")
    variance = x.pow(y=2).mean(axis=-1, keepdim=True)
    x = x * paddle.rsqrt(x=variance + eps)
    x = x * (1.0 + w.astype(dtype="float32"))
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-06):
    orig_dtype = x.dtype
    x = x + residual
    residual = x
    x = x.astype(dtype="float32")
    variance = x.pow(y=2).mean(axis=-1, keepdim=True)
    x = x * paddle.rsqrt(x=variance + eps)
    x = x * (1.0 + w.astype(dtype="float32"))
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to("float32")
    x = x + residual.to("float32")
    residual = x.to(orig_dtype)
    variance = x.pow(y=2).mean(axis=-1, keepdim=True)
    x = x * paddle.rsqrt(x=variance + eps)
    x = (x * weight.astype(dtype="float32")).to(orig_dtype)
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous):
    if contiguous:
        x = paddle.randn(shape=[batch_size, hidden_size]).to(0).to(dtype)
    else:
        x = paddle.randn(shape=[batch_size, hidden_size * 2]).to(dtype)
        x = x[:, :hidden_size]
    major, _ = get_compute_capability(x.place)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    w = paddle.randn(shape=hidden_size).to(0).to(dtype)
    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = paddle.empty_like(x=x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl)
    assert paddle.allclose(x=y_ref, y=y, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype, enable_pdl, contiguous):
    eps = 1e-06
    if contiguous:
        x = paddle.randn(shape=[batch_size, hidden_size], dtype=dtype)
    else:
        x = paddle.randn(shape=[batch_size, hidden_size * 2]).to(dtype)
        x = x[:, :hidden_size]
    major, _ = get_compute_capability(x.place)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    residual = paddle.randn(shape=x.shape, dtype=x.dtype)
    weight = paddle.randn(shape=hidden_size, dtype=dtype)
    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )
    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )
    assert paddle.allclose(x=x_fused, y=x_native, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(
        x=residual_fused, y=residual_native, rtol=0.001, atol=0.001
    ).item(), ""


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_norm(
    batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous
):
    if contiguous:
        x = paddle.randn(shape=[batch_size, hidden_size]).to(0).to(dtype)
    else:
        x = paddle.randn(shape=[batch_size, hidden_size * 2]).to(dtype)
        x = x[:, :hidden_size]
    major, _ = get_compute_capability(x.place)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    w = paddle.randn(shape=hidden_size).to(0).to(dtype)
    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = paddle.empty_like(x=x)
        flashinfer.norm.gemma_rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.gemma_rmsnorm(x, w, enable_pdl=enable_pdl)
    assert paddle.allclose(x=y_ref, y=y, rtol=0.001, atol=0.001).item(), ""


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_fused_add_rmsnorm(
    batch_size, hidden_size, dtype, enable_pdl, contiguous
):
    eps = 1e-06
    if contiguous:
        x = paddle.randn(shape=[batch_size, hidden_size], dtype=dtype)
    else:
        x = paddle.randn(shape=[batch_size, hidden_size * 2]).to(dtype)
        x = x[:, :hidden_size]
    major, _ = get_compute_capability(x.place)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    residual = paddle.randn(shape=x.shape, dtype=x.dtype)
    weight = paddle.randn(shape=hidden_size, dtype=dtype)
    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )
    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.gemma_fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )
    assert paddle.allclose(x=x_fused, y=x_native, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(
        x=residual_fused, y=residual_native, rtol=0.001, atol=0.001
    ).item(), ""


if __name__ == "__main__":
    test_fused_add_rmsnorm(1, 16384, "float16", True, True)
