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
import pytest

import flashinfer
from flashinfer.utils import determine_gemm_backend, is_sm90a_supported

DTYPES = ["float16"]
CUDA_DEVICES = ["cuda:0"]


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    jit_specs = [flashinfer.gemm.gen_gemm_module()]
    if is_sm90a_supported(device2str("cuda:0")):
        jit_specs.append(flashinfer.gemm.gen_gemm_sm90_module())
    flashinfer.jit.build_jit_specs(jit_specs, verbose=False)
    yield


@pytest.mark.parametrize("batch_size", [1, 77, 199])
@pytest.mark.parametrize("num_rows_per_batch", [3, 10, 99])
@pytest.mark.parametrize("d_in", [128, 1024, 4096])
@pytest.mark.parametrize("d_out", [128, 1024, 4096])
@pytest.mark.parametrize("use_weight_indices", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("backend", ["sm90", "sm80"])
def test_segment_gemm(
    batch_size,
    num_rows_per_batch,
    d_in,
    d_out,
    use_weight_indices,
    column_major,
    dtype,
    device,
    backend,
):
    paddle.seed(seed=42)
    if batch_size * num_rows_per_batch > 8192:
        pytest.skip("batch_size * num_rows_per_batch too large for test.")
    latest_supported_backend = determine_gemm_backend(device2str(device))
    if backend == "sm90" and latest_supported_backend == "sm80":
        pytest.skip("sm90 backend not supported on this device.")
    paddle.seed(seed=42)
    workspace_buffer = paddle.empty(shape=32 * 1024 * 1024, dtype="int8")
    segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer, backend=backend)
    x = paddle.randn(shape=[batch_size * num_rows_per_batch, d_in], dtype=dtype)
    if use_weight_indices:
        num_weights = 1024
        if column_major:
            weight = paddle.randn(shape=[num_weights, d_out, d_in], dtype=dtype)
        else:
            weight = paddle.randn(shape=[num_weights, d_in, d_out], dtype=dtype)
    elif column_major:
        weight = paddle.randn(shape=[batch_size, d_out, d_in], dtype=dtype)
    else:
        weight = paddle.randn(shape=[batch_size, d_in, d_out], dtype=dtype)
    y = segment_gemm.run(
        x,
        weight,
        batch_size,
        weight_column_major=column_major,
        seg_lens=paddle.full(
            shape=(batch_size,), fill_value=num_rows_per_batch, dtype="int64"
        ),
        weight_indices=paddle.arange(start=0, end=batch_size) % num_weights
        if use_weight_indices
        else None,
    )
    if use_weight_indices:
        for i in range(batch_size):
            assert paddle.allclose(
                x=y[i * num_rows_per_batch : (i + 1) * num_rows_per_batch],
                y=paddle.matmul(
                    x=x[i * num_rows_per_batch : (i + 1) * num_rows_per_batch].astype(
                        dtype="float32"
                    ),
                    y=weight[i % num_weights].astype(dtype="float32").T
                    if column_major
                    else weight[i % num_weights].astype(dtype="float32"),
                ).to(dtype),
                rtol=0.001,
                atol=0.001,
            ).item(), ""
    else:
        assert paddle.allclose(
            x=y,
            y=paddle.matmul(
                x=x.view(batch_size, num_rows_per_batch, d_in).astype(dtype="float32"),
                y=weight.astype(dtype="float32").transpose(
                    perm=dim2perm(weight.astype(dtype="float32").ndim, -1, -2)
                )
                if column_major
                else weight.astype(dtype="float32"),
            )
            .view(batch_size * num_rows_per_batch, d_out)
            .to(dtype),
            rtol=0.001,
            atol=0.002,
        ).item(), ""


if __name__ == "__main__":
    test_segment_gemm(199, 17, 128, 1024, False, False, "float16", "cuda:0", "auto")
    test_segment_gemm(199, 17, 128, 1024, False, True, "float16", "cuda:0", "auto")
    test_segment_gemm(199, 17, 128, 1024, True, False, "float16", "cuda:0", "auto")
    test_segment_gemm(199, 17, 128, 1024, True, True, "float16", "cuda:0", "auto")
