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
from typing import Literal

import numpy
import pytest

import flashinfer


def numpy_packbits_ref(x_cpu: paddle.Tensor, bitorder: Literal["big", "little"]):
    x_np = x_cpu.numpy()
    x_packed = numpy.packbits(x_np, bitorder=bitorder)
    return paddle.to_tensor(data=x_packed)


@pytest.mark.parametrize("num_elements", [1, 10, 99, 128, 999, 5000, 131072, 999999])
@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_packbits(num_elements, bitorder):
    paddle.seed(seed=42)
    x_cpu = paddle.rand(shape=num_elements) < 0.5
    x_gpu = x_cpu.to(0)
    x_packed_ref = numpy_packbits_ref(x_cpu, bitorder)
    x_packed = flashinfer.quantization.packbits(x_gpu, bitorder)
    assert paddle.equal_all(x=x_packed_ref.cpu(), y=x_packed.cpu()).item()


@pytest.mark.parametrize("batch_size", [1, 10, 99, 128, 777, 999])
@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_segment_packbits(batch_size, bitorder):
    paddle.seed(seed=42)
    old_indptr = paddle.cumsum(x=paddle.arange(end=batch_size + 1), axis=0).to(0)
    num_elements = old_indptr[-1].item()
    x_cpu = paddle.rand(shape=num_elements) < 0.5
    x_gpu = x_cpu.to(0)
    y_gpu, new_indptr = flashinfer.quantization.segment_packbits(
        x_gpu, old_indptr, bitorder
    )
    for i in range(batch_size):
        x_segment_i = x_gpu[old_indptr[i] : old_indptr[i + 1]]
        y_segment_i_ref = flashinfer.packbits(x_segment_i, bitorder)
        assert paddle.equal_all(
            x=y_gpu[new_indptr[i] : new_indptr[i + 1]], y=y_segment_i_ref
        ).item()


if __name__ == "__main__":
    test_packbits(999999, "big")
    test_segment_packbits(77, "little")
