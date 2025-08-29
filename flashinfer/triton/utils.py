from typing import List

import paddle


def check_input(x: paddle.Tensor):
    assert x.place.is_gpu_place(), f"{str(x)} must be a CUDA Tensor"
    assert x.is_contiguous(), f"{str(x)} must be contiguous"


def check_dim(d, x: paddle.Tensor):
    assert x.dim() == d, f"{str(x)} must be a {d}D tensor"


def check_shape(a: paddle.Tensor, b: paddle.Tensor):
    assert a.dim() == b.dim(), "tensors should have same dim"
    for i in range(a.dim()):
        assert (
            a.shape[i] == b.shape[i]
        ), f"tensors shape mismatch, {tuple(a.shape)} and {tuple(b.shape)}"


def check_device(tensors: List[paddle.Tensor]):
    device = tensors[0].place
    for t in tensors:
        assert (
            t.place == device
        ), f"All tensors should be on the same device, but got {device} and {t.place}"
