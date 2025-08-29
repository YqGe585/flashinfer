import sys


import paddle
import pytest
from flashinfer.paddle_utils import *

from flashinfer import mxfp8_dequantize_host, mxfp8_quantize


@pytest.mark.parametrize("m", [1, 1024])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_mxfp8_quantize_torch(m, k, dtype, is_sf_swizzled_layout, device):
    a = 16 * paddle.randn(shape=[m, k], dtype=dtype).to(device).contiguous()
    if device == "cpu":
        a = a.astype(dtype="float32")
    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)
    if device == "cuda":
        a_fp8 = a_fp8.cpu()
        a_sf = a_sf.cpu()
    a_pt = mxfp8_dequantize_host(
        a_fp8.view("uint8"), a_sf.view("uint8").reshape(-1), is_sf_swizzled_layout
    )
    if device == "cuda":
        a_pt = a_pt.cuda()
    paddle.device.synchronize()

    def check_accuracy(a, b, atol, rtol, percent):
        if paddle.any(x=paddle.isnan(x=a)):
            raise Exception("NaN in a")
        if paddle.any(x=paddle.isnan(x=b)):
            raise Exception("NaN in b")
        assert tuple(a.shape) == tuple(b.shape)
        left = paddle.abs(x=a - b)
        right = atol + rtol * paddle.abs(x=b)
        count = paddle.sum(x=left > right)
        mismatch_percent = count / a.size
        if mismatch_percent > 1 - percent:
            raise Exception(
                "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
            )

    check_accuracy(a_pt, a, 8, 0, 0.999)


def mxfp8_quantize_check_accuracy(a, b, atol, rtol, percent):
    if paddle.any(x=paddle.isnan(x=a)):
        raise Exception("NaN in a")
    if paddle.any(x=paddle.isnan(x=b)):
        raise Exception("NaN in b")
    assert tuple(a.shape) == tuple(b.shape)
    left = paddle.abs(x=a - b)
    right = atol + rtol * paddle.abs(x=b)
    count = paddle.sum(x=left > right)
    mismatch_percent = count / a.size
    if mismatch_percent > 1 - percent:
        raise Exception(
            "Mismatch percentage is %f for rtol %f" % (mismatch_percent, rtol)
        )


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_torch_host(m, k, dtype, is_sf_swizzled_layout):
    paddle.seed(seed=0)
    a = (paddle.randn(shape=[m, k], dtype="float32") * 16).cpu().contiguous()
    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout)
    a_pt = mxfp8_dequantize_host(
        a_fp8.view("uint8"), a_sf.view("uint8"), is_sf_swizzled_layout
    )
    paddle.device.synchronize()
    mxfp8_quantize_check_accuracy(a_pt, a, 8, 0, 0.999)


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_torch_device(m, k, dtype, is_sf_swizzled_layout):
    paddle.seed(seed=0)
    a = (paddle.randn(shape=[m, k], dtype="float32") * 16).to(dtype).cuda().contiguous()
    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, 32)
    a_pt = mxfp8_dequantize_host(
        a_fp8.cpu().view("uint8"), a_sf.cpu().view("uint8"), is_sf_swizzled_layout
    )
    paddle.device.synchronize()
    mxfp8_quantize_check_accuracy(
        a_pt.cpu().to("float32"), a.cpu().to("float32"), 8, 0, 0.999
    )


@pytest.mark.parametrize("m", [1, 2, 16, 1024])
@pytest.mark.parametrize("k", [1568])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@pytest.mark.parametrize("alignment", [64, 128])
def test_mxfp8_quantize_alignment_torch_device(
    m, k, dtype, is_sf_swizzled_layout, alignment
):
    paddle.seed(seed=0)
    a = (paddle.randn(shape=[m, k], dtype="float32") * 16).to(dtype).cuda().contiguous()
    padded_k = (k + alignment - 1) // alignment * alignment
    a_fp8, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout, alignment)
    assert tuple(a_fp8.shape)[1] == padded_k
    a_pt = mxfp8_dequantize_host(
        a_fp8.cpu().view("uint8"), a_sf.cpu().view("uint8"), is_sf_swizzled_layout
    )
    paddings = a_fp8.view("int8")[:, k:]
    assert paddle.all(x=paddings == 0), "Paddings should be zero"
    paddle.device.synchronize()
    mxfp8_quantize_check_accuracy(
        a_pt[:, :k].cpu().to("float32"), a.cpu().to("float32"), 8, 0, 0.999
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
