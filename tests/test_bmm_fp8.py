import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
import pytest
from paddle_utils import *

from flashinfer import autotune, bmm_fp8


>>>>>>def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = paddle.finfo(dtype=dtype)
    min_val, max_val = tuple(
        [
            paddle.amin(x, axis=None, keepdim=False),
            paddle.max(x, axis=None, keepdim=False),
        ]
    )
    amax = paddle.maximum(x=min_val.abs(), y=max_val.abs()).clip(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clip(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.astype(dtype="float32").reciprocal()


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [48, 128])
@pytest.mark.parametrize("n", [80, 64])
@pytest.mark.parametrize("k", [64, 256])
>>>>>>@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
>>>>>>@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("backend", ["cudnn", "cublas", "cutlass", "auto"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_bmm_fp8(b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning):
>>>>>>    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")
>>>>>>    if input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2:
        if backend == "cutlass":
            pytest.skip("Invalid combination: cutlass does not support e5m2")
    if auto_tuning and backend != "cutlass":
        pytest.skip("Invalid combination: auto_tuning only supported for cutlass")
    input = paddle.randn(shape=[b, m, k], dtype="bfloat16")
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)
    mat2 = paddle.randn(shape=[b, n, k], dtype="bfloat16").transpose(
        perm=dim2perm(paddle.randn(shape=[b, n, k], dtype="bfloat16").ndim, -2, -1)
    )
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)
    reference = paddle.bmm(x=input, y=mat2)
    res = paddle.empty(shape=[b, m, n], dtype=res_dtype)
    with autotune(auto_tuning):
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend=backend,
        )
    cos_sim = paddle.nn.functional.cosine_similarity(
        x1=reference.reshape(-1), x2=res.reshape(-1), axis=0
    )
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
