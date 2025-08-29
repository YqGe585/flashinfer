import sys

sys.path.append("/home/flashinfer_paddle")
import functools

import paddle
import pytest
from paddle_utils import *
from utils_fp4 import cast_from_fp4, recover_swizzled_scales, ref_fp4_quant

from flashinfer import (block_scale_interleave, e2m1_and_ufp8sf_scale_to_float,
                        fp4_quantize, mxfp4_dequantize, mxfp4_quantize)
from flashinfer.utils import is_sm100a_supported

DTYPES = ["float16", "bfloat16"]
SHAPES = [(128, 64), (256, 128), (120, 64), (200, 256)]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]
FLOAT4_E2M1_MAX = 6.0
>>>>>>FLOAT8_E4M3_MAX = paddle.finfo(dtype=torch.float8_e4m3fn).max
BLOCK_SIZE = 16


def swizzle_sf(
    unswizzled_sf: paddle.Tensor,
    original_row: int,
    original_col: int,
    scaling_vector_size: int = 16,
) -> paddle.Tensor:
    """
    Inverse of `unswizzle_sf`. Converts an unswizzled tensor back to swizzled form.

    Args:
        unswizzled_sf: Tensor of shape [row, col // scaling_vector_size].
        original_row: Original row dimension (e.g., 120).
        original_col: Original column dimension (e.g., 64).
        scaling_vector_size: Scaling factor (default 16).

    Returns:
        Swizzled tensor of shape [padded_row, padded_col // scaling_vector_size].
    """
    unswizzled_sf = unswizzled_sf.contiguous()
    factor = scaling_vector_size * 4
    padded_row = (original_row + 128 - 1) // 128 * 128
    padded_col = (original_col + factor - 1) // factor * factor
    pad_rows = padded_row - original_row
    pad_cols = (padded_col - original_col) // scaling_vector_size
    padded_sf = paddle.nn.functional.pad(
        x=unswizzled_sf,
        pad=(0, pad_cols, 0, pad_rows),
        mode="constant",
        value=0,
        pad_from_left_axis=False,
    ).contiguous()
    num_m_tiles = padded_row // 128
    num_k_tiles = padded_col // factor
    sf_reshaped = padded_sf.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_swizzled = sf_reshaped.transpose(perm=dim2perm(sf_reshaped.ndim, 1, 3))
    sf_swizzled = sf_swizzled.reshape(padded_row, padded_col // scaling_vector_size)
    return sf_swizzled.contiguous()


def unswizzle_sf(
    sf: paddle.Tensor, row: int, col: int, scaling_vector_size: int = 16
) -> paddle.Tensor:
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(perm=dim2perm(sf_reshaped.ndim, 1, 3))
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, : col // scaling_vector_size]
    return sf_unswizzle_sliced.contiguous()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sf_use_ue8m0", [False, True])
@pytest.mark.parametrize("is_swizzled", [False, True])
@paddle.no_grad()
def test_fp4_quantization(
    dtype: paddle.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
    sf_use_ue8m0: bool,
    is_swizzled: bool,
) -> None:
    if not is_sm100a_supported(device2str(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    paddle.device.set_device(device=device2str(device))
    paddle.seed(seed=seed)
    m, n = shape
    sf_vec_size = 32 if sf_use_ue8m0 else 16
    x = paddle.randn(shape=(m, n), dtype=dtype)
    tensor_amax = paddle.abs(x=x)._max().to("float32")
    if sf_use_ue8m0:
        global_scale = paddle.to_tensor(data=1.0, dtype="float32")
    else:
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_fp4_quant(x, global_scale, sf_vec_size, sf_use_ue8m0)
    out, out_scale = fp4_quantize(
        x, global_scale, sf_vec_size, sf_use_ue8m0, is_swizzled
    )
    assert n % sf_vec_size == 0, f"cols needs to be {sf_vec_size} divisible"
    if sf_use_ue8m0:
        out_scale = (out_scale.to("int32") << 23).view("float32")
    else:
>>>>>>        out_scale = out_scale.view(torch.float8_e4m3fn).to("float32")
    if is_swizzled:
        scale_ans = recover_swizzled_scales(
            out_scale.reshape(-1, n // sf_vec_size), m, n, sf_vec_size
        )
    else:
        scale_ans = out_scale
    out_ans = cast_from_fp4(out).reshape(m, n)
    assert paddle.allclose(x=out_ans, y=out_ref, rtol=1.0, atol=0.1).item(), ""
    assert paddle.allclose(x=scale_ans, y=scale_ref, rtol=0.1, atol=0.1).item(), ""


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@paddle.no_grad()
def test_scale_swizzling(
    dtype: paddle.dtype, shape: tuple[int, int], seed: int, device: str
) -> None:
    if not is_sm100a_supported(device2str("cuda")):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    paddle.device.set_device(device=device2str(device))
    paddle.seed(seed=seed)
    m, n = shape
    x = paddle.randn(shape=(m, n), dtype=dtype)
    tensor_amax = paddle.abs(x=x)._max().to("float32")
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    _, unswizzled_scale = fp4_quantize(x, global_scale, BLOCK_SIZE, False, False)
    _, swizzled_scale = fp4_quantize(x, global_scale, BLOCK_SIZE, False, True)
    assert n % BLOCK_SIZE == 0, f"cols needs to be {BLOCK_SIZE} divisible"
    recovered_unswizzled_scale = unswizzle_sf(swizzle_sf(unswizzled_scale, m, n), m, n)
    ref_unswizzled_scale = unswizzle_sf(swizzled_scale, m, n)
    assert_equal = functools.partial(paddle.allclose, rtol=0, atol=0)
    assert_equal(recovered_unswizzled_scale, unswizzled_scale)
    assert_equal(ref_unswizzled_scale, unswizzled_scale)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@paddle.no_grad()
def test_block_scale_interleave(shape: tuple[int, int], seed: int, device: str) -> None:
    """Test the block_scale_interleave function directly."""
    if not is_sm100a_supported(device2str("cuda")):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    paddle.device.set_device(device=device2str(device))
    paddle.seed(seed=seed)
    m, n = shape
    sf_vec_size = BLOCK_SIZE
    scale_shape = m, n // sf_vec_size
    unswizzled_sf = paddle.randint(low=0, high=256, shape=scale_shape, dtype="uint8")
    swizzled_sf = block_scale_interleave(unswizzled_sf)
    ref_swizzled_sf = swizzle_sf(unswizzled_sf, m, n, sf_vec_size)
    assert swizzled_sf.dtype == "uint8", f"Expected uint8, got {swizzled_sf.dtype}"
    assert swizzled_sf.place == unswizzled_sf.place, "Device mismatch"
    factor = sf_vec_size * 4
    padded_row = (m + 128 - 1) // 128 * 128
    padded_col = (n + factor - 1) // factor * factor
    expected_shape = padded_row, padded_col // sf_vec_size
    expected_size = expected_shape[0] * expected_shape[1]
    assert (
        expected_size == tuple(swizzled_sf.shape)[0]
    ), f"Expected size {expected_size}, got {tuple(swizzled_sf.shape)[0]}"
    assert_equal = functools.partial(paddle.allclose, rtol=0, atol=0)
    assert_equal(swizzled_sf.reshape(expected_shape), ref_swizzled_sf)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sf_use_ue8m0", [True, False])
@paddle.no_grad()
def test_e2m1_dequantization(
    shape: tuple[int, int], seed: int, device: str, sf_use_ue8m0: bool
) -> None:
    """Test roundtrip: fp4_quantize -> e2m1_and_ufp8sf_scale_to_float."""
    if not is_sm100a_supported(device2str("cuda")):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    paddle.device.set_device(device=device2str(device))
    paddle.seed(seed=seed)
    m, n = shape
    x = paddle.randn(shape=(m, n), dtype="float16")
    tensor_amax = paddle.abs(x=x)._max().to("float32")
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    is_sf_swizzled_layout = True
    block_size = 32 if sf_use_ue8m0 else 16
    quantized_tensor, scale_factors = fp4_quantize(
        x, global_scale, block_size, sf_use_ue8m0, is_sf_swizzled_layout
    )
    ufp8_type = 0 if sf_use_ue8m0 else 1
    dequantized_tensor = e2m1_and_ufp8sf_scale_to_float(
        quantized_tensor,
        scale_factors,
        1 / global_scale,
        sf_vec_size=block_size,
        ufp8_type=ufp8_type,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )
    dequantized_tensor = dequantized_tensor.to(device)
    x_float32 = x.to("float32")
    assert tuple(dequantized_tensor.shape) == tuple(
        x.shape
    ), f"Shape mismatch: expected {tuple(x.shape)}, got {tuple(dequantized_tensor.shape)}"
    assert (
        dequantized_tensor.dtype == "float32"
    ), f"Expected float32, got {dequantized_tensor.dtype}"
    assert (
        not paddle.isnan(x=dequantized_tensor).astype("bool").any()
    ), "Dequantized tensor contains NaN values"
    assert (
        not paddle.isinf(x=dequantized_tensor).astype("bool").any()
    ), "Dequantized tensor contains Inf values"
    assert paddle.allclose(
        x=dequantized_tensor, y=x_float32, rtol=0.3, atol=0.5
    ).item(), "Quantize -> dequantize roundtrip failed"


@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_mxfp4_quantize_roundtrip(device: str):
    if not is_sm100a_supported(device2str(device)):
        pytest.skip("Nvfp4 Requires compute capability of 10 or above")
    x = paddle.randn(shape=(128, 64), dtype="bfloat16") / 10
    quant_a, sfs = mxfp4_quantize(x)
    dq_a = mxfp4_dequantize(quant_a, sfs)
    assert paddle.allclose(
        x=dq_a.cpu().to("float32"), y=x.cpu().to("float32"), rtol=0.3, atol=0.5
    ).item(), "Quantize -> dequantize mxfp4 roundtrip failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
