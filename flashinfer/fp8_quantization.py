import functools
from types import SimpleNamespace
from typing import Optional, Tuple

import paddle

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .utils import device_support_pdl, register_custom_op, register_fake_op


def gen_mxfp8_quantization_sm100_module() -> JitSpec:
    return gen_jit_spec(
        "mxfp8_quantization_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/thop/fp8Quantize.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/kernels/quantization.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags
        + ["-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4"],
        extra_cflags=["-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4"],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
        ],
    )


@functools.cache
def get_mxfp8_quantization_sm100_module():
    module = gen_mxfp8_quantization_sm100_module().build_and_load()

    @register_custom_op("flashinfer::mxfp8_quantize_sm100", mutates_args="")
    def mxfp8_quantize_sm100(
        input: paddle.Tensor,
        is_sf_swizzled_layout: bool = True,
        alignment: int = 32,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Quantize input tensor to MxFP8 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
            alignment (int, optional): sfVecSize. Defaults to 32. Note that alignment is not used in the host kernel.
            enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
                If None, automatically detects based on device capability. Defaults to None.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Quantized tensor of shape [M, K] with dtype FLOAT8_E4M3
                - Scale factors tensor with shape determined by layout and sf_vec_size
        """
        if input.device.type == "cpu":
            return module.mxfp8_quantize_host(input, is_sf_swizzled_layout)
        else:
            if enable_pdl is None:
                enable_pdl = device_support_pdl(input.place)
            return module.mxfp8_quantize(
                input, is_sf_swizzled_layout, alignment, enable_pdl
            )

    @register_fake_op("flashinfer::mxfp8_quantize_sm100")
    def _fake_mxfp8_quantize_sm100(
        input: paddle.Tensor, is_sf_swizzled_layout: bool = True, alignment: int = 32
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        m, k = tuple(input.shape)
        return paddle.empty(shape=[m, k], dtype="int64"), paddle.empty(
            shape=[m * k // 32], dtype="int32"
        )

    @register_custom_op("flashinfer::mxfp8_dequantize_host_sm100", mutates_args=("",))
    def mxfp8_dequantize_host_sm100(
        input: paddle.Tensor,
        scale_tensor: paddle.Tensor,
        is_sf_swizzled_layout: bool = True,
    ) -> paddle.Tensor:
        """Dequantize input tensor from MxFP8 format.

        Args:
            input (torch.Tensor): Input tensor of shape [M, K] with dtype FLOAT8_E4M3.
            scale_tensor (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size.
            is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.

        Returns:
            torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.
        """
        return module.mxfp8_dequantize_host(input, scale_tensor, is_sf_swizzled_layout)

    @register_fake_op("flashinfer::mxfp8_dequantize_host_sm100")
    def _fake_mxfp8_dequantize_host_sm100(
        input: paddle.Tensor,
        scale_tensor: paddle.Tensor,
        is_sf_swizzled_layout: bool = True,
    ) -> paddle.Tensor:
        return paddle.empty(
            shape=[tuple(input.shape)[0], tuple(input.shape)[1]], dtype="float32"
        )

    return SimpleNamespace(
        mxfp8_quantize_sm100=mxfp8_quantize_sm100,
        mxfp8_dequantize_host_sm100=mxfp8_dequantize_host_sm100,
    )


def mxfp8_quantize(
    input: paddle.Tensor,
    is_sf_swizzled_layout: bool = True,
    alignment: int = 32,
    enable_pdl: Optional[bool] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Quantize input tensor to MxFP8 format.

    This function implements MxFP8 quantization that converts input tensors to a compressed MxFP8 format
    with associated scale factors. It supports various input data types and scale factor layouts.

    Args:
        input (torch.Tensor): Input tensor of shape [M, K] with dtype fp16/bf16/fp8_quantized.
        is_sf_swizzled_layout (bool, optional): Whether to use swizzled layout for scale factors. Defaults to True.
        alignment (int, optional): sfVecSize. Defaults to 32.
        enable_pdl (Optional[bool], optional): Whether to enable PDL (Programmatic Dependent Launch).
            If None, automatically detects based on device capability. Defaults to None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [M, K] with dtype FLOAT8_E4M3
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    sf_vec_size = 32
    assert tuple(input.shape)[-1] % sf_vec_size == 0
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.place)
    x_q, sf = get_mxfp8_quantization_sm100_module().mxfp8_quantize_sm100(
        input, is_sf_swizzled_layout, alignment, enable_pdl
    )
    return x_q, sf


def mxfp8_dequantize_host(
    input: paddle.Tensor,
    scale_tensor: paddle.Tensor,
    is_sf_swizzled_layout: bool = True,
) -> paddle.Tensor:
    """Dequantize input tensor from MxFP8 format.

    This function performs dequantization by converting a packed FP8 tensor in MxFP8 format
    back to float values using the associated scale factors.

    Args:
        input (torch.Tensor): Packed FP8 tensor in MxFP8 format of shape [M, K] with dtype FLOAT8_E4M3.
        scale_tensor (torch.Tensor): Scale factors tensor with shape determined by layout and sf_vec_size.
        is_sf_swizzled_layout (bool, optional): Whether scale factors use swizzled layout. Defaults to True.

    Returns:
        torch.Tensor: Dequantized float tensor of shape [M, K] with dtype float32.

    """
    return get_mxfp8_quantization_sm100_module().mxfp8_dequantize_host_sm100(
        input, scale_tensor, is_sf_swizzled_layout
    )
