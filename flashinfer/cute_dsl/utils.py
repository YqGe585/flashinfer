import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

"""
Copyright (c) 2025 by FlashInfer team.

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
import functools
import importlib.util

import cutlass


def is_cute_dsl_available() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


def get_cutlass_dtype(dtype: str) -> cutlass.dtype:
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
        "float8_e5m2": cutlass.Float8E5M2,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
    }
    return dtype_map[dtype]


def cutlass_to_torch_dtype(cutlass_dtype):
    """
    Return the corresponding torch.dtype per the given DSL type
    """
    torch_dtype = getattr(torch, cutlass_dtype.__name__.lower(), None)
    torch_type_map = {
        cutlass.TFloat32: "float32",
        cutlass.Float32: "float32",
        cutlass.Float16: "float16",
        cutlass.BFloat16: "bfloat16",
>>>>>>        cutlass.Float8E5M2: torch.float8_e5m2,
>>>>>>        cutlass.Float8E4M3FN: torch.float8_e4m3fn,
>>>>>>        cutlass.Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
    }
    if torch_dtype is None:
        torch_dtype = torch_type_map.get(cutlass_dtype)
    if torch_dtype is None:
        raise TypeError(f"{cutlass_dtype} is not supported by torch")
    return torch_dtype


@functools.cache
def get_num_sm(device: str) -> int:
    return paddle.device.cuda.get_device_properties(
        device=device2str(device)
    ).multi_processor_count
