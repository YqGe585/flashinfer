import sys

sys.path.append("/home/flashinfer_paddle")
import paddle
from paddle_utils import *

import flashinfer.utils as utils

FLOAT4_E2M1_MAX = 6.0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def cast_from_fp4(x):
    v_2nd = x & 15
    v_1st = x >> 4 & 15
    c = paddle.stack(x=(v_2nd, v_1st), axis=-1)
    new_shape = tuple(c.shape)[:-2] + (tuple(c.shape)[-2] * tuple(c.shape)[-1],)
    lookup_table = paddle.to_tensor(data=E2M1_TO_FLOAT32, place=c.place)
    out = lookup_table[c.to("int64")].reshape(new_shape).to("float32")
    return out


def cast_to_fp4(x):
    sign = paddle.sign(x=x)
    x = paddle.abs(x=x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, paddle.Tensor):
        return paddle.where(
            condition=x == 0, x=paddle.to_tensor(data=0.0, dtype=x.dtype), y=1.0 / x
        )
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_fp4_quant(x, global_scale, block_size, sf_use_ue8m0=False):
    assert isinstance(global_scale, (float, int)) or global_scale.dtype == "float32"
    sliced_shape = tuple(x.shape)[:-1] + (tuple(x.shape)[-1] // block_size, block_size)
    sliced_x = paddle.reshape(x=x, shape=sliced_shape)
    vec_max = (
        paddle.max(keepdim=True, x=paddle.abs(x=sliced_x), axis=-1),
        paddle.argmax(keepdim=True, x=paddle.abs(x=sliced_x), axis=-1),
    )[0].to("float32")
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    if sf_use_ue8m0:
        scale = scale.view("int32") + 8388607 & 2139095040
        scale = scale.view("float32")
    else:
>>>>>>        scale = scale.to(torch.float8_e4m3fn).to("float32")
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))
    scaled_x = sliced_x.to("float32") * output_scale
    clipped_x = paddle.clip(x=scaled_x, min=-6.0, max=6.0).reshape(tuple(x.shape))
    return cast_to_fp4(clipped_x), scale.squeeze(axis=-1)


def recover_swizzled_scales(scale, m, n, block_size, sf_start_index=0):
    assert sf_start_index + m <= tuple(scale.shape)[0]
    full_m = tuple(scale.shape)[0]
    scale_n = n // block_size
    rounded_n = utils.round_up(scale_n, 4)
    tmp = paddle.reshape(x=scale, shape=(1, full_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = paddle.transpose(x=tmp, perm=(0, 1, 4, 3, 2, 5))
    result = paddle.reshape(x=tmp, shape=(full_m, rounded_n)).to("float32")
    return result[sf_start_index : sf_start_index + m, :scale_n]
