import sys


import paddle
from flashinfer.paddle_utils import *

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
import pytest

import flashinfer.fused_moe as fused_moe
from flashinfer import (fp4_quantize, mxfp4_dequantize, mxfp4_dequantize_host,
                        mxfp4_quantize, mxfp8_dequantize_host, mxfp8_quantize)

FLOAT4_E2M1_MAX = 6.0
>>>>>>FLOAT8_E4M3_MAX = paddle.finfo(dtype=torch.float8_e4m3fn).max
>>>>>>FP8_DTYPE = torch.float8_e4m3fn


def dynamic_per_tensor_fp8_quant(
    x: paddle.to_tensor,
) -> tuple[paddle.to_tensor, paddle.to_tensor]:
    fp8_traits_max = FLOAT8_E4M3_MAX
    fp8_traits_min = -FLOAT8_E4M3_MAX
    fp8_max = paddle.to_tensor(data=fp8_traits_max).astype(dtype="float32")
    one = paddle.to_tensor(data=1.0).astype(dtype="float32")
    x_max = x.abs()._max().astype(dtype="float32")
    scale = x_max / fp8_max
    iscale = one / scale
    out = (
        (x.astype(dtype="float32") * iscale)
        .clip(min=fp8_traits_min, max=fp8_traits_max)
        .to(FP8_DTYPE)
    )
    return out, scale.view((1,))


def gen_tensor(shape, dtype, stype=None, scale=1.0):
    x = paddle.randn(shape=shape, dtype=dtype).cuda() * scale
    return x.to(stype) if stype else x


def cast_to_representable(x):
    x_q, x_scale = dynamic_per_tensor_fp8_quant(x)
    x = x_q.to(x.dtype) * x_scale.to(x.dtype)
    return x


def convert_swizzled_to_linear(a_sf_swizzled: paddle.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = paddle.reshape(x=a_sf_swizzled, shape=(1, m_tiles, k_tiles, 32, 4, 4))
    tmp = paddle.transpose(x=tmp, perm=(0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    assert tensor_fp4.dtype == "uint8"
    m, packed_k = tuple(tensor_fp4.shape)
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
>>>>>>    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to("float32") / global_scale
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(axis=-1)).reshape(m, k)
    return out.to(dtype=dtype)


def break_fp4_bytes(a, dtype):
    assert a.dtype == "uint8"
    m, n = tuple(a.shape)
    a_flat = a.flatten()
    high = (a_flat & 240) >> 4
    low = a_flat & 15
    combined = paddle.stack(x=(low, high), axis=1).flatten()
    signs = (combined & 8).to("bool")
    abs_vals = (combined & 7).to("int64")
    kE2M1ToFloat = paddle.to_tensor(
        data=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype="float32"
    )
    kE2M1 = kE2M1ToFloat.to(device=a.place)
    values = kE2M1[abs_vals] * paddle.where(condition=signs, x=-1.0, y=1.0)
    return values.reshape(m, n * 2).to(dtype=dtype)


def compute_routing(
    router_logits: paddle.Tensor, top_k: int
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = paddle.nn.functional.softmax(
        x=router_logits, axis=1, dtype="float32"
    )
    routing_weights, selected_experts = paddle.topk(x=routing_weights, k=top_k, axis=-1)
    routing_weights /= routing_weights.sum(axis=-1, keepdim=True)
    routing_weights = routing_weights.astype(dtype="float32")
    return routing_weights, selected_experts


def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids):
    B, D = tuple(a.shape)
    a = a.view(B, -1, D).tile(repeat_times=[1, topk, 1]).reshape(-1, D)
    out = paddle.zeros(shape=[B * topk, tuple(w2.shape)[1]], dtype=a.dtype)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(tuple(w1.shape)[0]):
        mask = topk_ids == i
        if mask.sum():
            m = tuple(w1[i].shape)[0]
            assert m % 2 == 0
            w1_expert, w3_expert = w1[i][m // 2 :, :], w1[i][: m // 2, :]
            inter = paddle.nn.functional.silu(x=a[mask] @ w1_expert.t()) * (
                a[mask] @ w3_expert.t()
            )
            inter_gs = paddle.to_tensor(data=1.0).cuda()
            inter_q, inter_blockscale = fp4_quantize(inter, inter_gs)
            inter = dequantize_nvfp4_to_dtype(
                inter_q,
                inter_blockscale,
                inter_gs,
                dtype=inter.dtype,
                device=inter.place,
                block_size=16,
            ).cuda()
            out[mask] = inter @ w2[i].transpose(perm=dim2perm(w2[i].ndim, 0, 1))
    return (
        out.view(B, -1, tuple(w2.shape)[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(axis=1)


def compute_with_experts(
    num_experts,
    x,
    w31_weight,
    w2_weight,
    selected_experts,
    routing_weights,
    alpha=None,
    beta=None,
    limit=None,
):
    results = paddle.zeros_like(x=x)
    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = paddle.where(condition=mask)
        w31_expert = w31_weight[expert_id]
        w2_expert = w2_weight[expert_id]
        w3_expert, w1_expert = paddle.chunk(x=w31_expert, chunks=2, axis=0)
        expert_inputs = x[batch_idx]
        if alpha is not None and limit is not None and beta is not None:
            x1 = expert_inputs @ w1_expert.t()
            x1 = x1.clip_(min=None, max=limit)
            x1_scaled = x1 * paddle.nn.functional.sigmoid(x=alpha * x1)
            x2 = expert_inputs @ w3_expert.t()
            x2 = x2.clip_(min=-limit, max=limit) + beta
            inter = x1_scaled * x2
        else:
            inter = paddle.nn.functional.silu(x=expert_inputs @ w1_expert.t()) * (
                expert_inputs @ w3_expert.t()
            )
        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results.view_as(other=x)


BATCH_SIZES = [1]
HIDDEN_SIZES = [128]
NUM_EXPERTS = [2]
TOP_K_VALUES = [2]
INTERMEDIATE_SIZES = [128]
EP_NUM_EXPERTS = [8]
EP_TOP_K = [2]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe(batch_size, hidden_size, num_experts, top_k, intermediate_size):
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )
    paddle.seed(seed=42)
    x = paddle.randn(shape=[batch_size, hidden_size], dtype="float16").cuda() / 5
    router_logits = paddle.randn(
        shape=[batch_size, num_experts], dtype="float32"
    ).cuda()
    w31_weight = (
        paddle.randn(
            shape=[num_experts, 2 * intermediate_size, hidden_size], dtype="float16"
        ).cuda()
        / 5
    )
    w2_weight = (
        paddle.randn(
            shape=[num_experts, hidden_size, intermediate_size], dtype="float16"
        ).cuda()
        / 5
    )
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )
    flash_output = paddle.empty_like(x=ref_output)
    flash_output = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to("int32"),
        routing_weights,
        w31_weight,
        w2_weight,
        flash_output.dtype,
        output=flash_output,
        quant_scales=None,
    )
    assert paddle.allclose(
        x=ref_output, y=flash_output[0], rtol=0.01, atol=0.01
    ).item(), ""


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
>>>>>>@pytest.mark.parametrize("otype, wtype", [("float16", torch.float8_e4m3fn)])
def test_moe_fp8(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, otype, wtype
):
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )
    paddle.seed(seed=42)
    input_shape = batch_size, hidden_size
    w31_shape = num_experts, 2 * intermediate_size, hidden_size
    w2_shape = num_experts, hidden_size, intermediate_size
    x = cast_to_representable(gen_tensor(input_shape, otype))
    router_logits = gen_tensor((batch_size, num_experts), otype)
    w31_weight = gen_tensor(w31_shape, otype, wtype)
    w2_weight = gen_tensor(w2_shape, otype, wtype)
    w31_scales = paddle.empty(shape=[num_experts, 2], dtype=otype).cuda()
    w2_scales = paddle.empty(shape=[num_experts, 1], dtype=otype).cuda()
    w31_dequantized = gen_tensor(w31_shape, otype)
    w2_dequantized = gen_tensor(w2_shape, otype)
    for expert_id in range(num_experts):
        w31 = cast_to_representable(gen_tensor(w31_shape[1:], otype, scale=0.1))
        w2 = cast_to_representable(gen_tensor(w2_shape[1:], otype, scale=0.09))
        w31_quant, s31 = dynamic_per_tensor_fp8_quant(w31)
        w2_quant, s2 = dynamic_per_tensor_fp8_quant(w2)
        paddle.assign(w31_quant, output=w31_weight.data[expert_id])
        paddle.assign(w2_quant, output=w2_weight.data[expert_id])
        paddle.assign(s31, output=w31_scales.data[expert_id])
        paddle.assign(s2, output=w2_scales.data[expert_id])
        paddle.assign(
            paddle.multiply(x=w31_quant.to(dtype=otype), y=paddle.to_tensor(s31)),
            output=w31_dequantized.data[expert_id],
        )
        paddle.assign(
            paddle.multiply(x=w2_quant.to(dtype=otype), y=paddle.to_tensor(s2)),
            output=w2_dequantized.data[expert_id],
        )
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    ref_output = compute_with_experts(
        num_experts,
        x,
        w31_dequantized,
        w2_dequantized,
        selected_experts,
        routing_weights,
    )
    flash_output = paddle.empty_like(x=ref_output)
    _, w1_scales = paddle.chunk(x=w31_scales, chunks=2, axis=-1)
    x_quant, hidden_states_scale = dynamic_per_tensor_fp8_quant(x)
    hidden_states_scale = paddle.to_tensor(data=hidden_states_scale[0]).cuda()
    quant_scales = [
        paddle.squeeze(x=w1_scales * hidden_states_scale).astype(dtype="float32"),
        paddle.to_tensor(data=1.0).cuda(),
        paddle.squeeze(x=1.0 * w2_scales).astype(dtype="float32"),
        hidden_states_scale,
    ]
    _ = fused_moe.cutlass_fused_moe(
        x_quant,
        selected_experts.to("int32"),
        routing_weights,
        w31_weight,
        w2_weight,
        otype,
        quant_scales=quant_scales,
        output=flash_output,
    )
    assert paddle.allclose(x=ref_output, y=flash_output, rtol=0.1, atol=0.1).item(), ""


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize(
    "otype, wtype",
>>>>>>    [("float16", torch.float8_e4m3fn), ("bfloat16", torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("quantized_input", [False, True])
@pytest.mark.skipif(
    paddle.device.cuda.get_device_capability()[0] != 10,
    reason="NVFP4 is only supported on SM100",
)
def test_moe_nvfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    wtype,
    quantized_input,
):
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )
    paddle.seed(seed=42)
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    w1 = paddle.randn(shape=(e, 2 * n, k), dtype=otype) / 10
    w1_cutlass = paddle.concat(x=(w1[:, n:, :], w1[:, :n, :]), axis=1).contiguous()
    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = paddle.empty(
>>>>>>        shape=(e, sf_w1_2n, sf_w1_k), dtype=torch.float8_e4m3fn
    )
    w1_blockscale_cutlass = paddle.empty(
>>>>>>        shape=(e, sf_w1_2n, sf_w1_k), dtype=torch.float8_e4m3fn
    )
    w2 = paddle.randn(shape=(e, k, n), dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
>>>>>>    w2_blockscale = paddle.empty(shape=(e, sf_w2_k, sf_w2_n), dtype=torch.float8_e4m3fn)
    w1_q = paddle.empty(shape=(e, 2 * n, k // 2), dtype="uint8")
    w1_q_cutlass = paddle.empty(shape=(e, 2 * n, k // 2), dtype="uint8")
    w2_q = paddle.empty(shape=(e, k, n // 2), dtype="uint8")
    w1_gs = paddle.empty(shape=(e,), dtype="float32")
    w2_gs = paddle.empty(shape=(e,), dtype="float32")
    for expert in range(e):
        w1_amax = paddle.abs(x=w1)._max().to("float32")
        w2_amax = paddle.abs(x=w2)._max().to("float32")
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])
        w1_q_cutlass[expert], w1_blockscale_cutlass[expert] = fp4_quantize(
            w1_cutlass[expert], w1_gs[expert]
        )
        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])
    x = paddle.randn(shape=[m, k], dtype=otype).cuda()
    a1_gs = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / paddle.abs(x=x)._max().to("float32").cuda()
    )
    a1_gs = paddle.to_tensor(data=1.0, dtype="float32", place="gpu")
    a2_gs = paddle.to_tensor(data=1.0, dtype="float32", place="gpu")
    router_logits = paddle.randn(shape=[m, e], dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    flash_output = paddle.zeros_like(x=x)
    quant_scales = [
        a1_gs,
        w1_blockscale.view("int32"),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view("int32"),
        1.0 / (a2_gs * w2_gs),
    ]
    hidden_states = x
    input_sf = None
    if quantized_input:
        hidden_states, input_sf = fp4_quantize(x, a1_gs)
    _ = fused_moe.cutlass_fused_moe(
        hidden_states,
        selected_experts.to("int32"),
        routing_weights,
        w1_q.contiguous().view("int64"),
        w2_q.contiguous().view("int64"),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        output=flash_output,
    )
    a_fp4, a_scale_interleaved = fp4_quantize(x, a1_gs)
    _, m_k = tuple(a_fp4.shape)
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a1_gs,
        dtype=otype,
        device=x.place,
        block_size=quant_blocksize,
    )
    w1_d = paddle.empty(shape=(e, 2 * n, k), dtype=otype)
    w2_d = paddle.empty(shape=(e, k, n), dtype=otype)
    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.place,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.place,
            block_size=quant_blocksize,
        )
    w1_q_cutlass = paddle.concat(
        x=(w1_q[:, n:, :], w1_q[:, :n, :]), axis=1
    ).contiguous()
    w1_blockscale_cutlass = paddle.concat(
        x=(w1_blockscale[:, n:, :], w1_blockscale[:, :n, :]), axis=1
    ).contiguous()
    ref_output = torch_moe_nvfp4(
        a_in_dtype, w1_d, w2_d, top_k, routing_weights, selected_experts
    )
    assert paddle.allclose(x=ref_output, y=flash_output, rtol=0.2, atol=0.2).item(), ""


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", EP_NUM_EXPERTS)
@pytest.mark.parametrize("top_k", EP_TOP_K)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_expert_parallel(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """
    Test expert parallelism with X GPUs and Y experts.
    Each GPU handles one expert and results are reduced.

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts (must be 2 for this test)
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
        activation: Activation function type
    """
    ep_size = num_experts // 2
    paddle.seed(seed=42)
    x = paddle.randn(shape=[batch_size, hidden_size], dtype="float16").cuda()
    w31_weight = (
        paddle.randn(
            shape=[num_experts, 2 * intermediate_size, hidden_size], dtype="float16"
        ).cuda()
        / 10
    )
    w2_weight = (
        paddle.randn(
            shape=[num_experts, hidden_size, intermediate_size], dtype="float16"
        ).cuda()
        / 10
    )
    selected_experts = paddle.stack(
        x=[paddle.randperm(n=num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()
    routing_weights = paddle.randn(shape=(batch_size, top_k)).cuda()
    routing_weights = paddle.nn.functional.softmax(x=routing_weights, axis=1)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )
    outputs = []
    flash_output = paddle.zeros_like(x=ref_output)
    for ep_rank in range(ep_size):
        out_hidden_states_local = paddle.zeros_like(x=x)
        experts_per_rank = num_experts // ep_size
        expert_start = ep_rank * experts_per_rank
        expert_end = expert_start + experts_per_rank
        w31_weight_local = w31_weight[expert_start:expert_end, :]
        w2_weight_local = w2_weight[expert_start:expert_end, :]
        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to("int32"),
            routing_weights,
            w31_weight_local.contiguous(),
            w2_weight_local.contiguous(),
            x.dtype,
            ep_size=ep_size,
            ep_rank=ep_rank,
            quant_scales=None,
            output=out_hidden_states_local,
        )
        outputs.append(out_hidden_states_local)
    for ep_rank in range(ep_size):
        flash_output += outputs[ep_rank]
    assert paddle.allclose(x=ref_output, y=flash_output, rtol=0.1, atol=0.1).item(), ""


TP_SIZES = [2, 4]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_tensor_parallel(
    batch_size, hidden_size, num_experts, tp_size, intermediate_size
):
    """
    Test tensor parallelism with:
    - w31 sharded along second dimension (non-contracting)
    - w2 sharded along third dimension (contracting)
    - All-reduce to sum partial results

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
        activation: Activation function type
    """
    paddle.seed(seed=42)
    top_k = 2
    x = paddle.randn(shape=[batch_size, hidden_size], dtype="float16").cuda()
    w31_weight = (
        paddle.randn(
            shape=[num_experts, 2 * intermediate_size, hidden_size], dtype="float16"
        ).cuda()
        / 10
    )
    w2_weight = (
        paddle.randn(
            shape=[num_experts, hidden_size, intermediate_size], dtype="float16"
        ).cuda()
        / 10
    )
    selected_experts = paddle.stack(
        x=[paddle.randperm(n=num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()
    routing_weights = paddle.randn(shape=(batch_size, top_k)).cuda()
    routing_weights = paddle.nn.functional.softmax(x=routing_weights, axis=1)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )
    outputs = []
    for tp_rank in range(tp_size):
        out_hidden_states_local = paddle.zeros_like(x=x)
        w3_weight, w1_weight = paddle.chunk(x=w31_weight, chunks=2, axis=1)
        w3_shard_size = intermediate_size // tp_size
        w3_start = tp_rank * w3_shard_size
        w3_end = w3_start + w3_shard_size
        w3_weight_local = w3_weight[:, w3_start:w3_end, :]
        w1_shard_size = intermediate_size // tp_size
        w1_start = tp_rank * w1_shard_size
        w1_end = w1_start + w1_shard_size
        w1_weight_local = w1_weight[:, w1_start:w1_end, :]
        w31_weight_local = paddle.concat(x=[w3_weight_local, w1_weight_local], axis=1)
        w2_shard_size = intermediate_size // tp_size
        w2_start = tp_rank * w2_shard_size
        w2_end = w2_start + w2_shard_size
        w2_weight_local = w2_weight[:, :, w2_start:w2_end]
        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to("int32"),
            routing_weights,
            w31_weight_local.contiguous(),
            w2_weight_local.contiguous(),
            x.dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_scales=None,
            output=out_hidden_states_local,
        )
        outputs.append(out_hidden_states_local)
    flash_output = sum(outputs)
    assert paddle.allclose(
        x=ref_output, y=flash_output, rtol=0.01, atol=0.01
    ).item(), ""


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", EP_NUM_EXPERTS)
@pytest.mark.parametrize("top_k", EP_TOP_K)
@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_tensor_expert_parallel(
    batch_size, hidden_size, num_experts, top_k, tp_size, intermediate_size
):
    """
    Test combined tensor parallelism and expert parallelism:
    - Expert parallelism: Distribute experts across GPUs
    - Tensor parallelism: For each expert's weights:
        - w31 sharded along second dimension (non-contracting)
        - w2 sharded along third dimension (contracting)
    - All-reduce to sum partial results

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        tp_size: Number of GPUs for tensor parallelism
        intermediate_size: Intermediate dimension size
    """
    paddle.seed(seed=42)
    x = paddle.randn(shape=[batch_size, hidden_size], dtype="float16").cuda()
    w31_weight = (
        paddle.randn(
            shape=[num_experts, 2 * intermediate_size, hidden_size], dtype="float16"
        ).cuda()
        / 10
    )
    w2_weight = (
        paddle.randn(
            shape=[num_experts, hidden_size, intermediate_size], dtype="float16"
        ).cuda()
        / 10
    )
    selected_experts = paddle.stack(
        x=[paddle.randperm(n=num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()
    routing_weights = paddle.randn(shape=(batch_size, top_k)).cuda()
    routing_weights = paddle.nn.functional.softmax(x=routing_weights, axis=1)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )
    ep_size = num_experts // 2
    outputs = []
    for ep_rank in range(ep_size):
        experts_per_rank = num_experts // ep_size
        expert_start = ep_rank * experts_per_rank
        expert_end = expert_start + experts_per_rank
        w31_weight_ep = w31_weight[expert_start:expert_end, :]
        w2_weight_ep = w2_weight[expert_start:expert_end, :]
        for tp_rank in range(tp_size):
            out_hidden_states_local = paddle.zeros_like(x=x)
            w3_weight, w1_weight = paddle.chunk(x=w31_weight_ep, chunks=2, axis=1)
            w3_shard_size = intermediate_size // tp_size
            w3_start = tp_rank * w3_shard_size
            w3_end = w3_start + w3_shard_size
            w3_weight_local = w3_weight[:, w3_start:w3_end, :]
            w1_shard_size = intermediate_size // tp_size
            w1_start = tp_rank * w1_shard_size
            w1_end = w1_start + w1_shard_size
            w1_weight_local = w1_weight[:, w1_start:w1_end, :]
            w31_weight_local = paddle.concat(
                x=[w3_weight_local, w1_weight_local], axis=1
            )
            w2_shard_size = intermediate_size // tp_size
            w2_start = tp_rank * w2_shard_size
            w2_end = w2_start + w2_shard_size
            w2_weight_local = w2_weight_ep[:, :, w2_start:w2_end]
            out_hidden_states_local = fused_moe.cutlass_fused_moe(
                x.contiguous(),
                selected_experts.to("int32"),
                routing_weights,
                w31_weight_local.contiguous(),
                w2_weight_local.contiguous(),
                x.dtype,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                quant_scales=None,
            )
            outputs.append(out_hidden_states_local[0])
    flash_output = sum(outputs)
    assert paddle.allclose(
        x=ref_output, y=flash_output, rtol=0.01, atol=0.01
    ).item(), ""


def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


def per_block_cast_to_fp8(
    x: paddle.Tensor, block_size_n: int = 128
) -> tuple[paddle.Tensor, paddle.Tensor]:
    assert x.dim() == 2
    m, n = tuple(x.shape)
    x_padded = paddle.zeros(
        shape=(ceil_div(m, 128) * 128, ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.shape[1] // 128, block_size_n)
    x_amax = (
        x_view.abs()
        .astype(dtype="float32")
        .amax(axis=(1, 3), keepdim=True)
        .clip(min=0.0001)
    )
>>>>>>    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(other=x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.shape[0], x_view.shape[2])
    return x_scaled_sub, scales


>>>>>>def per_token_group_quant_fp8(x, group_size, eps=1e-10, dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert (
        tuple(x.shape)[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"
    finfo = paddle.finfo(dtype=dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    x_ = x.reshape(x.size // group_size, group_size)
    amax = (
        (x_.abs().max(keepdim=True, axis=-1), x_.abs().argmax(keepdim=True, axis=-1))[0]
        .clip(min=eps)
        .to("float32")
    )
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clip(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(tuple(x.shape))
    x_s = x_s.reshape(tuple(x.shape)[:-1] + (tuple(x.shape)[-1] // group_size,))
    return x_q, x_s


def dequantize_block(
    x_quant: paddle.Tensor,
    scales: paddle.Tensor,
    dtype: paddle.dtype,
    original_shape: tuple,
) -> paddle.Tensor:
    """
    Dequantize a block-quantized tensor.

    Args:
        x_quant: Quantized tensor
        scales: Block scaling factors
        dtype: Target dtype for dequantization
        original_shape: Original shape of the tensor before padding

    Returns:
        torch.Tensor: Dequantized tensor
    """

    def transform_dim(a: paddle.Tensor, dim: int = -1) -> paddle.Tensor:
        if dim != -1:
            a = a.transpose(perm=dim2perm(a.ndim, dim, -1))
        a_broadcasted = a.unsqueeze(axis=-1).expand(shape=[*tuple(a.shape), 128])
        a_reshaped = a_broadcasted.reshape(
            *tuple(a.shape)[:-1], tuple(a.shape)[-1] * 128
        )
        if dim != -1:
            a_reshaped = a_reshaped.transpose(perm=dim2perm(a_reshaped.ndim, dim, -1))
        return a_reshaped

    if x_quant.dim() == 2:
        batch_size, hidden_size = tuple(x_quant.shape)
        num_blocks = (hidden_size + 127) // 128
        scales = scales.view(batch_size, num_blocks, 1).expand(shape=[-1, -1, 128])
        scales = scales[:, :, : hidden_size % 128] if hidden_size % 128 != 0 else scales
    else:
        *_dims, in_dim, out_dim = tuple(x_quant.shape)
        scales = transform_dim(scales, -1)
        scales = transform_dim(scales, -2)
        if in_dim % 128 != 0:
            scales = scales[..., : in_dim % 128, :]
        if out_dim % 128 != 0:
            scales = scales[..., :, : out_dim % 128]
    x_dequant = x_quant.to(dtype) * scales.to(dtype)
    return x_dequant.view(original_shape)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.skipif(
    paddle.device.cuda.get_device_capability()[0] != 10,
    reason="FP8 block scaling is only supported on SM100",
)
def test_moe_fp8_block_scaling(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """
    Test MoE with FP8 block scaling (Deepseek style):
    - Activation: 128x1 blocks
    - Weights: 128x128 blocks
    - Each block has its own scaling factor

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
        Only support bf16 for hidden_states
    """
    paddle.seed(seed=42)
    otype = "bfloat16"
    x = paddle.randn(shape=[batch_size, hidden_size], dtype=otype).cuda()
    w31_weight = (
        paddle.randn(
            shape=[num_experts, 2 * intermediate_size, hidden_size], dtype=otype
        ).cuda()
        / 10
    )
    w2_weight = (
        paddle.randn(
            shape=[num_experts, hidden_size, intermediate_size], dtype=otype
        ).cuda()
        / 10
    )
    selected_experts = paddle.stack(
        x=[paddle.randperm(n=num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()
    routing_weights = paddle.randn(shape=(batch_size, top_k)).cuda()
    routing_weights = paddle.nn.functional.softmax(x=routing_weights, axis=1)
    _ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )
    x_quant, x_scales = per_token_group_quant_fp8(x, group_size=128)
    w31_dequant = paddle.empty_like(x=w31_weight)
    w2_dequant = paddle.empty_like(x=w2_weight)
>>>>>>    w31_quant = paddle.empty_like(x=w31_weight).to(torch.float8_e4m3fn)
>>>>>>    w2_quant = paddle.empty_like(x=w2_weight).to(torch.float8_e4m3fn)
    w31_scales = paddle.randn(
        shape=[
            num_experts,
            ceil_div(2 * intermediate_size, 128),
            ceil_div(hidden_size, 128),
        ],
        dtype="float32",
    ).cuda()
    w2_scales = paddle.randn(
        shape=[
            num_experts,
            ceil_div(hidden_size, 128),
            ceil_div(intermediate_size, 128),
        ],
        dtype="float32",
    ).cuda()
    for expert_id in range(num_experts):
        w31, w31_s = per_block_cast_to_fp8(w31_weight[expert_id, :])
        w2, w2_s = per_block_cast_to_fp8(w2_weight[expert_id, :])
        paddle.assign(w31, output=w31_quant.data[expert_id])
        paddle.assign(w31_s, output=w31_scales.data[expert_id])
        paddle.assign(w2, output=w2_quant.data[expert_id])
        paddle.assign(w2_s, output=w2_scales.data[expert_id])
    x_dequant = dequantize_block(x_quant, x_scales, x.dtype, tuple(x.shape))
    w31_dequant = dequantize_block(
        w31_quant, w31_scales, w31_weight.dtype, tuple(w31_weight.shape)
    )
    w2_dequant = dequantize_block(
        w2_quant, w2_scales, w2_weight.dtype, tuple(w2_weight.shape)
    )
    _ref_output = compute_with_experts(
        num_experts,
        x_dequant,
        w31_dequant,
        w2_dequant,
        selected_experts,
        routing_weights,
    )
    quant_scales = [w31_scales, w2_scales]
    with pytest.raises(
        NotImplementedError,
        match="DeepSeek FP8 Block Scaling is not yet implemented in CUTLASS for Blackwell",
    ):
        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to("int32"),
            routing_weights,
            w31_quant.contiguous(),
            w2_quant.contiguous(),
            otype,
            tp_size=1,
            tp_rank=0,
            use_deepseek_fp8_block_scale=True,
            quant_scales=quant_scales,
        )


def quant_mxfp4_batches(a, num_experts):
    quant_a = []
    sfs = []
    for i in range(num_experts):
        a_fp4, a_sf = mxfp4_quantize(a[i].cuda())
        quant_a.append(a_fp4)
        sfs.append(a_sf)
    result_quant_a = paddle.stack(x=quant_a)
    result_sfs = paddle.stack(x=sfs)
    return result_quant_a, result_sfs


def dequant_mxfp4_batches(mat_fp4: paddle.Tensor, scale_tensor: paddle.Tensor):
    num_batches = mat_fp4.shape[0]
    scale_tensor = scale_tensor.view(num_batches, -1)
    return paddle.stack(
        x=[
            mxfp4_dequantize(mat_fp4[b, :, :], scale_tensor[b, :])
            for b in range(num_batches)
        ]
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("otype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    ("alpha", "beta", "limit"), [(None, None, None), (0.5, 0.0, 7.0), (1.702, 1.0, 7.0)]
)
@pytest.mark.skipif(
    paddle.device.cuda.get_device_capability()[0] != 10,
    reason="MXFP8xMXFP4 is only supported on SM100",
)
def test_moe_mxfp8_mxfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    alpha,
    beta,
    limit,
):
    """
    Test MoE with MXFP8 activations and MXFP4 weights.
    Uses mxfp8_quantize for activations and fp4_quantize for weights.
    """
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )
    paddle.seed(seed=42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    x = paddle.randn(shape=[m, k], dtype=otype).cuda()
    w1 = paddle.randn(shape=(e, 2 * n, k), dtype=otype) / 10
    w2 = paddle.randn(shape=(e, k, n), dtype=otype) / 10
    mxfp8_x, mxfp8_x_sf = mxfp8_quantize(x, True, 32)
    mxfp4_w1, mxfp4_w1_scale = quant_mxfp4_batches(w1, e)
    mxfp4_w2, mxfp4_w2_scale = quant_mxfp4_batches(w2, e)
    router_logits = paddle.randn(shape=[m, e], dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    fake_input_scale = paddle.ones(shape=e)
    quant_scales = [
        mxfp4_w1_scale.view("int32"),
        fake_input_scale,
        mxfp4_w2_scale.view("int32"),
        fake_input_scale,
    ]
    flash_output = paddle.zeros_like(x=x)
    if alpha is not None and limit is not None and beta is not None:
        alpha_t = paddle.ones(shape=e) * alpha
        limit_t = paddle.ones(shape=e) * limit
        beta_t = paddle.ones(shape=e) * beta
    else:
        alpha_t = None
        limit_t = None
        beta_t = None
    _ = fused_moe.cutlass_fused_moe(
        mxfp8_x,
        selected_experts.to("int32"),
        routing_weights,
        mxfp4_w1.contiguous().view("int64"),
        mxfp4_w2.contiguous().view("int64"),
        otype,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        input_sf=mxfp8_x_sf,
        use_mxfp8_act_scaling=True,
        output=flash_output,
    )
    dq_mxfp8_x = (
        mxfp8_dequantize_host(
            mxfp8_x.cpu().view("uint8"),
            mxfp8_x_sf.cpu().view("uint8").reshape(-1),
            True,
        )
        .cuda()
        .to(otype)
    )
    dq_mfxp4_w1 = (
        dequant_mxfp4_batches(
            mxfp4_w1.cpu().view("uint8"), mxfp4_w1_scale.cpu().view("uint8").reshape(-1)
        )
        .cuda()
        .to(otype)
    )
    dq_mfxp4_w2 = (
        dequant_mxfp4_batches(
            mxfp4_w2.cpu().view("uint8"), mxfp4_w2_scale.cpu().view("uint8").reshape(-1)
        )
        .cuda()
        .to(otype)
    )
    ref_output = compute_with_experts(
        e,
        dq_mxfp8_x,
        dq_mfxp4_w1,
        dq_mfxp4_w2,
        selected_experts,
        routing_weights,
        alpha,
        beta,
        limit,
    )
    assert paddle.allclose(x=ref_output, y=flash_output, rtol=0.1, atol=0.1).item(), ""


def dequant_mxfp4_batches_host(mat_fp4: paddle.Tensor, scale_tensor: paddle.Tensor):
    return paddle.stack(
        x=[
            mxfp4_dequantize_host(mat_fp4[b, :, :], scale_tensor[b, :, :])
            for b in range(mat_fp4.shape[0])
        ]
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize(
    ("alpha", "beta", "limit"), [(None, None, None), (0.5, 0.0, 7.0), (1.702, 1.0, 7.0)]
)
@pytest.mark.skipif(
    paddle.device.cuda.get_device_capability()[0] != 9,
    reason="BF16xMXFP4 is only supported on SM90",
)
def test_moe_bf16_mxfp4(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, alpha, beta, limit
):
    """
    Test MoE with bf16 activations and MXFP4 weights.
    Uses bf16 for activations and fp4_quantize for weights.
    """
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )
    paddle.seed(seed=42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    x = paddle.randn(shape=[m, k], dtype="bfloat16").cuda()
    w1 = paddle.randint(low=0, high=256, shape=(e, 2 * n, k // 2), dtype="uint8")
    w2 = paddle.randint(low=0, high=256, shape=(e, k, n // 2), dtype="uint8")
    w1_scale = paddle.randint(
        low=118, high=123, shape=(e, 2 * n, k // 32), dtype="uint8"
    )
    w2_scale = paddle.randint(low=118, high=123, shape=(e, k, n // 32), dtype="uint8")
    router_logits = paddle.randn(shape=[m, e], dtype="bfloat16").cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    flash_output = paddle.zeros_like(x=x)
    if alpha is not None and limit is not None and beta is not None:
        alpha_t = paddle.ones(shape=e) * alpha
        limit_t = paddle.ones(shape=e) * limit
        beta_t = paddle.ones(shape=e) * beta
    else:
        alpha_t = None
        limit_t = None
        beta_t = None
    pad_size = hidden_size - tuple(x.shape)[1]
    x_pad = paddle.nn.functional.pad(x=x, pad=(0, pad_size), pad_from_left_axis=False)
    quant_scales = [w1_scale.view("int32"), w2_scale.view("int32")]
    _ = fused_moe.cutlass_fused_moe(
        x_pad,
        selected_experts.to("int32"),
        routing_weights,
        w1.contiguous().view("uint8"),
        w2.contiguous().view("uint8"),
        "bfloat16",
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=flash_output,
    )
    dq_mfxp4_w1 = (
        dequant_mxfp4_batches_host(w1.cpu(), w1_scale.cpu()).cuda().to("bfloat16")
    )
    dq_mfxp4_w2 = (
        dequant_mxfp4_batches_host(w2.cpu(), w2_scale.cpu()).cuda().to("bfloat16")
    )
    ref_output = compute_with_experts(
        e,
        x,
        dq_mfxp4_w1,
        dq_mfxp4_w2,
        selected_experts,
        routing_weights,
        alpha,
        beta,
        limit,
    )
    assert paddle.allclose(x=ref_output, y=flash_output, rtol=0.1, atol=0.1).item(), ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
