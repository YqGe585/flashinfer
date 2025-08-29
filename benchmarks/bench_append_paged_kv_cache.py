import sys


import argparse
import dataclasses
from typing import Tuple

import numpy as np
import paddle
from flashinfer.paddle_utils import *

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
    num_kv_heads: int
    num_layers: int
    head_dim: int


def _make_70b(tp: int) -> ModelConfig:
    return ModelConfig(num_kv_heads=8 // tp, num_layers=80, head_dim=128)


MODELS = {
    "l1b": ModelConfig(num_kv_heads=8, num_layers=16, head_dim=64),
    "l3b": ModelConfig(num_kv_heads=8, num_layers=28, head_dim=128),
    "l8b": ModelConfig(num_kv_heads=8, num_layers=32, head_dim=128),
    "l70b-tp8": _make_70b(8),
}


@paddle.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--page-len", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    seqlens_ = [
        [1] * args.batch_size,
        [args.seqlen - args.batch_size + 1] + [1] * (args.batch_size - 1),
        [args.seqlen],
        [args.seqlen // args.batch_size] * args.batch_size,
    ]
    seqlen_strlen = max(len(str(seqlens)) for seqlens in seqlens_)
    page_len = int(args.page_len)
    dtype = getattr(torch, args.dtype)
    assert isinstance(dtype, paddle.dtype)
    device = device2str("cuda:0")
    total_pages = int(256000 / page_len)
>>>>>>    torch.cuda.profiler.start()
    for model_name, model in MODELS.items():
        page_shape = 2, page_len, model.num_kv_heads, model.head_dim
        layer_buf = paddle.empty(shape=(total_pages,) + page_shape, dtype=dtype)
        for seqlens in seqlens_:
            k = paddle.rand(
                shape=(sum(seqlens), model.num_kv_heads, model.head_dim), dtype=dtype
            )
            v = paddle.rand(
                shape=(sum(seqlens), model.num_kv_heads, model.head_dim), dtype=dtype
            )
            x_indptr = paddle.to_tensor(data=[0] + seqlens, dtype="int32", place=device)
            x_indptr = paddle.cumsum(x=x_indptr, axis=0, dtype="int32")
            kv_indices_host = []
            kv_indptr_host = [0]
            next_page_id = 0
            for seqlen in seqlens:
                npages = (seqlen + page_len - 1) // page_len
                kv_indices_host.extend(range(next_page_id, next_page_id + npages))
                next_page_id += npages
                kv_indptr_host.append(len(kv_indices_host))
            kv_indices = paddle.to_tensor(
                data=kv_indices_host, dtype="int32", place=device
            )
            kv_indptr = paddle.to_tensor(
                data=kv_indptr_host, dtype="int32", place=device
            )
            kv_last_page_len = paddle.to_tensor(
                data=[((seqlen - 1) % page_len + 1) for seqlen in seqlens],
                dtype="int32",
                place=device,
            )

>>>>>>            @torch.cuda.nvtx.range(f"convert model={model_name}, seqlens={seqlens}")
            def fn_convert() -> Tuple[paddle.Tensor, paddle.Tensor]:
                return flashinfer.get_batch_indices_positions(
                    x_indptr,
                    flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, page_len),
                    tuple(k.shape)[0],
                )

            batch_indices, positions = fn_convert()
            convert_latencies = bench_gpu_time(fn_convert)
            convert_latency_ms = np.median(convert_latencies)

>>>>>>            @torch.cuda.nvtx.range(f"append model={model_name}, seqlens={seqlens}")
            def fn() -> None:
                flashinfer.append_paged_kv_cache(
                    k,
                    v,
                    batch_indices,
                    positions,
                    layer_buf,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                    "NHD",
                )

            latencies = bench_gpu_time(fn)
            latency_ms = np.median(latencies)
            all_layers_latency_ms = convert_latency_ms + latency_ms * model.num_layers
            throughput = (
                k.size
                * k.element_size()
                * sum(1 for _ in ["k", "v"])
                * sum(1 for _ in ["read", "write"])
                / (latency_ms * 0.001)
            )
            print(
                f"model: {model_name:8}",
                f"seqlens: {seqlens!r:{seqlen_strlen}}",
                f"convert: {convert_latency_ms * 1000.0:2.0f}us",
                f"1layer: {latency_ms * 1000.0:2.0f}us",
                f"{model.num_layers}layers: {all_layers_latency_ms * 1000.0:3.0f}us",
                f"throughput: {throughput * 1e-09:8.3f}GB/s",
            )
        print("---")
>>>>>>    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
