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
    num_layers: int
    ckv_dim: int = 512
    kpe_dim: int = 64


MODELS = {
    "deepseek_r1": ModelConfig(num_layers=61),
    "deepseek_v2_lite": ModelConfig(num_layers=27),
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
    total_pages = int(25600 / page_len)
>>>>>>    torch.cuda.profiler.start()
    for model_name, model in MODELS.items():
        ckv_page_shape = page_len, model.ckv_dim
        kpe_page_shape = page_len, model.kpe_dim
        ckv_layer_buf = paddle.empty(shape=(total_pages,) + ckv_page_shape, dtype=dtype)
        kpe_layer_buf = paddle.empty(shape=(total_pages,) + kpe_page_shape, dtype=dtype)
        for seqlens in seqlens_:
            ckv = paddle.rand(shape=(sum(seqlens), model.ckv_dim), dtype=dtype)
            kpe = paddle.rand(shape=(sum(seqlens), model.kpe_dim), dtype=dtype)
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
                    tuple(ckv.shape)[0],
                )

            batch_indices, positions = fn_convert()
            convert_latencies = bench_gpu_time(fn_convert)
            convert_latency_ms = np.median(convert_latencies)

>>>>>>            @torch.cuda.nvtx.range(f"append model={model_name}, seqlens={seqlens}")
            def fn() -> None:
                flashinfer.append_paged_mla_kv_cache(
                    ckv,
                    kpe,
                    batch_indices,
                    positions,
                    ckv_layer_buf,
                    kpe_layer_buf,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                )

            latencies = bench_gpu_time(fn)
            latency_ms = np.median(latencies)
            all_layers_latency_ms = convert_latency_ms + latency_ms * model.num_layers
            throughput = (
                (ckv.size + kpe.size)
                * ckv.element_size()
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
