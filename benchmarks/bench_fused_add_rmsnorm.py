import argparse

import numpy as np
import paddle

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@paddle.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 19, 99, 989])
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[111, 500, 1024, 3072, 4096, 8192],
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16"]
    )
    args = parser.parse_args()
    eps = 1e-06
    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            for dtype_str in args.dtypes:
                dtype = getattr(torch, dtype_str)
                x = paddle.randn(shape=(batch_size, hidden_size), dtype=dtype)
                residual = paddle.randn(shape=x.shape, dtype=x.dtype)
                weight = paddle.randn(shape=hidden_size, dtype=dtype)

>>>>>>                @torch.cuda.nvtx.range(
                    f"fused_add_rmsnorm batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype_str}"
                )
                def fn() -> None:
                    flashinfer.fused_add_rmsnorm(x, residual, weight, eps)

                measurements = bench_gpu_time(fn)
                latency_ms = np.median(measurements)
                throughput = (
                    x.size * x.element_size() * 2
                    + residual.size * residual.element_size() * 2
                    + weight.size * weight.element_size()
                ) / (latency_ms * 0.001)
                print(
                    f"batch_size: {batch_size:3},",
                    f"hidden_size: {hidden_size:5},",
                    f"dtype: {dtype_str:8},",
                    f"latency: {latency_ms * 1000.0:2.0f}us,",
                    f"throughput: {throughput * 1e-09:7.3f}GB/s",
                )
        print("---")
>>>>>>    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
