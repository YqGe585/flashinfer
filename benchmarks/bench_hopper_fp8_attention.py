import numpy as np
import paddle

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def bench_single_prefill(seq_len, num_heads, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = paddle.randn(shape=[seq_len, num_qo_heads, head_dim], dtype="float16")
    k = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    v = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16")
    sm80_ms, sm90_ms = (
        np.median(
            bench_gpu_time(
                lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                    q, k, v, causal=causal, backend=backend
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
        )
        for backend in ["fa2", "fa3"]
    )
    q = paddle.randn(shape=[seq_len, num_qo_heads, head_dim], dtype="float16").to(
>>>>>>        dtype=paddle.float8_e4m3fn
    )
    k = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16").to(
>>>>>>        dtype=paddle.float8_e4m3fn
    )
    v = paddle.randn(shape=[seq_len, num_kv_heads, head_dim], dtype="float16").to(
>>>>>>        dtype=paddle.float8_e4m3fn
    )
    fp8_sm90_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                q, k, v, causal=causal, backend="fa3", o_dtype="float16"
            ),
            dry_run_time_ms=100,
            repeat_time_ms=1000,
        )
    )

    def flops(ms):
        if causal:
            return seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1000000000.0
        else:
            return seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1000000000.0

    print(
        f"bench_single_prefill (seq_len={seq_len}, num_heads={num_heads}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s, fa3-fp8: {flops(fp8_sm90_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    device_capability = paddle.device.cuda.get_device_capability()
    if device_capability[0] != 9:
        print(f"Current device capability: {device_capability}.")
        print("Current benchmark targets capability (9, 0). Returning...")
        exit()
    for seq_len in [4096, 8192, 16384]:
        for num_heads in [24, 32]:
            for causal in [True, False]:
                for head_dim in [64, 128, 256]:
                    bench_single_prefill(seq_len, num_heads, causal, head_dim)
