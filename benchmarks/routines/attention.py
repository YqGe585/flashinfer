import sys


from collections import defaultdict

import numpy as np
import paddle
from flashinfer.paddle_utils import *

import flashinfer
from flashinfer.testing.utils import (
    attention_tb_per_sec_with_actual_seq_lens,
    attention_tflops_per_sec_with_actual_seq_lens, bench_gpu_time,
    bench_gpu_time_with_cudagraph)

from .flashinfer_benchmark_utils import (dtype_str_to_torch_dtype, get_device,
                                         print_perf_metrics)


def run_attention_test(args):
    """
    Run an attention test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "BatchDecodeWithPagedKVCacheWrapper":
        return testBatchDecodeWithPagedKVCacheWrapper(args)
    elif args.routine == "BatchPrefillWithPagedKVCacheWrapper":
        return testBatchPrefillWithPagedKVCacheWrapper(args)
    elif args.routine == "BatchPrefillWithRaggedKVCacheWrapper":
        return testBatchPrefillWithRaggedKVCacheWrapper(args)
    elif args.routine == "BatchMLAPagedAttentionWrapper":
        return testBatchMLAPagedAttentionWrapper(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_attention_args(line, parser):
    """
    Parse command line arguments for attention test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["fa2"],
        choices=[
            "fa2",
            "fa2_tc",
            "fa3",
            "cudnn",
            "cutlass",
            "trtllm-gen",
            "trtllm-gen-native",
        ],
        help="Kernel backends to test. Default: fa2",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        required=False,
        default=0,
        help="Page size for paged attention. Required for paged attention. Ignored for non-paged attention.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size of test case."
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        required=False,
        default=1,
        help="Max sequence length of the query. Should be 1 for decode.",
    )
    parser.add_argument(
        "--s_kv",
        type=int,
        required=True,
        help="Max sequence length of the key and value.",
    )
    parser.add_argument(
        "--num_qo_heads", type=int, required=True, help="Number of query heads."
    )
    parser.add_argument(
        "--num_kv_heads", type=int, required=True, help="Number of key and value heads."
    )
    parser.add_argument(
        "--head_dim_qk",
        type=int,
        required=False,
        help="Head dimension of the query and key for prefill and decode MHA/GQA/MQA.",
    )
    parser.add_argument(
        "--head_dim_vo",
        type=int,
        required=False,
        help="Head dimension of the value and output for prefill and decode MHA/GQA/MQ.",
    )
    parser.add_argument(
        "--head_dim_ckv",
        type=int,
        required=False,
        help="Head dimension of compressed kv-cache tensor (without rope).",
    )
    parser.add_argument(
        "--head_dim_kpe",
        type=int,
        required=False,
        help="Head dimension of the rope part of the kv-cache tensor.",
    )
    parser.add_argument(
        "--q_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the query. Currently only bfloat16 is supported.",
    )
    parser.add_argument(
        "--kv_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the key and value. Currently only bfloat16 is supported.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Causal masking. Note: not padding masking. Only used for prefill tests.",
    )
    parser.add_argument(
        "--random_actual_seq_len",
        action="store_true",
        default=False,
        help="Use random actual sequence lengths for the query and key and value. Random values are generated between 1 and maximum sequence length. If False, use maximum sequence length.",
    )
    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] args = {args!r}")
    return args


def sample_actual_seq_lens(max_seqlen, batch_size, device, random_actual_seq_len):
    """
    Get an array of actual sequence lengths for given batch size and max sequence length.
    If random_actual_seq_len is True, sample actual sequence lengths randomly.
    Otherwise, set all actual sequence lengths to max_seqlen.

    Args:
        max_seqlen: Maximum sequence length.
        batch_size: Batch size.
        device: Device to sample on.
        random_actual_seq_len: Whether to sample actual sequence lengths randomly.

    Returns:
        actual_seq_lens: Actual sequence lengths for each batch.
    """
    if random_actual_seq_len:
        actual_seq_lens = paddle.randint(
            low=1, high=max_seqlen + 1, shape=(batch_size, 1, 1, 1), dtype="int32"
        )
    else:
        actual_seq_lens = paddle.full(
            shape=(batch_size, 1, 1, 1), fill_value=max_seqlen, dtype="int32"
        )
    return actual_seq_lens


def testBatchDecodeWithPagedKVCacheWrapper(args):
    """
    Test BatchDecodeWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa2_tc, cudnn, trtllm-gen, trtllm-gen-native backends.

    This test:
    1. Creates paged KV cache and query tensors
    2. Runs decode attention with different backends
    3. Verifies outputs match between backends
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchDecodeWithPagedKVCacheWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    q_init_dtype = "bfloat16"
    kv_init_dtype = "bfloat16"
    rtol = 0.2
    atol = 0.01
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in ["bfloat16", paddle.float8_e4m3fn]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in ["bfloat16", paddle.float8_e4m3fn]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")
    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    if "fa2" in backends:
        remove_fa2 = False
        head_grp_size = num_qo_heads // num_kv_heads
        if head_grp_size == 5:
            print(
                "[INFO] FA2 backend is not supported for this configuration. Skipping."
            )
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    if "fa2_tc" in backends:
        remove_fa2_tc = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
            paddle.float8_e5m2,
        ]:
            print("[INFO] FA2_TC backend does not support FP8. Skipping.")
            remove_fa2_tc = True
        if remove_fa2_tc:
            backends.remove("fa2_tc")
    if "cudnn" in backends:
        remove_cudnn = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
            paddle.float8_e5m2,
        ]:
            print("[INFO] cuDNN backend does not support FP8. Skipping.")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    actual_seq_lens_kv = sample_actual_seq_lens(
        s_kv, batch_size, device, args.random_actual_seq_len
    )
    sum_seq_kv = paddle.sum(x=actual_seq_lens_kv).item()
    avg_seq_len_kv = sum_seq_kv // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(
            f"[VVERBOSE] actual_seq_lens_kv.flatten() = {actual_seq_lens_kv.flatten()!r}"
        )
    q = paddle.rand(shape=[batch_size, num_qo_heads, head_dim_qk], dtype=q_init_dtype)
    if args.verbose >= 2:
        print(f"[VVERBOSE] q.shape = {tuple(q.shape)!r}")
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if args.verbose >= 2:
        print(f"[VVERBOSE] num_pages_per_seq = {num_pages_per_seq!r}")
        print(f"[VVERBOSE] total_num_pages = {total_num_pages!r}")
    kv_cache_shape = total_num_pages, 2, num_kv_heads, page_size, head_dim_qk
    kv_cache = paddle.randn(shape=kv_cache_shape, dtype=kv_init_dtype).to(device)
    if "trtllm-gen" in backends:
        kv_cache_for_trt = kv_cache.detach().clone()
    kv_cache = kv_cache.as_strided(
        shape=tuple(kv_cache.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim_qk,
            page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache_view, v_cache_view = kv_cache[:, 0, :, :, :], kv_cache[:, 1, :, :, :]
    if "trtllm-gen" in backends:
        paddle.assign(kv_cache, output=kv_cache_for_trt)
    v_cache = v_cache_view.as_strided(
        shape=tuple(v_cache_view.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache = k_cache_view.as_strided(
        shape=tuple(k_cache_view.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    block_tables = paddle.to_tensor(
        data=[
            [(k + i * num_pages_per_seq) for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype="int32",
        place=device,
    )
    kv_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(
                    x=(actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                    axis=0,
                ),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indices = paddle.zeros(shape=kv_indptr[-1], dtype="int32")
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = paddle.arange(
            start=i * num_pages_per_seq,
            end=i * num_pages_per_seq + (end_idx - start_idx),
        )
    kv_last_page_len = (
        paddle.where(
            condition=actual_seq_lens_kv.flatten() % page_size == 0,
            x=paddle.full(shape=(batch_size,), fill_value=page_size),
            y=actual_seq_lens_kv.flatten() % page_size,
        )
        .astype(dtype="int32")
        .to(device)
    )
    ragged_q = (
        paddle.arange(start=0, end=batch_size + 1) * (num_qo_heads * head_dim_qk)
    ).astype(dtype="int64")
    scale = float(1.0 / head_dim_qk**0.5)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    if args.verbose >= 2:
        print(f"[VVERBOSE] kv_cache.shape = {tuple(kv_cache.shape)!r}")
        print(f"[VVERBOSE] kv_cache.stride() = {kv_cache.get_strides()!r}")
        print(f"[VVERBOSE] block_tables.shape = {tuple(block_tables.shape)!r}")
        print(f"[VVERBOSE] kv_indptr.shape = {tuple(kv_indptr.shape)!r}")
        print(f"[VVERBOSE] kv_indices.shape = {tuple(kv_indices.shape)!r}")
        print(f"[VVERBOSE] kv_last_page_len.shape = {tuple(kv_last_page_len.shape)!r}")
        print(f"[VVERBOSE] scale = {scale!r}")
    backend_wrappers = {}
    for backend in backends:
        if backend in ["fa2", "fa2_tc", "trtllm-gen"]:
            plan_kv_indptr = (
                kv_indptr.clone().detach() if backend == "trtllm-gen" else kv_indptr
            )
            backend_wrappers[backend] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "HND",
                use_cuda_graph=is_cuda_graph_compatible,
                use_tensor_cores=backend != "fa2",
                paged_kv_indptr_buffer=plan_kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last_page_len,
                backend=backend,
            )
            backend_wrappers[backend].plan(
                plan_kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                q_data_type=q_dtype,
                data_type=kv_dtype,
            )
    k_scale, v_scale = None, None
    if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        q = q.to(q_dtype)
    if kv_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        k_data, v_data = paddle.chunk(x=kv_cache, chunks=2, axis=1)
        k_scale = k_data.amax().item() / 256
        v_scale = v_data.amax().item() / 256
        k_fp8 = (k_data / k_scale).to(kv_dtype)
        v_fp8 = (v_data / v_scale).to(kv_dtype)
        kv_cache = paddle.concat(x=[k_fp8, v_fp8], axis=1)
        if "trtllm-gen" in backends:
            k_data, v_data = paddle.chunk(x=kv_cache_for_trt, chunks=2, axis=1)
            k_fp8 = (k_data / k_scale).to(kv_dtype)
            v_fp8 = (v_data / v_scale).to(kv_dtype)
            kv_cache_for_trt = paddle.concat(x=[k_fp8, v_fp8], axis=1)

    def run_backend_wrapper(backend):
        if backend in ["fa2", "fa2_tc", "trtllm-gen"]:
            return backend_wrappers[backend].run(
                q, kv_cache, k_scale=k_scale, v_scale=v_scale
            )
        elif backend == "cudnn":
            return flashinfer.decode.cudnn_batch_decode_with_kv_cache(
                q,
                k_cache,
                v_cache,
                scale,
                workspace_buffer,
                max_sequence_kv=s_kv,
                actual_seq_lens_kv=actual_seq_lens_kv,
                block_tables=block_tables,
                is_cuda_graph_compatible=is_cuda_graph_compatible,
                batch_offsets_q=ragged_q,
                batch_offsets_o=ragged_q,
            )
        elif backend == "trtllm-gen-native":
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv,
                max_seq_len=s_kv,
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=1.0 if v_scale is None else v_scale,
            )
        else:
            raise ValueError(f"Backend {backend} not supported")

    has_reference_output = False
    if run_refcheck and "fa2" in backends:
        reference_output = (
            backend_wrappers["fa2"]
            .run(q, kv_cache, k_scale=k_scale, v_scale=v_scale)
            .detach()
        )
        has_reference_output = True
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
                if args.verbose >= 2:
                    print(
                        "[VVERBOSE] Reference output is FP8. Converting to float32 for reference check."
                    )
                reference_output = reference_output.to("float32")
                tested_outputs = [output.to("float32") for output in tested_outputs]
            for i in range(len(tested_outputs)):
                try:
                    assert paddle.allclose(
                        x=reference_output, y=tested_outputs[i], rtol=rtol, atol=atol
                    ).item(), ""
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            actual_seq_lens_q_flat = paddle.ones_like(x=actual_seq_lens_kv_flat)
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                False,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                o_dtype=q_dtype,
            )
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["page_size"] = page_size
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = False
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBatchPrefillWithPagedKVCacheWrapper(args):
    """
    Test BatchPrefillWithPagedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, trtllm-gen, trtllm-gen-native, and cudnn backends.

    This test:
    1. Creates paged KV cache and query tensors for prefill
    2. Runs prefill attention with different backends
    3. Verifies outputs match between backends (if refcheck enabled)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: Dictionary containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchPrefillWithPagedKVCacheWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    q_init_dtype = "bfloat16"
    kv_init_dtype = "bfloat16"
    rtol = 0.2
    atol = 0.01
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in ["bfloat16", paddle.float8_e4m3fn]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in ["bfloat16", paddle.float8_e4m3fn]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")
    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo
    causal = args.causal
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    if "fa2" in backends:
        remove_fa2 = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
            print("[INFO] FA2 backend does not support FP8. Skipping.")
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    if "fa3" in backends:
        remove_fa3 = False
        device_capability = paddle.device.cuda.get_device_capability()
        if device_capability[0] != 9:
            print(
                f"[INFO] FA3 backend does not support capability {device_capability}. Skipping."
            )
            remove_fa3 = True
        if remove_fa3:
            backends.remove("fa3")
    if "cudnn" in backends:
        remove_cudnn = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
>>>>>>            paddle.float8_e5m2,
        ]:
            print("[INFO] cuDNN backend does not support FP8. Skipping.")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")
    if "trtllm-gen" in backends:
        remove_trtllm = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
>>>>>>            paddle.float8_e5m2,
        ]:
            print("[INFO] trtllm-gen backend does not support FP8. Skipping.")
            remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm-gen")
    if "cutlass" in backends:
        print("[INFO] CUTLASS backend does not support prefill. Skipping.")
        remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return
    layer_not_supported = False
    if not (head_dim_qk == 128 and head_dim_qk == head_dim_vo or head_dim_qk == 192):
        print("[ERROR] Head dimension must be 128 or 192")
        layer_not_supported = True
    if layer_not_supported:
        print("[ERROR] Layer not supported. Exiting.")
        return
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    actual_seq_lens_q = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    )
    actual_seq_lens_kv = actual_seq_lens_q.clone()
    avg_seq_len_q = actual_seq_lens_q.sum().item() // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_q}")
    if args.verbose >= 2:
        print(
            f"[VVERBOSE] actual_seq_lens_q.flatten() = {actual_seq_lens_q.flatten()!r}"
        )
    cumsum_s_qo = paddle.sum(x=actual_seq_lens_q)
    q = paddle.randn(shape=[cumsum_s_qo, num_qo_heads, head_dim_qk], dtype=q_init_dtype)
    if args.verbose >= 2:
        print(f"[VVERBOSE] q.shape = {tuple(q.shape)!r}")
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if args.verbose >= 2:
        print(f"[VVERBOSE] num_pages_per_seq = {num_pages_per_seq!r}")
        print(f"[VVERBOSE] total_num_pages = {total_num_pages!r}")
    kv_cache_shape = total_num_pages, 2, num_kv_heads, page_size, head_dim_qk
    kv_cache = paddle.randn(shape=kv_cache_shape, dtype=kv_init_dtype).to(device)
    kv_cache = kv_cache.as_strided(
        shape=tuple(kv_cache.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim_qk,
            page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache_view, v_cache_view = kv_cache[:, 0, :, :, :], kv_cache[:, 1, :, :, :]
    v_cache = v_cache_view.as_strided(
        shape=tuple(v_cache_view.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    k_cache = k_cache_view.as_strided(
        shape=tuple(k_cache_view.shape),
        stride=(
            2 * page_size * num_kv_heads * head_dim_qk,
            head_dim_qk,
            num_kv_heads * head_dim_qk,
            1,
        ),
    )
    block_tables = paddle.to_tensor(
        data=[
            [(k + i * num_pages_per_seq) for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype="int32",
        place=device,
    )
    actual_seq_lens_q_device = actual_seq_lens_q.to(device)
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    q_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(x=actual_seq_lens_q_device.view(-1), axis=0)
                * head_dim_qk
                * num_qo_heads,
            ]
        )
        .astype(dtype="int64")
        .to(device)
    )
    qo_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(x=actual_seq_lens_q_device.view(-1), axis=0),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(
                    x=(actual_seq_lens_kv_device.flatten() + page_size - 1)
                    // page_size,
                    axis=0,
                ),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indices = paddle.zeros(shape=kv_indptr[-1], dtype="int32")
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = paddle.arange(
            start=i * num_pages_per_seq,
            end=i * num_pages_per_seq + (end_idx - start_idx),
        )
    kv_last_page_len = (
        paddle.where(
            condition=actual_seq_lens_kv_device.flatten() % page_size == 0,
            x=paddle.full(shape=(batch_size,), fill_value=page_size),
            y=actual_seq_lens_kv_device.flatten() % page_size,
        )
        .astype(dtype="int32")
        .to(device)
    )
    scale = float(1.0 / head_dim_qk**0.5)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    if args.verbose >= 2:
        print(f"[VVERBOSE] kv_cache.shape = {tuple(kv_cache.shape)!r}")
        print(f"[VVERBOSE] kv_cache.stride() = {kv_cache.get_strides()!r}")
        print(f"[VVERBOSE] block_tables.shape = {tuple(block_tables.shape)!r}")
        print(f"[VVERBOSE] qo_indptr.shape = {tuple(qo_indptr.shape)!r}")
        print(f"[VVERBOSE] qo_indptr.dtype = {qo_indptr.dtype!r}")
        print(f"[VVERBOSE] kv_indptr.shape = {tuple(kv_indptr.shape)!r}")
        print(f"[VVERBOSE] kv_indices.shape = {tuple(kv_indices.shape)!r}")
        print(f"[VVERBOSE] kv_last_page_len.shape = {tuple(kv_last_page_len.shape)!r}")
        print(f"[VVERBOSE] scale = {scale!r}")
    backend_wrappers = {}
    for backend in backends:
        if backend in ["fa2", "fa3", "trtllm-gen"]:
            backend_wrappers[
                backend
            ] = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                "HND",
                use_cuda_graph=is_cuda_graph_compatible,
                qo_indptr_buf=qo_indptr,
                paged_kv_indptr_buf=kv_indptr,
                paged_kv_indices_buf=kv_indices,
                paged_kv_last_page_len_buf=kv_last_page_len,
                backend=backend,
            )
            backend_wrappers[backend].plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                page_size,
                pos_encoding_mode="NONE",
                causal=causal,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
            )
    k_scale, v_scale = None, None
    if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        q = q.to(q_dtype)
    if kv_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        k_data, v_data = paddle.chunk(x=kv_cache, chunks=2, axis=1)
        k_scale = k_data.amax().item() / 256
        v_scale = v_data.amax().item() / 256
        k_fp8 = (k_data / k_scale).to(kv_dtype)
        v_fp8 = (v_data / v_scale).to(kv_dtype)
        kv_cache = paddle.concat(x=[k_fp8, v_fp8], axis=1)

    def run_backend_wrapper(backend):
        if backend in ["fa2", "fa3", "trtllm-gen"]:
            return backend_wrappers[backend].run(
                q, kv_cache, k_scale=k_scale, v_scale=v_scale
            )
        elif backend == "cudnn":
            return flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache,
                v_cache,
                scale,
                workspace_buffer,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
                actual_seq_lens_q=actual_seq_lens_q_device,
                actual_seq_lens_kv=actual_seq_lens_kv_device,
                block_tables=block_tables,
                causal=causal,
                return_lse=True,
                is_cuda_graph_compatible=is_cuda_graph_compatible,
                batch_offsets_q=q_indptr,
                batch_offsets_o=q_indptr,
            )[0]
        elif backend == "trtllm-gen-native":
            return flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=workspace_buffer,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv_device,
                max_q_len=s_qo,
                max_kv_len=s_kv,
                bmm1_scale=scale if k_scale is None else k_scale * scale,
                bmm2_scale=1.0 if v_scale is None else v_scale,
                batch_size=batch_size,
                cum_seq_lens_q=qo_indptr,
                cum_seq_lens_kv=kv_indptr,
            )
        else:
            raise ValueError(f"Backend {backend} not supported")

    has_reference_output = False
    if run_refcheck and "fa2" in backends:
        reference_output = backend_wrappers["fa2"].run(
            q, kv_cache, k_scale=k_scale, v_scale=v_scale
        )
        has_reference_output = True
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend)
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
                if args.verbose >= 2:
                    print(
                        "[VVERBOSE] Reference output is FP8. Converting to float32 for reference check."
                    )
                reference_output = reference_output.to("float32")
                tested_outputs = [output.to("float32") for output in tested_outputs]
            for i in range(len(tested_backends)):
                try:
                    assert paddle.allclose(
                        x=reference_output, y=tested_outputs[i], rtol=rtol, atol=atol
                    ).item(), ""
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_q_flat = actual_seq_lens_q.flatten().to("cpu")
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                o_dtype=q_dtype,
            )
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["page_size"] = page_size
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = causal
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBatchPrefillWithRaggedKVCacheWrapper(args):
    """
    Test BatchPrefillWithRaggedKVCacheWrapper API and equivalent cuDNN API.
    Supports fa2, fa3, cutlass, and cudnn backends.

    This test:
    1. Creates ragged KV cache and query tensors for prefill
    2. Runs prefill attention with different backends
    3. Verifies outputs match between backends (if refcheck enabled)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: Dictionary containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchPrefillWithRaggedKVCacheWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    q_init_dtype = "bfloat16"
    kv_init_dtype = "bfloat16"
    rtol = 0.2
    atol = 0.01
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in ["bfloat16", paddle.float8_e4m3fn, paddle.float8_e5m2]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in ["bfloat16", paddle.float8_e4m3fn, paddle.float8_e5m2]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")
    backends = args.backends
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim_qk = args.head_dim_qk
    head_dim_vo = args.head_dim_vo
    causal = args.causal
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    if "cudnn" in backends:
        remove_cudnn = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
>>>>>>            paddle.float8_e5m2,
        ]:
            print("[INFO] CUDNN backend does not support FP8. Skipping.")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")
    if "cutlass" in backends:
        remove_cutlass = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
>>>>>>            paddle.float8_e5m2,
        ]:
            print("[INFO] CUTLASS backend does not support FP8. Skipping.")
            remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")
    if "trtllm-gen" in backends:
        print("[INFO] trtllm-gen backend does not support ragged prefill. Skipping.")
        remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm-gen")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return
    layer_not_supported = False
    if not (head_dim_qk == 128 and head_dim_qk == head_dim_vo or head_dim_qk == 192):
        print("[ERROR] Head dimension must be 128 or 192")
        layer_not_supported = True
    if layer_not_supported:
        print("[ERROR] Layer not supported. Exiting.")
        return
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    actual_seq_lens_q = sample_actual_seq_lens(
        s_qo, batch_size, None, args.random_actual_seq_len
    )
    actual_seq_lens_kv = actual_seq_lens_q.clone()
    avg_seq_len_q = actual_seq_lens_q.sum().item() // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_q}")
    if args.verbose >= 2:
        print(
            f"[VVERBOSE] actual_seq_lens_q.flatten() = {actual_seq_lens_q.flatten()!r}"
        )
    cumsum_s_qo = paddle.sum(x=actual_seq_lens_q)
    cumsum_s_kv = paddle.sum(x=actual_seq_lens_kv)
    q = paddle.randn(shape=[cumsum_s_qo, num_qo_heads, head_dim_qk], dtype=q_init_dtype)
    if args.verbose >= 2:
        print(f"[VVERBOSE] q.shape = {tuple(q.shape)!r}")
    k = paddle.randn(
        shape=[cumsum_s_kv, num_kv_heads, head_dim_qk], dtype=kv_init_dtype
    )
    v = paddle.randn(
        shape=[cumsum_s_kv, num_kv_heads, head_dim_vo], dtype=kv_init_dtype
    )
    block_tables = None
    actual_seq_lens_q_device = actual_seq_lens_q.to(device)
    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)
    q_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(x=actual_seq_lens_q_device.view(-1), axis=0)
                * head_dim_qk
                * num_qo_heads,
            ]
        )
        .astype(dtype="int64")
        .to(device)
    )
    k_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_kv_device.view(-1), axis=0)
            * head_dim_qk
            * num_kv_heads,
        ]
    ).astype(dtype="int64")
    v_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_kv_device.view(-1), axis=0)
            * head_dim_vo
            * num_kv_heads,
        ]
    ).astype(dtype="int64")
    o_indptr = paddle.concat(
        x=[
            paddle.to_tensor(data=[0], place=device),
            paddle.cumsum(x=actual_seq_lens_q_device.view(-1), axis=0)
            * head_dim_vo
            * num_qo_heads,
        ]
    ).astype(dtype="int64")
    batch_offsets_stats = paddle.concat(
        x=[
            paddle.zeros(shape=[1], dtype=actual_seq_lens_q_device.dtype),
            paddle.cumsum(x=actual_seq_lens_q_device.flatten(), axis=0) * num_qo_heads,
        ]
    ).cuda()
    qo_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(x=actual_seq_lens_q_device.view(-1), axis=0),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(x=actual_seq_lens_kv_device.view(-1), axis=0),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    scale = float(1.0 / head_dim_qk**0.5)
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    if args.verbose >= 2:
        print(f"[VVERBOSE] k.shape = {tuple(k.shape)!r}")
        print(f"[VVERBOSE] v.shape = {tuple(v.shape)!r}")
        print(f"[VVERBOSE] qo_indptr.shape = {tuple(qo_indptr.shape)!r}")
        print(f"[VVERBOSE] kv_indptr.shape = {tuple(kv_indptr.shape)!r}")
        print(f"[VVERBOSE] scale = {scale!r}")
    backend_wrappers = {}
    for backend in backends:
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen"]:
            backend_wrappers[
                backend
            ] = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=is_cuda_graph_compatible,
                qo_indptr_buf=qo_indptr,
                kv_indptr_buf=kv_indptr,
                backend=backend,
            )
            backend_wrappers[backend].plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo=head_dim_vo,
                causal=causal,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
            )
    k_scale, v_scale = None, None
    if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        q = q.to(q_dtype)
    if kv_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        k_scale = k.amax().item() / 256
        v_scale = v.amax().item() / 256
        k = (k / k_scale).to(kv_dtype)
        v = (v / v_scale).to(kv_dtype)

    def run_backend_wrapper(backend):
        if backend in ["cutlass", "fa2", "fa3", "trtllm-gen"]:
            return backend_wrappers[backend].run_return_lse(q, k, v)[0]
        elif backend == "cudnn":
            return flashinfer.prefill.cudnn_batch_prefill_with_kv_cache(
                q,
                k,
                v,
                scale,
                workspace_buffer,
                max_token_per_sequence=s_qo,
                max_sequence_kv=s_kv,
                actual_seq_lens_q=actual_seq_lens_q_device,
                actual_seq_lens_kv=actual_seq_lens_kv_device,
                block_tables=block_tables,
                causal=causal,
                return_lse=True,
                batch_offsets_q=q_indptr,
                batch_offsets_k=k_indptr,
                batch_offsets_v=v_indptr,
                batch_offsets_o=o_indptr,
                batch_offsets_stats=batch_offsets_stats,
                is_cuda_graph_compatible=True,
            )[0]
        else:
            raise ValueError(f"Backend {backend} not supported")

    has_reference_output = False
    if run_refcheck and "fa2" in backends:
        reference_output = backend_wrappers["fa2"].run_return_lse(q, k, v)[0]
        has_reference_output = True
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend)
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
                if args.verbose >= 2:
                    print(
                        "[VVERBOSE] Reference output is FP8. Converting to float32 for reference check."
                    )
                reference_output = reference_output.to("float32")
                tested_outputs = [output.to("float32") for output in tested_outputs]
            for i in range(len(tested_backends)):
                try:
                    assert paddle.allclose(
                        x=reference_output, y=tested_outputs[i], rtol=rtol, atol=atol
                    ).item(), ""
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_q_flat = actual_seq_lens_q.flatten().to("cpu")
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                actual_seq_lens_q_flat,
                actual_seq_lens_kv_flat,
                head_dim_qk,
                head_dim_vo,
                num_qo_heads,
                num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                o_dtype=q_dtype,
            )
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["page_size"] = 0
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["num_kv_heads"] = num_kv_heads
                cur_res["head_dim_qk"] = head_dim_qk
                cur_res["head_dim_vo"] = head_dim_vo
                cur_res["causal"] = causal
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_q
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testBatchMLAPagedAttentionWrapper(args):
    """
    Test BatchMLAPagedAttentionWrapper and equivalent APIs.
    Supports fa2. and trtllm-gen-native.

    This test:
    1. Creates paged query and key-value cache tensors
    2. Runs MLA with different backends
    3. Verifies outputs match between backends
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBatchMLAPagedAttentionWrapper")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    q_init_dtype = "bfloat16"
    kv_init_dtype = "bfloat16"
    rtol = 0.2
    atol = 0.01
    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    if q_dtype not in ["bfloat16", paddle.float8_e4m3fn]:
        raise ValueError(f"Unsupported q_dtype: {args.q_dtype}")
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if kv_dtype not in ["bfloat16", paddle.float8_e4m3fn]:
        raise ValueError(f"Unsupported kv_dtype: {args.kv_dtype}")
    backends = args.backends
    page_size = args.page_size
    batch_size = args.batch_size
    s_qo = args.s_qo
    s_kv = args.s_kv
    num_qo_heads = args.num_qo_heads
    assert args.head_dim_ckv is not None, "head_dim_ckv must be provided for MLA"
    assert args.head_dim_kpe is not None, "head_dim_kpe must be provided for MLA"
    head_dim_ckv = args.head_dim_ckv
    head_dim_kpe = args.head_dim_kpe
    is_cuda_graph_compatible = not args.no_cuda_graph
    causal = False
    run_refcheck = args.refcheck
    if "fa2" in backends:
        remove_fa2 = False
        if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2] or kv_dtype in [
            paddle.float8_e4m3fn,
>>>>>>            paddle.float8_e5m2,
        ]:
            print("[INFO] FA2 backend does not support FP8. Skipping.")
            remove_fa2 = True
        if remove_fa2:
            backends.remove("fa2")
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    actual_seq_lens_kv = sample_actual_seq_lens(
        s_kv, batch_size, device, args.random_actual_seq_len
    )
    sum_seq_kv = paddle.sum(x=actual_seq_lens_kv).item()
    avg_seq_len_kv = sum_seq_kv // batch_size
    if args.verbose >= 1:
        print(f"[VERBOSE] Average actual seq len: {avg_seq_len_kv}")
    if args.verbose >= 2:
        print(
            f"[VVERBOSE] actual_seq_lens_kv.flatten() = {actual_seq_lens_kv.flatten()!r}"
        )
    q_nope = paddle.rand(
        shape=[batch_size, num_qo_heads, head_dim_ckv], dtype=q_init_dtype
    )
    q_pe = paddle.zeros(
        shape=[batch_size, num_qo_heads, head_dim_kpe], dtype=q_init_dtype
    )
    q = paddle.concat(x=[q_nope, q_pe], axis=2)
    if args.verbose >= 2:
        print(f"[VVERBOSE] q_nope.shape = {tuple(q_nope.shape)!r}")
        print(f"[VVERBOSE] q_pe.shape = {tuple(q_pe.shape)!r}")
        print(f"[VVERBOSE] q.shape = {tuple(q.shape)!r}")
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    block_tables = paddle.to_tensor(
        data=[
            [(k + i * num_pages_per_seq) for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype="int32",
        place=device,
    )
    if args.verbose >= 2:
        print(f"[VVERBOSE] num_pages_per_seq = {num_pages_per_seq!r}")
        print(f"[VVERBOSE] total_num_pages = {total_num_pages!r}")
        print(f"[VVERBOSE] block_tables.shape = {tuple(block_tables.shape)!r}")
    ckv_cache_shape = total_num_pages, page_size, head_dim_ckv
    ckv_cache = paddle.randn(shape=ckv_cache_shape, dtype=kv_init_dtype)
    kpe_cache_shape = total_num_pages, page_size, head_dim_kpe
    kpe_cache = paddle.randn(shape=kpe_cache_shape, dtype=q_init_dtype)
    kv_cache = paddle.concat(x=[ckv_cache, kpe_cache], axis=2)
    qo_indptr = paddle.arange(start=0, end=batch_size + 1).astype(dtype="int32")
    kv_indptr = (
        paddle.concat(
            x=[
                paddle.to_tensor(data=[0], place=device),
                paddle.cumsum(
                    x=(actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                    axis=0,
                ),
            ]
        )
        .astype(dtype="int32")
        .to(device)
    )
    kv_indices = paddle.zeros(shape=kv_indptr[-1], dtype="int32")
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = paddle.arange(
            start=i * num_pages_per_seq,
            end=i * num_pages_per_seq + (end_idx - start_idx),
        )
    sm_scale = 1.0 / (head_dim_ckv + head_dim_kpe) ** 0.5
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8")
    if args.verbose >= 2:
        print(f"[VVERBOSE] ckv_cache.shape = {tuple(ckv_cache.shape)!r}")
        print(f"[VVERBOSE] kpe_cache.shape = {tuple(kpe_cache.shape)!r}")
        print(f"[VVERBOSE] kv_cache.shape = {tuple(kv_cache.shape)!r}")
        print(f"[VVERBOSE] qo_indptr.shape = {tuple(qo_indptr.shape)!r}")
        print(f"[VVERBOSE] kv_indptr.shape = {tuple(kv_indptr.shape)!r}")
        print(f"[VVERBOSE] kv_indices.shape = {tuple(kv_indices.shape)!r}")
        print(
            f"[VVERBOSE] actual_seq_lens_kv.shape = {tuple(actual_seq_lens_kv.shape)!r}"
        )
        print(f"[VVERBOSE] sm_scale = {sm_scale!r}")
        print(f"[VVERBOSE] workspace_buffer.shape = {tuple(workspace_buffer.shape)!r}")
    if "fa2" in backends:
        fi_fa2_mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            float_workspace_buffer=workspace_buffer,
            use_cuda_graph=is_cuda_graph_compatible,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=actual_seq_lens_kv,
            backend="fa2",
        )
        fi_fa2_mla_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=actual_seq_lens_kv,
            num_heads=num_qo_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=page_size,
            causal=causal,
            sm_scale=sm_scale,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
    if q_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        q = q.to(q_dtype)
        q_pe = q_pe.to(q_dtype)
        q_nope = q_nope.to(q_dtype)
    if kv_dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        ckv_cache = ckv_cache.to(kv_dtype)
        kpe_cache = kpe_cache.to(kv_dtype)
        kv_cache = kv_cache.to(kv_dtype)

    def run_backend_wrapper(backend):
        if backend == "fa2":
            return fi_fa2_mla_wrapper.run(
                q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False
            )
        if backend == "trtllm-gen-native":
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(axis=1),
                kv_cache=kv_cache.unsqueeze(axis=1),
                workspace_buffer=workspace_buffer,
                qk_nope_head_dim=128,
                kv_lora_rank=head_dim_ckv,
                qk_rope_head_dim=head_dim_kpe,
                block_tables=block_tables,
                seq_lens=actual_seq_lens_kv,
                max_seq_len=s_kv,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
            ).squeeze(1)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    if run_refcheck and "fa2" in backends:
        reference_output = fi_fa2_mla_wrapper.run(
            q_nope, q_pe, ckv_cache, kpe_cache, return_lse=False
        )
        has_reference_output = True
    else:
        has_reference_output = False
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend_wrapper(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                num_iters_within_graph=20,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend_wrapper(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=False,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 1:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
                reference_output = reference_output.to("float32")
                tested_outputs = [output.to("float32") for output in tested_outputs]
            for i in range(len(tested_outputs)):
                try:
                    assert paddle.allclose(
                        x=reference_output, y=tested_outputs[i], rtol=rtol, atol=atol
                    ).item(), ""
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            actual_seq_lens_kv_flat = actual_seq_lens_kv.flatten().to("cpu")
            actual_seq_lens_q_flat = paddle.ones_like(
                x=actual_seq_lens_kv.flatten().to("cpu")
            )
            o_mem_bytes = (
                actual_seq_lens_q_flat.size
                * num_qo_heads
                * head_dim_ckv
                * q_dtype.element_size()
            )
            qkv_mem_bytes = sum(
                [
                    (_.size * _.element_size())
                    for _ in [q_nope, q_pe, ckv_cache, kpe_cache]
                ]
            )
            total_mem_bytes = o_mem_bytes + qkv_mem_bytes
            tb_per_sec = (total_mem_bytes / (median_time * 1000000000.0)).item()
            tflops_total = (
                2
                * paddle.dot(
                    x=actual_seq_lens_q_flat.to("float32"),
                    y=actual_seq_lens_kv_flat.to("float32"),
                )
                * num_qo_heads
                * (2 * head_dim_ckv + head_dim_kpe)
            )
            tflops = (tflops_total / (median_time * 1000000000.0)).item()
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["backend"] = backend
                cur_res["page_size"] = page_size
                cur_res["batch_size"] = batch_size
                cur_res["s_qo"] = s_qo
                cur_res["s_kv"] = s_kv
                cur_res["num_qo_heads"] = num_qo_heads
                cur_res["head_dim_ckv"] = head_dim_ckv
                cur_res["head_dim_kpe"] = head_dim_kpe
                cur_res["causal"] = False
                cur_res["q_dtype"] = q_dtype
                cur_res["kv_dtype"] = kv_dtype
                cur_res["avg_actual_seq_len"] = avg_seq_len_kv
                cur_res["random_actual_seq_len"] = args.random_actual_seq_len
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
