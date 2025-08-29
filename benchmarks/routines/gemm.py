import sys

sys.path.append("/home/flashinfer")
from collections import defaultdict

import einops
import numpy as np
import paddle
from paddle_utils import *

import flashinfer
from flashinfer.testing.utils import (bench_gpu_time,
                                      bench_gpu_time_with_cudagraph,
                                      dequantize_fp8, quantize_fp8)

from .flashinfer_benchmark_utils import (dtype_str_to_torch_dtype, get_device,
                                         print_perf_metrics)


def run_gemm_test(args):
    """
    Run a gemm test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "gemm_fp8_nt_groupwise":
        return testGemmFp8NtGroupwise(args)
    elif args.routine == "group_gemm_fp8_nt_groupwise":
        return testGroupGemmFp8NtGroupwise(args)
    elif args.routine == "bmm_fp8":
        return testBmmFp8(args)
    elif args.routine == "mm_fp4":
        return testMmFp4(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_gemm_args(line, parser):
    """
    Parse command line arguments for gemm test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size of test case.",
    )
    parser.add_argument(
        "--m", type=int, required=True, help="Number of rows in the first matrix."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of columns in the second matrix."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of columns in the first matrix and number of rows in the second matrix.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        required=False,
        default=128,
        help="Tile size for the gemm operation.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        required=False,
        default=1,
        help="Group size for the group gemm operation.",
    )
    parser.add_argument(
        "--scale_major_mode",
        type=str,
        required=False,
        default="MN",
        choices=["MN", "K"],
        help="Scale major mode.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        help="Data type of the input.",
    )
    parser.add_argument(
        "--mat2_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        help="Data type of the mat2.",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the output.",
    )
    parser.add_argument(
        "--mma_sm",
        type=int,
        required=False,
        default=1,
        choices=[1, 2],
        help="How many SMs to use for the MMA operation, must be 1 or 2",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cudnn"],
        choices=["cudnn", "cublas", "trtllm", "cutlass"],
        help="Kernel backends to test. Default: cudnn",
    )
    parser.add_argument(
        "--use_128x4_sf_layout",
        action="store_true",
        help="Use 128x4 SF layout for the input and mat2.",
    )
    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] args = {args!r}")
    return args


>>>>>>def to_float8(x, dtype=paddle.float8_e4m3fn):
    finfo = paddle.finfo(dtype=dtype)
    min_val, max_val = tuple(
        [
            paddle.amin(x, axis=None, keepdim=False),
            paddle.max(x, axis=None, keepdim=False),
        ]
    )
    amax = paddle.maximum(x=min_val.abs(), y=max_val.abs()).clip(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clip(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.astype(dtype="float32").reciprocal()


def testGemmFp8NtGroupwise(args):
    """
    Test gemm_fp8_nt_groupwise API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs gemm_fp8_nt_groupwise
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testGemmFp8NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    tile_size = args.tile_size
    scale_major_mode = args.scale_major_mode
    mma_sm = args.mma_sm
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in ["bfloat16", "float16"]:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")
    if "trtllm" in backends:
        remove_trtllm = True
        print("[INFO] trtllm backend testing not supported yet")
        if remove_trtllm:
            backends.remove("trtllm")
    a_val = paddle.randn(shape=(m, k), dtype="float32")
    b_val = paddle.randn(shape=(n, k), dtype="float32") / np.sqrt(k)
    if args.verbose >= 2:
        print(f"[VVERBOSE] a_val.shape = {tuple(a_val.shape)!r}")
        print(f"[VVERBOSE] b_val.shape = {tuple(b_val.shape)!r}")
    if scale_major_mode == "K":
        a_scale_shape = m, k // tile_size
        b_scale_shape = n // tile_size, k // tile_size
    else:
        a_scale_shape = k // tile_size, m
        b_scale_shape = k // tile_size, n // tile_size
    a_tile_shape = 1, tile_size
    b_tile_shape = tile_size, tile_size
    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)
    if "trtllm" in backends:
        a_scale_shape_trtllm = m, k // tile_size
        b_scale_shape_trtllm = k // tile_size, n // tile_size
        a_fp8_trtllm, a_scale_trtllm = quantize_fp8(
            a_val, a_scale_shape_trtllm, a_tile_shape, "K"
        )
        b_fp8_trtllm, b_scale_trtllm = quantize_fp8(
            b_val, b_scale_shape_trtllm, b_tile_shape, "MN"
        )
    if args.verbose >= 2:
        print(f"[VVERBOSE] a_fp8.shape = {tuple(a_fp8.shape)!r}")
        print(f"[VVERBOSE] b_fp8.shape = {tuple(b_fp8.shape)!r}")
        print(f"[VVERBOSE] a_scale.shape = {tuple(a_scale.shape)!r}")
        print(f"[VVERBOSE] b_scale.shape = {tuple(b_scale.shape)!r}")
    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)

    def run_backend(backend):
        if backend == "cutlass":
            return flashinfer.gemm.gemm_fp8_nt_groupwise(
                a=a_fp8,
                b=b_fp8,
                a_scale=a_scale,
                b_scale=b_scale,
                scale_major_mode=scale_major_mode,
                out_dtype=out_dtype,
                mma_sm=mma_sm,
                backend="cutlass",
            )
        elif backend == "trtllm":
            return flashinfer.gemm.gemm_fp8_nt_groupwise(
                a=a_fp8,
                b=b_fp8,
                a_scale=a_scale,
                b_scale=b_scale,
                scale_major_mode=None,
                out_dtype=out_dtype,
                mma_sm=mma_sm,
                backend="trtllm",
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_output = einops.einsum(a_dequant, b_dequant, "m k, n k -> m n").to(
            out_dtype
        )
        has_reference_output = True
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                try:
                    assert paddle.allclose(
                        x=reference_output, y=tested_outputs[i], rtol=0.01, atol=0.01
                    ).item(), ""
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            problem_bytes = (
                m * k + n * k
>>>>>>            ) * paddle.float8_e4m3fn.itemsize + m * n * out_dtype.element_size()
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["m"] = m
            cur_res["n"] = n
            cur_res["k"] = k
            cur_res["tile_size"] = tile_size
            cur_res["scale_major_mode"] = scale_major_mode
            cur_res["out_dtype"] = out_dtype
            cur_res["mma_sm"] = mma_sm
            cur_res["backend"] = backend
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testGroupGemmFp8NtGroupwise(args):
    """
    Test group_gemm_fp8_nt_groupwise API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs group_gemm_fp8_nt_groupwise
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testGroupGemmFp8NtGroupwise")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = ["cutlass"]
    m = args.m
    n = args.n
    k = args.k
    group_size = args.group_size
    tile_size = args.tile_size
    scale_major_mode = args.scale_major_mode
    mma_sm = args.mma_sm
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in ["bfloat16", "float16"]:
        raise ValueError(f"Unsupported output dtype: {args.out_dtype}")
    a_val = paddle.randn(shape=(group_size * m, k), dtype="float32")
    b_val = paddle.randn(shape=(group_size, n, k), dtype="float32") / np.sqrt(k)
    if args.verbose >= 2:
        print(f"[VVERBOSE] a_val.shape = {tuple(a_val.shape)!r}")
        print(f"[VVERBOSE] b_val.shape = {tuple(b_val.shape)!r}")
    if scale_major_mode == "K":
        a_scale_shape = group_size * m, k // tile_size
        b_scale_shape = group_size, n // tile_size, k // tile_size
    else:
        a_scale_shape = k // tile_size, m * group_size
        b_scale_shape = group_size, k // tile_size, n // tile_size
    a_tile_shape = 1, tile_size
    b_tile_shape = 1, tile_size, tile_size
    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)
    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    m_indptr = paddle.arange(start=0, end=group_size + 1, dtype="int32") * m
    if args.verbose >= 2:
        print(f"[VVERBOSE] a_fp8.shape = {tuple(a_fp8.shape)!r}")
        print(f"[VVERBOSE] b_fp8.shape = {tuple(b_fp8.shape)!r}")
        print(f"[VVERBOSE] a_scale.shape = {tuple(a_scale.shape)!r}")
        print(f"[VVERBOSE] b_scale.shape = {tuple(b_scale.shape)!r}")
        print(f"[VVERBOSE] m_indptr.shape = {tuple(m_indptr.shape)!r}")

    def run_backend(backend):
        if backend == "cutlass":
            return flashinfer.gemm.group_gemm_fp8_nt_groupwise(
                a=a_fp8,
                b=b_fp8,
                a_scale=a_scale,
                b_scale=b_scale,
                m_indptr=m_indptr,
                scale_major_mode=scale_major_mode,
                out_dtype=out_dtype,
                mma_sm=mma_sm,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_output = (
            einops.einsum(
                a_dequant.view((group_size, m, k)), b_dequant, "b m k, b n k -> b m n"
            )
            .view((group_size * m, n))
            .to(out_dtype)
        )
        has_reference_output = True
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
        else:
            backend_times[cur_backend] = bench_gpu_time(
                fn=lambda: run_backend(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                try:
                    assert paddle.allclose(
                        x=reference_output, y=tested_outputs[i], rtol=0.01, atol=0.01
                    ).item(), ""
                except AssertionError as e:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        print(e)
                        raise
    res = []
    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k * group_size
            problem_bytes = (
>>>>>>                (group_size * m * k + group_size * n * k) * paddle.float8_e4m3fn.itemsize
                + group_size * m * n * out_dtype.element_size()
            )
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
        if args.output_path is not None:
            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["m"] = m
            cur_res["n"] = n
            cur_res["k"] = k
            cur_res["group_size"] = group_size
            cur_res["tile_size"] = tile_size
            cur_res["scale_major_mode"] = scale_major_mode
            cur_res["out_dtype"] = out_dtype
            cur_res["mma_sm"] = mma_sm
            cur_res["backend"] = backend
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res


def testBmmFp8(args):
    """
    Test bmm_fp8 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs bmm_fp8
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testBmmFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = args.backends
    batch_size = args.batch_size
    m = args.m
    n = args.n
    k = args.k
    input_dtype = args.input_dtype
    mat2_dtype = args.mat2_dtype
    res_dtype = args.out_dtype
    backends = args.backends
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
>>>>>>    if input_dtype not in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        raise ValueError(
            f"Unsupported input dtype: {input_dtype}. Supported dtypes are fp8_e4m3 and fp8_e5m2."
        )
    mat2_dtype = dtype_str_to_torch_dtype(args.mat2_dtype)
>>>>>>    if mat2_dtype not in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
        raise ValueError(
            f"Unsupported mat2 dtype: {mat2_dtype}. Supported dtypes are fp8_e4m3 and fp8_e5m2."
        )
    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in ["bfloat16", "float16"]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )
    input = paddle.randn(shape=[batch_size, m, k], dtype="bfloat16")
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)
    mat2 = paddle.randn(shape=[batch_size, n, k], dtype="bfloat16").transpose(
        perm=dim2perm(
            paddle.randn(shape=[batch_size, n, k], dtype="bfloat16").ndim, -2, -1
        )
    )
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)
    if args.verbose >= 2:
        print(f"[VVERBOSE] input_fp8.shape = {tuple(input_fp8.shape)!r}")
        print(f"[VVERBOSE] input_fp8.dtype = {input_fp8.dtype!r}")
        print(f"[VVERBOSE] mat2_fp8.shape = {tuple(mat2_fp8.shape)!r}")
        print(f"[VVERBOSE] mat2_fp8.dtype = {mat2_fp8.dtype!r}")
        print(f"[VVERBOSE] input_inv_s = {input_inv_s!r}")
        print(f"[VVERBOSE] input_inv_s.dtype = {input_inv_s.dtype!r}")
        print(f"[VVERBOSE] mat2_inv_s = {mat2_inv_s!r}")
        print(f"[VVERBOSE] mat2_inv_s.dtype = {mat2_inv_s.dtype!r}")

    def run_backend(backend):
        if backend in ["cudnn", "cublas", "cutlass"]:
            return flashinfer.gemm.bmm_fp8(
                A=input_fp8,
                B=mat2_fp8,
                A_scale=input_inv_s,
                B_scale=mat2_inv_s,
                dtype=res_dtype,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_output = paddle.bmm(x=input, y=mat2)
        has_reference_output = True
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend(cur_backend),
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
                fn=lambda: run_backend(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
>>>>>>            if reference_output.dtype in [paddle.float8_e4m3fn, paddle.float8_e5m2]:
                print(
                    "[INFO] Reference output is FP8. Converting to float32 for reference check."
                )
                reference_output = reference_output.to("float32")
                tested_outputs = [output.to("float32") for output in tested_outputs]
            for i in range(len(tested_backends)):
                try:
                    cos_sim = paddle.nn.functional.cosine_similarity(
                        x1=reference_output.reshape(-1),
                        x2=tested_outputs[i].reshape(-1),
                        axis=0,
                    )
                    assert cos_sim > 0.99
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
            problem_flops = 2 * m * n * k * batch_size
            problem_bytes = (
                m * k * input_dtype.element_size()
                + n * k * mat2_dtype.element_size()
                + m * n * res_dtype.element_size()
            )
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["batch_size"] = batch_size
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["input_dtype"] = input_dtype
                cur_res["mat2_dtype"] = mat2_dtype
                cur_res["out_dtype"] = res_dtype
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res


def testMmFp4(args):
    """
    Test mm_fp4 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP4
    3. Runs mm_fp4
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMmFp4")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")
    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    res_dtype = args.out_dtype
    backends = args.backends
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck
    use_128x4_sf_layout = args.use_128x4_sf_layout
    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in ["bfloat16", "float16"]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )
    if "trtllm" in backends:
        remove_trtllm = False
        if res_dtype == "float16":
            print("[INFO] trtllm backend does not suppot float16 output")
            remove_trtllm = True
        if remove_trtllm:
            backends.remove("trtllm")
    if "cutlass" in backends:
        remove_cutlass = False
        if not use_128x4_sf_layout:
            print("[INFO] cutlass backend does not suppot use_128x4_sf_layout=False")
            remove_cutlass = True
        if remove_cutlass:
            backends.remove("cutlass")
    if "cudnn" in backends:
        remove_cudnn = False
        if not use_128x4_sf_layout:
            print("[INFO] cudnn backend does not suppot use_128x4_sf_layout=False")
            remove_cudnn = True
        if remove_cudnn:
            backends.remove("cudnn")
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return
    input = paddle.randn(shape=[m, k], dtype="bfloat16")
    mat2 = paddle.randn(shape=[n, k], dtype="bfloat16")
    a_sf_layout = (
        flashinfer.SfLayout.layout_128x4
        if use_128x4_sf_layout
        else flashinfer.SfLayout.layout_8x4
    )
    global_sf_input = 448 * 6 / input.astype(dtype="float32").abs().nan_to_num()._max()
    global_sf_mat2 = 448 * 6 / mat2.astype(dtype="float32").abs().nan_to_num()._max()
    input_fp4, input_inv_s = flashinfer.nvfp4_quantize(
        input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
    )
    mat2_fp4, mat2_inv_s = flashinfer.nvfp4_quantize(
        mat2,
        global_sf_mat2,
        sfLayout=flashinfer.SfLayout.layout_128x4,
        do_shuffle=False,
    )
    if "trtllm" in backends:
        mat2_fp4_trtllm, mat2_inv_s_trtllm = flashinfer.nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=True,
        )
    if args.verbose >= 2:
        print(f"[VVERBOSE] input_fp4.shape = {tuple(input_fp4.shape)!r}")
        print(f"[VVERBOSE] input_fp4.dtype = {input_fp4.dtype!r}")
        print(f"[VVERBOSE] mat2_fp4.shape = {tuple(mat2_fp4.shape)!r}")
        print(f"[VVERBOSE] mat2_fp4.dtype = {mat2_fp4.dtype!r}")
    alpha = 1.0 / (global_sf_input * global_sf_mat2)

    def run_backend(backend):
        if backend in ["cudnn", "trtllm", "cutlass"]:
            return flashinfer.gemm.mm_fp4(
                a=input_fp4,
                b=mat2_fp4.T if backend != "trtllm" else mat2_fp4_trtllm.T,
                a_descale=input_inv_s,
                b_descale=mat2_inv_s.T if backend != "trtllm" else mat2_inv_s_trtllm.T,
                alpha=alpha,
                out_dtype=res_dtype,
                block_size=16,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    has_reference_output = False
    if run_refcheck:
        reference_output = paddle.mm(input=input, mat2=mat2.T)
        has_reference_output = True
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(cur_backend).detach()
        if is_cuda_graph_compatible:
            backend_times[cur_backend] = bench_gpu_time_with_cudagraph(
                fn=lambda: run_backend(cur_backend),
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
                fn=lambda: run_backend(cur_backend),
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.num_iters,
                l2_flush=True,
                l2_flush_size_mb=256,
                l2_flush_device=device,
                sleep_after_run=True,
            )
    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                try:
                    cos_sim = paddle.nn.functional.cosine_similarity(
                        x1=reference_output.reshape(-1),
                        x2=tested_outputs[i].reshape(-1),
                        axis=0,
                    )
                    assert cos_sim > 0.97
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
            problem_flops = 2 * m * n * k
            problem_bytes = m * k * 0.5 + n * k * 0.5 + m * n * res_dtype.element_size()
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)
            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = res_dtype
                cur_res["use_128x4_sf_layout"] = use_128x4_sf_layout
                cur_res["backend"] = backend
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res
