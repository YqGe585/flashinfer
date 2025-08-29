import paddle
import pytest

import flashinfer
import flashinfer.triton


def torch_gemm(a, b, c, alpha, beta):
    x = paddle.matmul(x=a, y=b.T)
    c = alpha * x + beta * c
    return c


def torch_addmm(a, b, c, alpha=1.0, beta=0.0):
    C = paddle.addmm(input=c, x=a, y=b.T, beta=beta, alpha=alpha)
    return C


@pytest.mark.parametrize("M", [128, 512, 1024, 8192])
@pytest.mark.parametrize("N", [128, 512, 1024, 8192])
@pytest.mark.parametrize("K", [128, 512, 1024, 8192])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("num_sms", [1, 16, 64, 128, 132, 133])
@pytest.mark.parametrize(
    "dtype", [paddle.float8_e4m3fn, "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [True, False])
def test_sm_constraint_gemm(M, N, K, alpha, beta, num_sms, dtype, EPILOGUE_SUBTILE):
    out_dtype = dtype if dtype != paddle.float8_e4m3fn else "bfloat16"
    a = paddle.randn(shape=(M, K), dtype="float16").to(dtype)
    b = paddle.randn(shape=(K, N), dtype="float16").to(dtype)
    b = b.T.contiguous()
    c = paddle.randn(shape=(M, N), dtype=out_dtype)
    c_unmodified = c.clone()
    c0 = c.clone()
    c1 = c.clone()
    c_torch = torch_gemm(a.to(out_dtype), b.to(out_dtype), c.to(out_dtype), alpha, beta)
    c_persistent = flashinfer.triton.sm_constraint_gemm.gemm_persistent(
        a, b.T, c=c, alpha=alpha, beta=beta, num_sms=num_sms
    )
    c_naive = flashinfer.triton.sm_constraint_gemm.gemm(
        a, b.T, c=c0, alpha=alpha, beta=beta
    )
    c_descriptor = None
    if dtype != "float32":
        c_descriptor = flashinfer.triton.sm_constraint_gemm.gemm_descriptor_persistent(
            a,
            b,
            c=c1,
            alpha=alpha,
            beta=beta,
            num_sms=num_sms,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
        )
    torch_atol = 20.0 if out_dtype == "bfloat16" else 1.0
    in_place_persistent = (
        c_persistent.data_ptr() == c.data_ptr()
        and paddle.allclose(x=c_persistent.to(out_dtype), y=c.to(out_dtype)).item()
    )
    assert in_place_persistent
    in_place_naive = (
        c_naive.data_ptr() == c0.data_ptr()
        and paddle.allclose(x=c_naive.to(out_dtype), y=c0.to(out_dtype)).item()
    )
    assert in_place_naive
    if c_descriptor is not None:
        in_place_descriptor = (
            c_descriptor.data_ptr() == c1.data_ptr()
            and paddle.allclose(x=c_descriptor.to(out_dtype), y=c1.to(out_dtype)).item()
        )
        assert in_place_descriptor
    torch_vs_triton_persistent = paddle.allclose(
        x=c_torch.to(out_dtype), y=c_persistent.to(out_dtype), atol=torch_atol
    ).item()
    if not torch_vs_triton_persistent:
        print_all_on_failure(
            a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor, out_dtype
        )
        print("compare c_torch and c_persistent")
        print_max_diff_on_failure(c_torch, c_persistent, out_dtype)
    assert torch_vs_triton_persistent
    if c_descriptor is not None:
        torch_vs_triton_descriptor = paddle.allclose(
            x=c_torch.to(out_dtype), y=c_descriptor.to(out_dtype), atol=torch_atol
        ).item()
        if not torch_vs_triton_descriptor:
            print_all_on_failure(
                a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor
            )
            print("compare c_torch and c_descriptor")
            print_max_diff_on_failure(c_torch, c_descriptor, out_dtype)
        assert torch_vs_triton_descriptor
    triton_atol = 10.0 if out_dtype == "bfloat16" else 1.0
    naive_vs_persistent = paddle.allclose(
        x=c_naive.to(out_dtype), y=c_persistent.to(out_dtype), atol=triton_atol
    ).item()
    if not naive_vs_persistent:
        print_all_on_failure(
            a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor, out_dtype
        )
        print("compare c_naive and c_persistent")
        print_max_diff_on_failure(c_naive, c_persistent, out_dtype)
    assert naive_vs_persistent
    if c_descriptor is not None:
        descriptor_atol = 10.0 if out_dtype == "bfloat16" else 1.0
        naive_vs_descriptor = paddle.allclose(
            x=c_naive.to(out_dtype), y=c_descriptor.to(out_dtype), atol=descriptor_atol
        ).item()
        if not naive_vs_descriptor:
            print_all_on_failure(
                a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor
            )
            print("compare c_naive and c_descriptor")
            print_max_diff_on_failure(c_naive, c_descriptor, out_dtype)
        assert naive_vs_descriptor


def print_all_on_failure(
    a, b, c_unmodified, c_torch, c_naive, c_persistent, c_descriptor
):
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c_unmodified: {c_unmodified}")
    if c_torch is not None:
        print(f"c_torch: {c_torch}")
    print(f"c_naive: {c_naive}")
    print(f"c_persistent: {c_persistent}")
    if c_descriptor is not None:
        print(f"c_descriptor: {c_descriptor}")


def print_max_diff_on_failure(target1, target2, out_dtype):
    max_diff = paddle.max(x=paddle.abs(x=target1.to(out_dtype) - target2.to(out_dtype)))
    print(f"max diff: {max_diff}")
    max_diff_index = paddle.argmax(
        x=paddle.abs(x=target1.to(out_dtype) - target2.to(out_dtype))
    )
    print(f"max diff index: {max_diff_index}")
    if target1.dim() > 1:
>>>>>>        max_diff_index = torch.unravel_index(max_diff_index, tuple(target1.shape))
    print(f"target1[max_diff_index]: {target1[max_diff_index]}")
    print(f"target2[max_diff_index]: {target2[max_diff_index]}")
