import paddle
import pytest

import flashinfer
import flashinfer.triton


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_merge_state(seq_len, num_heads, head_dim):
    va = (
        paddle.randn(shape=[seq_len, num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    sa = paddle.randn(shape=[seq_len, num_heads], dtype="float32").to("gpu:0")
    vb = (
        paddle.randn(shape=[seq_len, num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    sb = paddle.randn(shape=[seq_len, num_heads], dtype="float32").to("gpu:0")
    v_merged, s_merged = flashinfer.triton.cascade.merge_state(va, sa, vb, sb)
    v_merged_std, s_merged_std = flashinfer.merge_state(va, sa, vb, sb)
    assert paddle.allclose(x=v_merged, y=v_merged_std, atol=0.01).item()
    assert paddle.allclose(x=s_merged, y=s_merged_std, atol=0.01).item()


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_merge_state_in_place(seq_len, num_heads, head_dim):
    v = paddle.randn(shape=[seq_len, num_heads, head_dim]).astype(dtype="float16")
    v_std = v.clone()
    v, v_std = v.to("gpu:0"), v_std.to("gpu:0")
    s = paddle.randn(shape=[seq_len, num_heads], dtype="float32")
    s_std = s.clone()
    s, s_std = s.to("gpu:0"), s_std.to("gpu:0")
    v_other = (
        paddle.randn(shape=[seq_len, num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    s_other = paddle.randn(shape=[seq_len, num_heads], dtype="float32").to("gpu:0")
    flashinfer.merge_state_in_place(v_std, s_std, v_other, s_other)
    flashinfer.triton.cascade.merge_state_in_place(v, s, v_other, s_other)
    assert paddle.allclose(x=v, y=v_std, atol=0.01).item()
    assert paddle.allclose(x=s, y=s_std, atol=0.01).item()


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_states", [100])
def test_merge_states(seq_len, num_states, num_heads, head_dim):
    v = (
        paddle.randn(shape=[seq_len, num_states, num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    s = paddle.randn(shape=[seq_len, num_states, num_heads], dtype="float32").to(
        "gpu:0"
    )
    v_merged_std, s_merged_std = flashinfer.merge_states(v, s)
    v_merged, s_merged = flashinfer.triton.cascade.merge_states(v, s)
    assert paddle.allclose(x=v_merged, y=v_merged_std, atol=0.01).item()
    assert paddle.allclose(x=s_merged, y=s_merged_std, atol=0.01).item()


@pytest.mark.parametrize("seq_len", [2048])
@pytest.mark.parametrize("num_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
def test_variable_length_merge_states(seq_len, num_heads, head_dim):
    max_index_sets = 512
    lengths = paddle.randint(low=1, high=max_index_sets, shape=(seq_len,))
    indptr = [0]
    for i in range(seq_len):
        indptr.append(indptr[-1] + lengths[i])
    v = (
        paddle.randn(shape=[indptr[-1], num_heads, head_dim])
        .astype(dtype="float16")
        .to("gpu:0")
    )
    s = paddle.randn(shape=[indptr[-1], num_heads], dtype="float32").to("gpu:0")
    indptr = paddle.to_tensor(data=indptr, dtype="int32").to("gpu:0")
    v_merged, s_merged = flashinfer.triton.cascade.variable_length_merge_states(
        v, s, indptr
    )
    for i in range(seq_len):
        sub_v = v[indptr[i] : indptr[i + 1]]
        sub_s = s[indptr[i] : indptr[i + 1]]
        sub_v = paddle.unsqueeze(x=sub_v, axis=0)
        sub_s = paddle.unsqueeze(x=sub_s, axis=0)
        v_merged_std, s_merged_std = flashinfer.merge_states(sub_v, sub_s)
        v_merged_std = paddle.squeeze(x=v_merged_std, axis=0)
        s_merged_std = paddle.squeeze(x=s_merged_std, axis=0)
        assert tuple(v_merged[i].shape) == tuple(v_merged_std.shape)
        assert paddle.allclose(x=v_merged[i], y=v_merged_std, atol=0.01).item()
        assert paddle.allclose(x=s_merged[i], y=s_merged_std, atol=0.01).item()
