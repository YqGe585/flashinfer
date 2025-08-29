from typing import Optional

import paddle

from .kernels.cascade import (merge_state_in_place_kernel, merge_state_kernel,
                              merge_states_kernel,
                              variable_length_merge_states_kernel)
from .utils import check_device, check_dim, check_input, check_shape


def merge_state(
    v_a: paddle.Tensor, s_a: paddle.Tensor, v_b: paddle.Tensor, s_b: paddle.Tensor
):
    check_input(v_a)
    check_input(s_a)
    check_input(v_b)
    check_input(s_b)
    check_device([v_a, s_a, v_b, s_b])
    check_dim(3, v_a)
    check_dim(2, s_a)
    check_dim(3, v_b)
    check_dim(2, s_b)
    check_shape(v_a, v_b)
    check_shape(s_a, s_b)
    assert v_a.shape[0] == s_a.shape[0]
    assert v_a.shape[1] == s_b.shape[1]
    s_a = s_a.to("float32")
    s_b = s_b.to("float32")
    seq_len = v_a.shape[0]
    num_heads = v_a.shape[1]
    head_dim = v_a.shape[2]
    v_merged = paddle.empty_like(x=v_a).to(s_a.place)
    s_merged = paddle.empty(shape=(seq_len, num_heads)).to(s_a.place)
    bdx = head_dim
    bdy = num_heads
    merge_state_kernel[lambda meta: (seq_len,)](
        v_a, s_a, v_b, s_b, v_merged, s_merged, num_heads, head_dim, bdx=bdx, bdy=bdy
    )
    return v_merged, s_merged


def merge_state_in_place(
    v: paddle.Tensor,
    s: paddle.Tensor,
    v_other: paddle.Tensor,
    s_other: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
):
    check_input(v)
    check_input(s)
    check_input(v_other)
    check_input(s_other)
    check_device([v, s, v_other, s_other])
    check_dim(3, v)
    check_dim(2, s)
    check_dim(3, v_other)
    check_dim(2, s_other)
    check_shape(v, v_other)
    check_shape(s, s_other)
    assert v.shape[0] == s.shape[0]
    assert v.shape[1] == s.shape[1]
    assert s.dtype == "float32"
    assert s_other.dtype == "float32"
    if mask is not None:
        check_dim(1, mask)
        assert v.shape[0] == mask.shape[0]
        assert mask.place == v.place
    seq_len = v.shape[0]
    num_heads = v.shape[1]
    head_dim = v.shape[2]
    bdx = head_dim
    bdy = num_heads
    merge_state_in_place_kernel[
        seq_len,
    ](v, s, v_other, s_other, num_heads, head_dim, mask, bdx=bdx, bdy=bdy)


def merge_states(v: paddle.Tensor, s: paddle.Tensor):
    check_input(v)
    check_input(s)
    check_device([v, s])
    check_dim(4, v)
    check_dim(3, s)
    assert v.shape[0] == s.shape[0]
    assert v.shape[1] == s.shape[1]
    assert v.shape[2] == s.shape[2]
    seq_len = v.shape[0]
    num_index_sets = v.shape[1]
    num_heads = v.shape[2]
    head_dim = v.shape[3]
    s = s.to("float32")
    v_merged = paddle.empty(shape=(seq_len, num_heads, head_dim), dtype=v.dtype)
    s_merged = paddle.empty(shape=(seq_len, num_heads), dtype=s.dtype)
    bdx = head_dim
    bdy = num_heads
    merge_states_kernel[
        seq_len,
    ](v, s, v_merged, s_merged, num_index_sets, num_heads, head_dim, bdx=bdx, bdy=bdy)
    return v_merged, s_merged


def variable_length_merge_states(
    v: paddle.Tensor, s: paddle.Tensor, indptr: paddle.Tensor
):
    check_input(v)
    check_input(s)
    check_device([v, s])
    check_dim(3, v)
    check_dim(2, s)
    assert v.shape[0] == s.shape[0]
    assert v.shape[1] == s.shape[1]
    seq_len = indptr.shape[0] - 1
    num_heads = v.shape[1]
    head_dim = v.shape[2]
    s = s.to("float32")
    indptr = indptr.to("int32")
    v_merged = paddle.empty(shape=(seq_len, num_heads, head_dim), dtype=v.dtype)
    s_merged = paddle.empty(shape=(seq_len, num_heads), dtype=s.dtype)
    bdx = head_dim
    bdy = num_heads
    variable_length_merge_states_kernel[
        seq_len,
    ](v, s, indptr, v_merged, s_merged, num_heads, head_dim, bdx=bdx, bdy=bdy)
    return v_merged, s_merged
