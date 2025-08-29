import paddle

"""
Attention with Linear Biases (ALiBi) reference implementation.

Code adapted from https://github.com/labmlai/annotated_deep_learning_paper_implementations

Licensed under MIT, you may obtain a copy of the License at

  https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license

Source:
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/285cb3735bde02fbc8c19ddeb24d0ae7e77135c1/labml_nn/transformers/mha.py
- https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/285cb3735bde02fbc8c19ddeb24d0ae7e77135c1/labml_nn/transformers/alibi/__init__.py
"""
import math
from typing import Optional


def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\\frac{1}{2^{\\frac{8}{n}}} = 2^{-\\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\\frac{1}{2^1}, \\frac{1}{2^2}, \\dots, \\frac{1}{2^8}$$
    """
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = paddle.pow(x=m_0, y=paddle.arange(start=1, end=1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = paddle.pow(
            x=m_hat_0, y=paddle.arange(start=1, end=1 + 2 * (n_heads - n), step=2)
        )
        m = paddle.concat(x=[m, m_hat])
    return m


@paddle.no_grad()
def get_alibi_biases(n_heads: int, mask: paddle.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[seq_len_q, seq_len_k, n_heads, ]` with ALiBi attention biases.
    """
    m = get_slopes(n_heads).to(mask.place)
    distance = paddle.arange(dtype="int64", end=tuple(mask.shape)[1])[None, :]
    return distance[:, :, None] * m[None, None, :]


def alibi_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    mask: Optional[paddle.Tensor] = None,
):
    """
    query: [q_len, num_heads, head_dim]
    key: [kv_len, num_heads, head_dim]
    value: [kv_len, num_heads, head_dim]
    mask: [q_len, kv_len]
    """
    q_len, num_heads, head_dim = tuple(query.shape)
    scores = paddle.einsum(
        "qhd,khd->qkh", query.astype(dtype="float32"), key.astype(dtype="float32")
    )
    scores *= 1.0 / math.sqrt(head_dim)
    alibi_biases = get_alibi_biases(num_heads, mask)
    scores += alibi_biases
    scores = scores.masked_fill(mask=mask.unsqueeze(axis=-1) == 0, value=float("-inf"))
    attn = paddle.nn.functional.softmax(x=scores, axis=1)
    return paddle.einsum("ovh,vhd->ohd", attn, value.astype(dtype="float32")).to(query)
