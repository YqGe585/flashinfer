import sys

sys.path.append("/home/flashinfer")
import paddle
from paddle_utils import *

"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pytest

import flashinfer


def normal_distribution(std):
    def normal_noise(shape, device):
        return paddle.randn(shape=shape) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = paddle.rand(shape=shape)
        eps = 1e-20
        return paddle.log(x=-paddle.log(x=U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
)
@pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
@pytest.mark.parametrize("temperature_arr", [True, False])
@pytest.mark.parametrize("neg_inf_input", [True, False])
def test_softmax(
    batch_size, vocab_size, distribution, temperature, temperature_arr, neg_inf_input
):
    paddle.seed(seed=42)
    logits = distribution((batch_size, vocab_size), "cuda:0")
    if neg_inf_input:
        num_inf = paddle.randint(low=0, high=logits.size - 1, shape=()).item()
        inf_idx = paddle.randperm(n=logits.size)[:num_inf]
        logits.view(-1).index_fill_(axis=0, index=inf_idx, value=float("-inf"))
    if temperature_arr:
        temperature_arr = paddle.full(shape=(batch_size,), fill_value=temperature)
        probs = flashinfer.sampling.softmax(logits, temperature=temperature_arr)
        logits_scaled = logits / temperature_arr.unsqueeze(axis=-1)
    else:
        probs = flashinfer.sampling.softmax(logits, temperature=temperature)
        logits_scaled = logits / temperature
    probs_ref = paddle.nn.functional.softmax(x=logits_scaled, axis=-1)
    assert paddle.allclose(x=probs, y=probs_ref, atol=1e-05).item()


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
)
@pytest.mark.parametrize("zero_ratio", [0.0, 0.5, 0.9])
def test_sampling_freq(vocab_size, distribution, zero_ratio):
    paddle.seed(seed=42)
    num_trials = 5000000
    logits = distribution((1, vocab_size), "cuda:0")
    zero_indices = paddle.randperm(n=vocab_size)[: int(vocab_size * zero_ratio)]
    logits[:, zero_indices] = -float("inf")
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    counter = paddle.zeros(shape=vocab_size, dtype="int32")
    samples = flashinfer.sampling.sampling_from_probs(
        probs, indices=paddle.zeros(shape=num_trials, dtype="int32")
    )
    counter.put_along_axis_(
        axis=0,
        indices=samples.astype(dtype="int64"),
        values=paddle.ones_like(x=samples),
        reduce="add",
    )
    freq = counter.astype(dtype="float32") / num_trials
    assert paddle.all(x=counter[zero_indices] == 0)
    similarity = paddle.nn.functional.cosine_similarity(x1=freq, x2=probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
)
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling_freq(vocab_size, distribution, p):
    paddle.seed(seed=42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    sorted_prob, indices = paddle.sort(x=probs, descending=False), paddle.argsort(
        x=probs, descending=False
    )
    cdf = paddle.cumsum(x=sorted_prob, axis=-1)
    mask = paddle.zeros(shape=[1, vocab_size], dtype="int32")
    mask.put_along_axis_(
        axis=1,
        indices=indices,
        values=(cdf > 1 - p).astype(dtype="int32"),
        reduce="add",
    )
    renorm_probs = flashinfer.sampling.top_p_renorm_probs(probs, p)
    counter = paddle.zeros(shape=vocab_size, dtype="int32")
    num_trials = 5000000
    samples = flashinfer.sampling.top_p_sampling_from_probs(
        probs, p, indices=paddle.zeros(shape=num_trials, dtype="int32")
    )
    counter.put_along_axis_(
        axis=0,
        indices=samples.astype(dtype="int64"),
        values=paddle.ones_like(x=samples),
        reduce="add",
    )
    freq = counter.astype(dtype="float32") / num_trials
    assert paddle.all(x=mask[paddle.arange(end=1), samples] == 1)
    similarity = paddle.nn.functional.cosine_similarity(x1=freq, x2=renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
)
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_freq(vocab_size, distribution, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    paddle.seed(seed=42)
    logits = distribution((1, vocab_size), "cuda:0")
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    sorted_prob, _ = paddle.sort(x=probs, descending=True), paddle.argsort(
        x=probs, descending=True
    )
    pivot = sorted_prob[:, k - 1]
    mask = (probs >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
    renorm_probs = flashinfer.sampling.top_k_renorm_probs(probs, k)
    counter = paddle.zeros(shape=vocab_size, dtype="int32")
    num_trials = 5000000
    samples = flashinfer.sampling.top_k_sampling_from_probs(
        probs, k, indices=paddle.zeros(shape=num_trials, dtype="int32")
    )
    counter.put_along_axis_(
        axis=0,
        indices=samples.astype(dtype="int64"),
        values=paddle.ones_like(x=samples),
        reduce="add",
    )
    freq = counter.astype(dtype="float32") / num_trials
    assert paddle.all(x=mask[paddle.arange(end=1), samples] == 1)
    similarity = paddle.nn.functional.cosine_similarity(x1=freq, x2=renorm_probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling(batch_size, vocab_size):
    paddle.seed(seed=42)
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    num_trails = 5000
    for _ in range(num_trails):
        samples = flashinfer.sampling.sampling_from_probs(normalized_prob)
        assert paddle.all(x=samples < vocab_size) and paddle.all(x=samples >= 0)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
def test_sampling_from_logits(batch_size, vocab_size):
    paddle.seed(seed=42)
    logits = paddle.randn(shape=[batch_size, vocab_size])
    num_trails = 5000
    for _ in range(num_trails):
        samples = flashinfer.sampling.sampling_from_logits(logits)
        assert paddle.all(x=samples < vocab_size) and paddle.all(x=samples >= 0)


@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize(
    "distribution",
    [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
)
def test_sampling_from_logits_freq(vocab_size, distribution):
    paddle.seed(seed=42)
    num_trials = 5000000
    logits = distribution((1, vocab_size), "cuda:0")
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    counter = paddle.zeros(shape=vocab_size, dtype="int32")
    samples = flashinfer.sampling.sampling_from_logits(
        logits, indices=paddle.zeros(shape=num_trials, dtype="int32")
    )
    counter.put_along_axis_(
        axis=0,
        indices=samples.astype(dtype="int64"),
        values=paddle.ones_like(x=samples),
        reduce="add",
    )
    freq = counter.astype(dtype="float32") / num_trials
    similarity = paddle.nn.functional.cosine_similarity(x1=freq, x2=probs)
    assert similarity > 0.99, f"similarity: {similarity}"


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_sampling(batch_size, vocab_size, p):
    paddle.seed(seed=42)
    eps = 0.0001
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, indices = paddle.sort(
        x=normalized_prob, descending=False
    ), paddle.argsort(x=normalized_prob, descending=False)
    cdf = paddle.cumsum(x=sorted_prob, axis=-1)
    mask = paddle.zeros(shape=[batch_size, vocab_size], dtype="int32")
    mask.put_along_axis_(
        axis=1,
        indices=indices,
        values=(cdf > 1 - p - eps).astype(dtype="int32"),
        reduce="add",
    )
    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_p_sampling_from_probs(normalized_prob, p)
        assert paddle.all(x=samples < vocab_size) and paddle.all(x=samples >= 0)
        assert paddle.all(x=mask[paddle.arange(end=batch_size), samples] == 1)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    paddle.seed(seed=42)
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, _ = paddle.sort(x=normalized_prob, descending=True), paddle.argsort(
        x=normalized_prob, descending=True
    )
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)
        assert paddle.all(x=samples < vocab_size) and paddle.all(x=samples >= 0)
        assert paddle.all(
            x=mask[paddle.arange(end=batch_size), samples] == 1
        ), normalized_prob[paddle.arange(end=batch_size), samples]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_sampling_with_variable_k(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    paddle.seed(seed=42)
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, _ = paddle.sort(x=normalized_prob, descending=True), paddle.argsort(
        x=normalized_prob, descending=True
    )
    k = paddle.randint(low=1, high=k + 1, shape=(batch_size,))
    pivot = sorted_prob[paddle.arange(end=batch_size), k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_sampling_from_probs(normalized_prob, k)
        assert paddle.all(x=samples < vocab_size) and paddle.all(x=samples >= 0)
        assert paddle.all(
            x=mask[paddle.arange(end=batch_size), samples] == 1
        ), normalized_prob[paddle.arange(end=batch_size), samples]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_min_p_sampling(batch_size, vocab_size, p):
    paddle.seed(seed=42)
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, indices = paddle.sort(
        x=normalized_prob, descending=False
    ), paddle.argsort(x=normalized_prob, descending=False)
    top_probs = sorted_prob[:, -1].unsqueeze(axis=-1)
    scaled_p = p * top_probs
    mask = paddle.zeros(shape=[batch_size, vocab_size], dtype="int32")
    mask.put_along_axis_(
        axis=1,
        indices=indices,
        values=(sorted_prob >= scaled_p).astype(dtype="int32"),
        reduce="add",
    )
    min_p_tensor = paddle.full(shape=(batch_size,), fill_value=p)
    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.min_p_sampling_from_probs(
            normalized_prob, min_p_tensor
        )
        assert paddle.all(x=mask[paddle.arange(end=batch_size), samples] == 1), samples[
            paddle.nonzero(x=mask[paddle.arange(end=batch_size), samples] == 0)
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    paddle.seed(seed=42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    eps = 0.0001
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, indices = paddle.sort(
        x=normalized_prob, descending=False
    ), paddle.argsort(x=normalized_prob, descending=False)
    cdf = paddle.cumsum(x=sorted_prob, axis=-1)
    mask_top_p = paddle.zeros(shape=[batch_size, vocab_size], dtype="int32")
    mask_top_p.put_along_axis_(
        axis=1,
        indices=indices,
        values=(cdf > 1 - p - eps).astype(dtype="int32"),
        reduce="add",
    )
    sorted_prob, _ = paddle.sort(x=normalized_prob, descending=True), paddle.argsort(
        x=normalized_prob, descending=True
    )
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
    mask = paddle.minimum(x=mask_top_p, y=mask_top_k)
    top_p_tensor = paddle.full(shape=(batch_size,), fill_value=p)
    top_k_tensor = paddle.full(shape=(batch_size,), fill_value=k)
    num_trails = 1000
    for _ in range(num_trails):
        samples = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            normalized_prob, top_k_tensor, top_p_tensor, filter_apply_order="joint"
        )
        assert paddle.all(x=samples < vocab_size) and paddle.all(x=samples >= 0)
        assert paddle.all(
            x=mask[paddle.arange(end=batch_size), samples] == 1
        ), normalized_prob[paddle.arange(end=batch_size), samples]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [100])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_sampling_from_probs_logits_alignment(batch_size, vocab_size, k, p):
    paddle.seed(seed=42)
    logits = paddle.randn(shape=[batch_size, vocab_size]) * 5
    generator_logits = paddle.framework.core.default_cpu_generator()
    generator_probs = generator_logits.clone_state()
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="top_k_first", generator=generator_logits
    )
    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        paddle.nn.functional.softmax(x=logits, axis=-1),
        k,
        p,
        filter_apply_order="top_k_first",
        generator=generator_probs,
    )
    assert paddle.all(x=samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_logits(batch_size, vocab_size, p):
    paddle.seed(seed=42)
    logits = paddle.rand(shape=[batch_size, vocab_size]) * 5
    generator_logits = paddle.framework.core.default_cpu_generator()
    generator_probs = generator_logits.clone_state()
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    samples = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="joint", generator=generator_logits
    )
    samples_ref = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        paddle.nn.functional.softmax(x=logits, axis=-1),
        k,
        p,
        filter_apply_order="joint",
        generator=generator_probs,
    )
    assert paddle.all(x=samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    paddle.seed(seed=42)
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, indices = paddle.sort(
        x=normalized_prob, descending=False
    ), paddle.argsort(x=normalized_prob, descending=False)
    cdf = paddle.cumsum(x=sorted_prob, axis=-1)
    mask = paddle.zeros(shape=[batch_size, vocab_size], dtype="int32")
    mask.put_along_axis_(
        axis=1,
        indices=indices,
        values=(cdf >= 1 - p).astype(dtype="int32"),
        reduce="add",
    )
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        axis=-1, keepdim=True
    )
    renorm_prob = flashinfer.sampling.top_p_renorm_probs(normalized_prob, p)
    assert paddle.allclose(
        x=renorm_prob_ground_truth, y=renorm_prob, rtol=0.001, atol=0.001
    ).item(), ""


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    paddle.seed(seed=42)
    pre_norm_prob = paddle.rand(shape=[batch_size, vocab_size])
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(axis=-1, keepdim=True)
    sorted_prob, _ = paddle.sort(x=normalized_prob, descending=True), paddle.argsort(
        x=normalized_prob, descending=True
    )
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        axis=-1, keepdim=True
    )
    renorm_prob = flashinfer.sampling.top_k_renorm_probs(normalized_prob, k)
    for i in range(batch_size):
        assert paddle.allclose(
            x=renorm_prob_ground_truth[i], y=renorm_prob[i], rtol=0.001, atol=0.001
        ).item(), ""


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
@pytest.mark.parametrize("neginf_input", [False, True])
def test_top_k_mask_logits(batch_size, vocab_size, k, neginf_input):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    paddle.seed(seed=42)
    logits = paddle.randn(shape=[batch_size, vocab_size]) * 5
    if neginf_input:
        num_neginf = paddle.randint(
            low=1, high=vocab_size * batch_size, shape=(1,)
        ).item()
        idxs = paddle.randperm(n=batch_size * vocab_size)[:num_neginf]
        logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")
    probs = paddle.nn.functional.softmax(x=logits, axis=-1)
    masked_logits = flashinfer.sampling.top_k_mask_logits(logits, k)
    renormed_probs = paddle.nn.functional.softmax(x=masked_logits, axis=-1)
    renormed_probs_ref = flashinfer.sampling.top_k_renorm_prob(probs, k)
    assert paddle.allclose(
        x=renormed_probs, y=renormed_probs_ref, rtol=0.001, atol=0.001
    ).item(), ""


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("num_speculate_tokens", [1, 3, 5, 7])
@pytest.mark.parametrize("onehot_target", [False, True])
def test_chain_speculative_sampling(
    batch_size, vocab_size, num_speculate_tokens, onehot_target
):
    pre_norm_draft_prob = paddle.rand(
        shape=[batch_size, num_speculate_tokens, vocab_size]
    )
    normalized_draft_prob = pre_norm_draft_prob / pre_norm_draft_prob.sum(
        axis=-1, keepdim=True
    )
    draft_token_ids = paddle.randint(
        low=0, high=vocab_size, shape=(batch_size, num_speculate_tokens)
    )
    if not onehot_target:
        pre_norm_target_prob = paddle.rand(
            shape=[batch_size, num_speculate_tokens + 1, vocab_size]
        )
        target_onehot_prob = pre_norm_target_prob / pre_norm_target_prob.sum(
            axis=-1, keepdim=True
        )
    else:
        target_token_ids = paddle.randint(
            low=0, high=vocab_size, shape=(batch_size, num_speculate_tokens + 1)
        )
        target_token_ids[..., :num_speculate_tokens] = draft_token_ids
        target_onehot_prob = paddle.zeros(
            shape=(batch_size, num_speculate_tokens + 1, vocab_size)
        )
        target_onehot_prob.put_along_axis_(
            axis=2,
            indices=target_token_ids.unsqueeze(axis=-1),
            values=1,
            broadcast=False,
        )
    for trials in range(10):
        accepted_num = paddle.zeros(shape=batch_size, dtype="int32")
        emitted_num = paddle.zeros(shape=batch_size, dtype="int32")
        (
            output_token_ids,
            accepted_num,
            emitted_num,
        ) = flashinfer.sampling.chain_speculative_sampling(
            normalized_draft_prob,
            draft_token_ids,
            target_onehot_prob,
            accepted_num,
            emitted_num,
        )
        if onehot_target:
            assert paddle.all(x=output_token_ids == target_token_ids)
        else:
            assert paddle.all(x=output_token_ids[output_token_ids >= 0] < vocab_size)
            assert tuple(output_token_ids.shape) == (
                batch_size,
                num_speculate_tokens + 1,
            )
            matches = output_token_ids[..., :-1] != draft_token_ids
            for row in range(batch_size):
                paddle.utils.try_import("warnings").warn(
                    "Now, the return shape is inconsistent with torch when as_tuple is True"
                )
                mismatch_idx = paddle.nonzero(x=matches[row], as_tuple=True)[0]
                if len(mismatch_idx) > 0:
                    assert paddle.all(x=mismatch_idx[1:] == mismatch_idx[:-1] + 1)
                    assert paddle.all(
                        x=output_token_ids[row, mismatch_idx[0] + 1 :] == -1
                    )
        assert paddle.all(x=emitted_num + 1 == (output_token_ids != -1).sum(axis=1))


if __name__ == "__main__":
    test_sampling_from_logits_freq(128256, gumbel_distribution(0.1))
