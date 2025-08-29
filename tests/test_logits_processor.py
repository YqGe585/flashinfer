import numpy as np
import paddle
import pytest

import flashinfer
from flashinfer.logits_processor import (LogitsPipe, MinP, Sample, Softmax,
                                         Temperature, TensorType, TopK, TopP)


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


def set_random_seed(seed=42):
    paddle.seed(seed=seed)
    np.random.seed(seed)


def get_generators():
    gen1 = paddle.framework.core.default_cpu_generator()
    gen1.manual_seed(42)
    gen2 = gen1.clone_state()
    return gen1, gen2


class TestLogitsPipeCompilation:
    """Test LogitsPipe with compile=True vs compile=False"""

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
    def test_temperature_softmax(
        self, batch_size, vocab_size, distribution, temperature
    ):
        set_random_seed(42)
        logits = distribution((batch_size, vocab_size), "cuda:0")
        pipe_compiled = LogitsPipe([Temperature(), Softmax()], compile=True)
        pipe_no_compile = LogitsPipe([Temperature(), Softmax()], compile=False)
        probs_compiled = pipe_compiled(logits, temperature=temperature)
        probs_no_compile = pipe_no_compile(logits, temperature=temperature)
        assert paddle.allclose(x=probs_compiled, y=probs_no_compile, atol=1e-05).item()

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("zero_ratio", [0.0, 0.5, 0.9])
    def test_probs_sample_freq(self, vocab_size, distribution, zero_ratio):
        set_random_seed(42)
        num_trials = 5000000
        logits = distribution((1, vocab_size), "cuda:0")
        zero_indices = paddle.randperm(n=vocab_size)[: int(vocab_size * zero_ratio)]
        logits[:, zero_indices] = -float("inf")
        probs = paddle.nn.functional.softmax(x=logits, axis=-1)
        pipe_compiled = LogitsPipe(
            [Sample()], compile=True, input_type=TensorType.PROBS
        )
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32")
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe(
            [Sample()], compile=False, input_type=TensorType.PROBS
        )
        counter_no_compile = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_no_compile = pipe_no_compile(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32")
        )
        counter_no_compile.put_along_axis_(
            axis=0,
            indices=samples_no_compile.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_no_compile),
            reduce="add",
        )
        freq_no_compile = counter_no_compile.astype(dtype="float32") / num_trials
        assert paddle.all(x=counter_compiled[zero_indices] == 0) and paddle.all(
            x=counter_no_compile[zero_indices] == 0
        )
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=probs
        )
        similarity_no_compile = paddle.nn.functional.cosine_similarity(
            x1=freq_no_compile, x2=probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert (
            similarity_no_compile > 0.99
        ), f"Non-compiled similarity: {similarity_no_compile}"
        freq_similarity = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=freq_no_compile, axis=0
        )
        assert (
            freq_similarity > 0.99
        ), f"Compiled vs non-compiled similarity: {freq_similarity}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    def test_logits_sample_freq(self, vocab_size, distribution):
        set_random_seed(42)
        num_trials = 5000000
        logits = distribution((1, vocab_size), "cuda:0")
        probs = paddle.nn.functional.softmax(x=logits, axis=-1)
        pipe_compiled = LogitsPipe(
            [Sample()], compile=True, input_type=TensorType.LOGITS
        )
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            logits, indices=paddle.zeros(shape=num_trials, dtype="int32")
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe(
            [Sample()], compile=False, input_type=TensorType.LOGITS
        )
        counter_no_compile = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_no_compile = pipe_no_compile(
            logits, indices=paddle.zeros(shape=num_trials, dtype="int32")
        )
        counter_no_compile.put_along_axis_(
            axis=0,
            indices=samples_no_compile.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_no_compile),
            reduce="add",
        )
        freq_no_compile = counter_no_compile.astype(dtype="float32") / num_trials
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=probs
        )
        similarity_no_compile = paddle.nn.functional.cosine_similarity(
            x1=freq_no_compile, x2=probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert (
            similarity_no_compile > 0.99
        ), f"Non-compiled similarity: {similarity_no_compile}"
        freq_similarity = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=freq_no_compile, axis=0
        )
        assert (
            freq_similarity > 0.99
        ), f"Compiled vs non-compiled similarity: {freq_similarity}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("k", [10, 100, 500])
    def test_probs_top_k_sample_freq(self, vocab_size, distribution, k):
        if k > vocab_size:
            pytest.skip("k should be less than vocab_size")
        set_random_seed(42)
        num_trials = 5000000
        logits = distribution((1, vocab_size), "cuda:0")
        probs = paddle.nn.functional.softmax(x=logits, axis=-1)
        sorted_prob, _ = paddle.sort(x=probs, descending=True), paddle.argsort(
            x=probs, descending=True
        )
        pivot = sorted_prob[:, k - 1]
        mask = (probs >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0
        pipe_compiled = LogitsPipe(
            [TopK(), Sample()], compile=True, input_type=TensorType.PROBS
        )
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32"), top_k=k
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe(
            [TopK(), Sample()], compile=False, input_type=TensorType.PROBS
        )
        counter_no_compile = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_no_compile = pipe_no_compile(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32"), top_k=k
        )
        counter_no_compile.put_along_axis_(
            axis=0,
            indices=samples_no_compile.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_no_compile),
            reduce="add",
        )
        freq_no_compile = counter_no_compile.astype(dtype="float32") / num_trials
        assert paddle.all(x=mask[paddle.arange(end=1), samples_compiled] == 1)
        assert paddle.all(x=mask[paddle.arange(end=1), samples_no_compile] == 1)
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=masked_probs
        )
        similarity_no_compile = paddle.nn.functional.cosine_similarity(
            x1=freq_no_compile, x2=masked_probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert (
            similarity_no_compile > 0.99
        ), f"Non-compiled similarity: {similarity_no_compile}"
        freq_similarity = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=freq_no_compile, axis=0
        )
        assert (
            freq_similarity > 0.99
        ), f"Compiled vs non-compiled similarity: {freq_similarity}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_probs_top_p_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        eps = 0.0001
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
            values=(cdf > 1 - p - eps).astype(dtype="int32"),
            reduce="add",
        )
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0
        pipe_compiled = LogitsPipe([TopP(), Sample()], compile=True)
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32"), top_p=p
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe(
            [TopP(), Sample()], compile=False, input_type=TensorType.PROBS
        )
        counter_no_compile = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_no_compile = pipe_no_compile(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32"), top_p=p
        )
        counter_no_compile.put_along_axis_(
            axis=0,
            indices=samples_no_compile.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_no_compile),
            reduce="add",
        )
        freq_no_compile = counter_no_compile.astype(dtype="float32") / num_trials
        assert paddle.all(x=mask[paddle.arange(end=1), samples_compiled] == 1)
        assert paddle.all(x=mask[paddle.arange(end=1), samples_no_compile] == 1)
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=masked_probs
        )
        similarity_no_compile = paddle.nn.functional.cosine_similarity(
            x1=freq_no_compile, x2=masked_probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert (
            similarity_no_compile > 0.99
        ), f"Non-compiled similarity: {similarity_no_compile}"
        freq_similarity = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=freq_no_compile, axis=0
        )
        assert (
            freq_similarity > 0.99
        ), f"Compiled vs non-compiled similarity: {freq_similarity}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
    def test_probs_min_p_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        logits = distribution((1, vocab_size), "cuda:0")
        probs = paddle.nn.functional.softmax(x=logits, axis=-1)
        sorted_prob, indices = paddle.sort(x=probs, descending=False), paddle.argsort(
            x=probs, descending=False
        )
        top_probs = sorted_prob[:, -1].unsqueeze(axis=-1)
        scaled_p = p * top_probs
        mask = paddle.zeros(shape=[1, vocab_size], dtype="int32")
        mask.put_along_axis_(
            axis=1,
            indices=indices,
            values=(sorted_prob >= scaled_p).astype(dtype="int32"),
            reduce="add",
        )
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0
        pipe_compiled = LogitsPipe([MinP(), Sample()], compile=True)
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32"), min_p=p
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe([MinP(), Sample()], compile=False)
        counter_no_compile = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_no_compile = pipe_no_compile(
            probs, indices=paddle.zeros(shape=num_trials, dtype="int32"), min_p=p
        )
        counter_no_compile.put_along_axis_(
            axis=0,
            indices=samples_no_compile.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_no_compile),
            reduce="add",
        )
        freq_no_compile = counter_no_compile.astype(dtype="float32") / num_trials
        assert paddle.all(x=mask[paddle.arange(end=1), samples_compiled] == 1)
        assert paddle.all(x=mask[paddle.arange(end=1), samples_no_compile] == 1)
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=masked_probs
        )
        similarity_no_compile = paddle.nn.functional.cosine_similarity(
            x1=freq_no_compile, x2=masked_probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"
        assert (
            similarity_no_compile > 0.99
        ), f"Non-compiled similarity: {similarity_no_compile}"
        freq_similarity = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=freq_no_compile, axis=0
        )
        assert (
            freq_similarity > 0.99
        ), f"Compiled vs non-compiled similarity: {freq_similarity}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_probs_top_k_top_p_joint_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        eps = 0.0001
        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")
        logits = distribution((1, vocab_size), "cuda:0")
        probs = paddle.nn.functional.softmax(x=logits, axis=-1)
        sorted_prob_asc, idx_asc = paddle.sort(
            x=probs, descending=False
        ), paddle.argsort(x=probs, descending=False)
        cdf = paddle.cumsum(x=sorted_prob_asc, axis=-1)
        mask_top_p = paddle.zeros(shape=[1, vocab_size], dtype="int32")
        mask_top_p.put_along_axis_(
            axis=1,
            indices=idx_asc,
            values=(cdf > 1 - p - eps).astype(dtype="int32"),
            reduce="add",
        )
        sorted_prob_desc, _ = paddle.sort(x=probs, descending=True), paddle.argsort(
            x=probs, descending=True
        )
        pivot = sorted_prob_desc[:, k - 1]
        mask_top_k = (probs >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
        mask = paddle.minimum(x=mask_top_k, y=mask_top_p)
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0
        pipe_compiled = LogitsPipe(
            [TopK(joint_topk_topp=True), TopP(), Sample()],
            compile=True,
            input_type=TensorType.PROBS,
        )
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            probs,
            indices=paddle.zeros(shape=num_trials, dtype="int32"),
            top_k=k,
            top_p=p,
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe(
            [TopK(), TopP(), Sample()], compile=False, input_type=TensorType.PROBS
        )
        samples_no_compile = pipe_no_compile(
            probs,
            indices=paddle.zeros(shape=num_trials, dtype="int32"),
            top_k=k,
            top_p=p,
        )
        assert paddle.all(x=mask[paddle.arange(end=1), samples_compiled] == 1)
        assert paddle.all(x=mask[paddle.arange(end=1), samples_no_compile] == 1)
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=masked_probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"

    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize(
        "distribution",
        [normal_distribution(1), normal_distribution(5), gumbel_distribution(0.1)],
    )
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_logits_top_k_top_p_joint_sample_freq(self, vocab_size, distribution, p):
        set_random_seed(42)
        num_trials = 5000000
        eps = 0.0001
        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")
        logits = distribution((1, vocab_size), "cuda:0")
        probs = paddle.nn.functional.softmax(x=logits, axis=-1)
        sorted_prob_asc, idx_asc = paddle.sort(
            x=probs, descending=False
        ), paddle.argsort(x=probs, descending=False)
        cdf = paddle.cumsum(x=sorted_prob_asc, axis=-1)
        mask_top_p = paddle.zeros(shape=[1, vocab_size], dtype="int32")
        mask_top_p.put_along_axis_(
            axis=1,
            indices=idx_asc,
            values=(cdf > 1 - p - eps).astype(dtype="int32"),
            reduce="add",
        )
        sorted_prob_desc, _ = paddle.sort(x=probs, descending=True), paddle.argsort(
            x=probs, descending=True
        )
        pivot = sorted_prob_desc[:, k - 1]
        mask_top_k = (probs >= pivot.unsqueeze(axis=-1)).astype(dtype="int32")
        mask = paddle.minimum(x=mask_top_k, y=mask_top_p)
        masked_probs = probs.clone()
        masked_probs[mask == 0] = 0
        pipe_compiled = LogitsPipe(
            [Softmax(), TopK(joint_topk_topp=True), TopP(), Sample()],
            compile=True,
            input_type=TensorType.LOGITS,
        )
        counter_compiled = paddle.zeros(shape=vocab_size, dtype="int32")
        samples_compiled = pipe_compiled(
            logits,
            indices=paddle.zeros(shape=num_trials, dtype="int32"),
            top_k=k,
            top_p=p,
        )
        counter_compiled.put_along_axis_(
            axis=0,
            indices=samples_compiled.astype(dtype="int64"),
            values=paddle.ones_like(x=samples_compiled),
            reduce="add",
        )
        freq_compiled = counter_compiled.astype(dtype="float32") / num_trials
        pipe_no_compile = LogitsPipe(
            [Softmax(), TopK(), TopP(), Sample()],
            compile=False,
            input_type=TensorType.LOGITS,
        )
        samples_no_compile = pipe_no_compile(
            logits,
            indices=paddle.zeros(shape=num_trials, dtype="int32"),
            top_k=k,
            top_p=p,
        )
        assert paddle.all(x=mask[paddle.arange(end=1), samples_compiled] == 1)
        assert paddle.all(x=mask[paddle.arange(end=1), samples_no_compile] == 1)
        similarity_compiled = paddle.nn.functional.cosine_similarity(
            x1=freq_compiled, x2=masked_probs
        )
        assert similarity_compiled > 0.99, f"Compiled similarity: {similarity_compiled}"


class TestLogitsPipeVsSamplingOps:
    """Test LogitsPipe implementations against direct sampling operations"""

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("temperature", [1.0, 0.5, 0.1])
    @pytest.mark.parametrize("temperature_arr", [True, False])
    def test_temperature_softmax(
        self, batch_size, vocab_size, temperature, temperature_arr
    ):
        set_random_seed(42)
        logits = paddle.randn(shape=[batch_size, vocab_size])
        if temperature_arr:
            temperature = paddle.rand(shape=batch_size)
        samples_direct = flashinfer.sampling.softmax(
            logits=logits, temperature=temperature
        )
        pipe = LogitsPipe([Temperature(), Softmax()])
        samples_pipe = pipe(logits, temperature=temperature)
        assert paddle.allclose(x=samples_pipe, y=samples_direct, atol=1e-05).item()

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_topp(self, batch_size, vocab_size, p):
        set_random_seed(42)
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        samples_direct = flashinfer.sampling.top_p_renorm_probs(probs, p)
        pipe = LogitsPipe([TopP()])
        samples_pipe = pipe(probs, top_p=p)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("k", [10, 100, 500])
    def test_probs_topk(self, batch_size, vocab_size, k):
        set_random_seed(42)
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        samples_direct = flashinfer.sampling.top_k_renorm_probs(probs, k)
        pipe = LogitsPipe([TopK()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, top_k=k)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("k", [10, 100, 500])
    @pytest.mark.parametrize("neginf_input", [False, True])
    def test_logits_topk(self, batch_size, vocab_size, k, neginf_input):
        if k > vocab_size:
            pytest.skip("k should be less than vocab_size")
        set_random_seed(42)
        logits = paddle.randn(shape=[batch_size, vocab_size])
        if neginf_input:
            num_neginf = paddle.randint(
                low=1, high=vocab_size * batch_size, shape=(1,)
            ).item()
            idxs = paddle.randperm(n=batch_size * vocab_size)[:num_neginf]
            logits[idxs // vocab_size, idxs % vocab_size] = -float("inf")
        samples_direct = flashinfer.sampling.top_k_mask_logits(logits, k)
        pipe = LogitsPipe([TopK()], input_type=TensorType.LOGITS)
        samples_pipe = pipe(logits, top_k=k)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    def test_probs_sample(self, batch_size, vocab_size):
        set_random_seed(42)
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.sampling_from_probs(probs, generator=gen1)
        pipe = LogitsPipe([Sample()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    def test_logits_sample(self, batch_size, vocab_size):
        set_random_seed(42)
        logits = paddle.randn(shape=[batch_size, vocab_size])
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.sampling_from_logits(
            logits, generator=gen1
        )
        pipe = LogitsPipe([Sample()], input_type=TensorType.LOGITS)
        samples_pipe = pipe(logits, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("k", [10, 100, 500])
    def test_probs_topk_sample(self, batch_size, vocab_size, k):
        if k > vocab_size:
            pytest.skip("k should be less than vocab_size")
        set_random_seed(42)
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, generator=gen1
        )
        pipe = LogitsPipe([TopK(), Sample()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, top_k=k, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_probs_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, generator=gen1
        )
        pipe = LogitsPipe([TopP(), Sample()])
        samples_pipe = pipe(probs, top_p=p, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
    def test_probs_minp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.min_p_sampling_from_probs(
            probs, p, generator=gen1
        )
        pipe = LogitsPipe([MinP(), Sample()])
        samples_pipe = pipe(probs, min_p=p, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_joint_probs_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)
        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, p, filter_apply_order="joint", generator=gen1
        )
        pipe = LogitsPipe(
            [TopK(joint_topk_topp=True), TopP(), Sample()], input_type=TensorType.PROBS
        )
        samples_pipe = pipe(probs, top_k=k, top_p=p, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_sequential_probs_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)
        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")
        probs = paddle.rand(shape=[batch_size, vocab_size])
        probs = probs / probs.sum(axis=-1, keepdim=True)
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, k, p, filter_apply_order="top_k_first", generator=gen1
        )
        pipe = LogitsPipe([TopK(), TopP(), Sample()], input_type=TensorType.PROBS)
        samples_pipe = pipe(probs, top_k=k, top_p=p, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_joint_logits_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)
        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")
        logits = paddle.randn(shape=[batch_size, vocab_size])
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, filter_apply_order="joint", generator=gen1
        )
        pipe = LogitsPipe(
            [Softmax(), TopK(joint_topk_topp=True), TopP(), Sample()],
            input_type=TensorType.LOGITS,
        )
        samples_pipe = pipe(logits, top_k=k, top_p=p, generator=gen2)
        assert paddle.all(x=samples_pipe == samples_direct)

    @pytest.mark.parametrize("batch_size", [1, 99, 989])
    @pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
    @pytest.mark.parametrize("p", [0.1, 0.5])
    def test_sequential_logits_topk_topp_sample(self, batch_size, vocab_size, p):
        set_random_seed(42)
        if p == 0.1:
            k = int(vocab_size * 0.5)
        elif p == 0.5:
            k = int(vocab_size * 0.1)
        else:
            raise ValueError("p not recognized")
        logits = paddle.randn(shape=[batch_size, vocab_size])
        gen1, gen2 = get_generators()
        samples_direct = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, filter_apply_order="top_k_first", generator=gen1
        )
        topk_mask_pipe = LogitsPipe([TopK()], input_type=TensorType.LOGITS)
        topp_pipe = LogitsPipe([Softmax(), TopP(), Sample()])
        samples_pipe = topp_pipe(
            topk_mask_pipe(logits, top_k=k), top_p=p, generator=gen2
        )
        assert paddle.all(x=samples_pipe == samples_direct)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
