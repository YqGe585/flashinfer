import sys

sys.path.append("/home/flashinfer_paddle")
from typing import Optional, Tuple

import paddle
from paddle_utils import *

import flashinfer


def wmape(target: paddle.Tensor, preds: paddle.Tensor):
    sum_abs_error = (preds - target).abs().sum().detach().item()
    sum_scale = target.abs().sum().detach().item()
    return sum_abs_error / sum_scale


from rope_reference import *


class DeepseekV2RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=hidden_size)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class DeepseekV2AttentionVanilla(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 5120
        self.num_heads = 128
        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.q_head_dim = 192
        self.rope_theta = 10000
        self.q_a_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.q_lora_rank, bias_attr=False
        )
        init_Normal = paddle.nn.initializer.Normal()
        init_Normal(self.q_a_proj.weight)
        self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)
        self.q_b_proj = paddle.nn.Linear(
            in_features=self.q_lora_rank,
            out_features=self.num_heads * self.q_head_dim,
            bias_attr=False,
        )
        init_Normal = paddle.nn.initializer.Normal()
        init_Normal(self.q_b_proj.weight)
        self.kv_b_proj = paddle.nn.Linear(
            in_features=self.kv_lora_rank,
            out_features=self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias_attr=False,
        )
        init_Normal = paddle.nn.initializer.Normal()
        init_Normal(self.kv_b_proj.weight)
        self.o_proj = paddle.nn.Linear(
            in_features=self.num_heads * self.v_head_dim,
            out_features=self.hidden_size,
            bias_attr=False,
        )
        init_Normal = paddle.nn.initializer.Normal()
        init_Normal(self.o_proj.weight)
        self.softmax_scale = self.q_head_dim**-0.5

    def run_decode(
        self,
        hidden_states: paddle.Tensor,
        compressed_kv_normed_cache: paddle.Tensor,
        k_pe_cache: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = tuple(hidden_states.shape)
        if q_len != 1:
            raise ValueError(
                f"Only support decode, but got hidden_states[{tuple(hidden_states.shape)}]"
            )
        ckv_bsz, kv_len, ckv_dim = tuple(compressed_kv_normed_cache.shape)
        if ckv_bsz != bsz or ckv_dim != self.kv_lora_rank:
            raise ValueError(
                f"Unexpected shape: compressed_kv_normed_cache[{tuple(compressed_kv_normed_cache.shape)}]"
            )
        kpe_bsz, kpe_len, kpe_dim = tuple(k_pe_cache.shape)
        if kpe_bsz != bsz or kpe_dim != self.qk_rope_head_dim or kv_len != kpe_len:
            raise ValueError(f"Unexpected shape: k_pe_cache[{tuple(k_pe_cache.shape)}]")
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(
            perm=dim2perm(
                q.view(bsz, q_len, self.num_heads, self.q_head_dim).ndim, 1, 2
            )
        )
        q_nope, q_pe = paddle_split(
            x=q, num_or_sections=[self.qk_nope_head_dim, self.qk_rope_head_dim], axis=-1
        )
        k_pe = k_pe_cache.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(
            perm=dim2perm(
                k_pe_cache.view(bsz, kv_len, 1, self.qk_rope_head_dim).ndim, 1, 2
            )
        )
        kv = (
            self.kv_b_proj(compressed_kv_normed_cache)
            .view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(
                perm=dim2perm(
                    self.kv_b_proj(compressed_kv_normed_cache)
                    .view(
                        bsz,
                        kv_len,
                        self.num_heads,
                        self.qk_nope_head_dim + self.v_head_dim,
                    )
                    .ndim,
                    1,
                    2,
                )
            )
        )
        k_nope, value_states = paddle_split(
            x=kv, num_or_sections=[self.qk_nope_head_dim, self.v_head_dim], axis=-1
        )
        if tuple(k_nope.shape) != (bsz, self.num_heads, kv_len, self.qk_nope_head_dim):
            raise ValueError(f"k_nope[{tuple(k_nope.shape)}]")
        if tuple(value_states.shape) != (bsz, self.num_heads, kv_len, self.v_head_dim):
            raise ValueError(f"value_states[{tuple(value_states.shape)}]")
        freqs_cis = precompute_freqs_cis(
            self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
        ).to(q_pe.place)
        q_pe, k_pe = apply_rotary_emb(
            q_pe.transpose(perm=dim2perm(q_pe.ndim, 1, 2)).tile(
                repeat_times=[1, kv_len, 1, 1]
            ),
            k_pe.transpose(perm=dim2perm(k_pe.ndim, 1, 2)),
            freqs_cis,
        )
        q_pe = q_pe[:, -1:, :, :].transpose(
            perm=dim2perm(q_pe[:, -1:, :, :].ndim, 1, 2)
        )
        k_pe = k_pe.transpose(perm=dim2perm(k_pe.ndim, 1, 2))
        query_states = paddle.empty(
            shape=[bsz, self.num_heads, q_len, self.q_head_dim], dtype=q.dtype
        )
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        key_states = paddle.empty(
            shape=[bsz, self.num_heads, kv_len, self.q_head_dim], dtype=k_pe.dtype
        )
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        attn_weights = (
            paddle.matmul(
                x=query_states,
                y=key_states.transpose(perm=dim2perm(key_states.ndim, 2, 3)),
            )
            * self.softmax_scale
        )
        attn_weights = paddle.nn.functional.softmax(
            x=attn_weights, axis=-1, dtype="float32"
        ).to(query_states.dtype)
        attn_output = paddle.matmul(x=attn_weights, y=value_states)
        attn_output = attn_output.transpose(
            perm=dim2perm(attn_output.ndim, 1, 2)
        ).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        output = self.o_proj(attn_output)
        return output


class DeepseekV2AttentionMatAbsorbDecode(paddle.nn.Layer):
    def __init__(self, mla_vanilla: DeepseekV2AttentionVanilla):
        super().__init__()
        self.hidden_size = mla_vanilla.hidden_size
        self.num_heads = mla_vanilla.num_heads
        self.q_lora_rank = mla_vanilla.q_lora_rank
        self.qk_rope_head_dim = mla_vanilla.qk_rope_head_dim
        self.kv_lora_rank = mla_vanilla.kv_lora_rank
        self.v_head_dim = mla_vanilla.v_head_dim
        self.qk_nope_head_dim = mla_vanilla.qk_nope_head_dim
        self.q_head_dim = mla_vanilla.q_head_dim
        self.softmax_scale = mla_vanilla.softmax_scale
        self.rope_theta = mla_vanilla.rope_theta
        self.W_DQ = mla_vanilla.q_a_proj.weight.transpose(
            perm=dim2perm(mla_vanilla.q_a_proj.weight.ndim, 0, 1)
        )
        self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)
        W_UQ, W_QR = paddle_split(
            x=mla_vanilla.q_b_proj.weight.t().view(
                self.q_lora_rank, self.num_heads, self.q_head_dim
            ),
            num_or_sections=[self.qk_nope_head_dim, self.qk_rope_head_dim],
            axis=-1,
        )
        self.W_QR = W_QR.reshape(
            self.q_lora_rank, self.num_heads * self.qk_rope_head_dim
        )
        W_UK, W_UV = paddle_split(
            x=mla_vanilla.kv_b_proj.weight.t().view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ),
            num_or_sections=[self.qk_nope_head_dim, self.v_head_dim],
            axis=-1,
        )
        self.W_UQ_UK = paddle.einsum("q n d, l n d -> q n l", W_UQ, W_UK).flatten(
            start_axis=1
        )
        W_O = mla_vanilla.o_proj.weight.view(
            self.hidden_size, self.num_heads, self.v_head_dim
        )
        self.W_UV_O = paddle.einsum("l n d, h n d -> n l h", W_UV, W_O).flatten(
            start_axis=0, stop_axis=1
        )

    def run_proof_of_concept(
        self,
        hidden_states: paddle.Tensor,
        compressed_kv_normed_cache: paddle.Tensor,
        k_pe_cache: paddle.Tensor,
        use_flashinfer_kernel: bool,
        convert_float16: bool,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        c_Q = paddle.matmul(x=hidden_states, y=self.W_DQ)
        c_Q = self.q_a_layernorm(c_Q)
        q_pe = paddle.matmul(x=c_Q, y=self.W_QR)
        q_pe = q_pe.reshape(bsz, self.num_heads, self.qk_rope_head_dim)
        q_nope = paddle.matmul(x=c_Q, y=self.W_UQ_UK)
        q_nope = q_nope.reshape(bsz, self.num_heads, self.kv_lora_rank)
        q_kv_dtype = "float16"
        if convert_float16:
            q_nope = q_nope.to(q_kv_dtype)
            q_pe = q_pe.to(q_kv_dtype)
            compressed_kv_normed_cache = compressed_kv_normed_cache.to(q_kv_dtype)
            k_pe_cache = k_pe_cache.to(q_kv_dtype)
        if not use_flashinfer_kernel:
            freqs_cis = precompute_freqs_cis(
                self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
            ).to(k_pe_cache.place)
            q_pe, k_pe_cache = apply_rotary_emb(
                q_pe.unsqueeze(axis=1).tile(repeat_times=[1, kv_len, 1, 1]),
                k_pe_cache.unsqueeze(axis=2),
                freqs_cis,
            )
            q_pe = q_pe[:, -1:, :, :].squeeze(axis=1)
            k_pe_cache = k_pe_cache.squeeze(axis=2)
            attn_weights_pe = paddle.matmul(
                x=q_pe, y=k_pe_cache.transpose(perm=dim2perm(k_pe_cache.ndim, 1, 2))
            )
            attn_weights_nope = paddle.matmul(
                x=q_nope,
                y=compressed_kv_normed_cache.transpose(
                    perm=dim2perm(compressed_kv_normed_cache.ndim, 1, 2)
                ),
            )
            attn_weights = (attn_weights_pe + attn_weights_nope) * self.softmax_scale
            attn_weights = paddle.nn.functional.softmax(
                x=attn_weights, axis=-1, dtype="float32"
            ).to(q_nope.dtype)
            attn_output = paddle.matmul(x=attn_weights, y=compressed_kv_normed_cache)
        else:
            print("Now use MLA decode kernel!\n")
            if kv_len % page_size != 0:
                raise ValueError(
                    "For simplicity, kv_len should be multiple of page_size."
                )
            freqs_cis = precompute_freqs_cis(
                self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
            ).to(k_pe_cache.place)
            q_pe, k_pe_cache = apply_rotary_emb(
                q_pe.unsqueeze(axis=1).tile(repeat_times=[1, kv_len, 1, 1]),
                k_pe_cache.unsqueeze(axis=2),
                freqs_cis,
            )
            q_pe = q_pe[:, -1:, :, :].squeeze(axis=1).contiguous()
            k_pe_cache = k_pe_cache.squeeze(axis=2)
            num_pages_per_seq = kv_len // page_size
            total_num_pages = num_pages_per_seq * bsz
            kv_indptr = (
                paddle.arange(start=0, end=bsz + 1).to(dev_id).astype(dtype="int32")
                * num_pages_per_seq
            )
            kv_indices = (
                paddle.arange(start=0, end=total_num_pages)
                .to(dev_id)
                .astype(dtype="int32")
            )
            kv_last_page_len = paddle.full(
                shape=(bsz,), fill_value=page_size, dtype="int32"
            ).to(dev_id)
            paged_ckv_cache = compressed_kv_normed_cache.reshape(
                total_num_pages, page_size, self.kv_lora_rank
            )
            paged_kpe_cache = k_pe_cache.reshape(
                total_num_pages, page_size, self.qk_rope_head_dim
            )
            workspace_buffer = paddle.empty(shape=64 * 1024 * 1024, dtype="int8").to(
                dev_id
            )
            wrapper = flashinfer.BatchDecodeMlaWithPagedKVCacheWrapper(
                workspace_buffer,
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=kv_indptr,
                paged_kv_indices_buffer=kv_indices,
                paged_kv_last_page_len_buffer=kv_last_page_len,
            )
            wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads=self.num_heads,
                head_dim_compressed_kv=self.kv_lora_rank,
                page_size=page_size,
                sm_scale=self.softmax_scale,
                rope_theta=self.rope_theta,
                data_type=q_kv_dtype,
                q_data_type=q_kv_dtype,
            )
            attn_output = wrapper.run(q_nope, q_pe, paged_ckv_cache, paged_kpe_cache)
            s = paddle.device.Stream()
            s.wait_stream(paddle.device.current_stream())
            with paddle.device.stream_guard(stream=s):
                for _ in range(3):
                    o, lse = wrapper.run(
                        q_nope, q_pe, paged_ckv_cache, paged_kpe_cache, return_lse=True
                    )
            paddle.device.current_stream().wait_stream(s)
>>>>>>            g = torch.cuda.CUDAGraph()
>>>>>>            with torch.cuda.graph(g):
                attn_output = wrapper.run(
                    q_nope, q_pe, paged_ckv_cache, paged_kpe_cache
                )
            g.replay()
        output = paddle.matmul(
            x=attn_output.to(self.W_UV_O.dtype).reshape(
                bsz, self.num_heads * self.kv_lora_rank
            ),
            y=self.W_UV_O,
        )
        return output


if __name__ == "__main__":
    dev_id = 0
    paddle.seed(seed=666)
    paddle.set_grad_enabled(mode=False)
    mla_vanilla = DeepseekV2AttentionVanilla().cuda(device_id=device2int(dev_id))
    bsz = 6
    kv_len = 640
    page_size = 16
    hidden_states = paddle.randn(shape=[bsz, 1, mla_vanilla.hidden_size]).to(dev_id)
    compressed_kv_normed_cache = paddle.randn(
        shape=[bsz, kv_len, mla_vanilla.kv_lora_rank]
    ).to(dev_id)
    k_pe_cache = paddle.randn(shape=[bsz, kv_len, mla_vanilla.qk_rope_head_dim]).to(
        dev_id
    )
    output_vanilla = mla_vanilla.run_decode(
        hidden_states, compressed_kv_normed_cache, k_pe_cache
    )
    mla_mat_absorb = DeepseekV2AttentionMatAbsorbDecode(mla_vanilla).cuda(
        device_id=device2int(dev_id)
    )
    output_mat_absorbed_use_torch_f32 = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(axis=1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flashinfer_kernel=False,
        convert_float16=False,
    )
    output_mat_absorbed_use_torch_f16 = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(axis=1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flashinfer_kernel=False,
        convert_float16=True,
    )
    output_mat_absorbed_use_flashinfer = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(axis=1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flashinfer_kernel=True,
        convert_float16=True,
    )
    cos_use_torch_f32 = paddle.nn.functional.cosine_similarity(
        x1=output_vanilla.reshape(-1),
        x2=output_mat_absorbed_use_torch_f32.reshape(-1),
        axis=0,
    )
    print(f"cos_use_torch_f32 = {cos_use_torch_f32}")
    assert cos_use_torch_f32 > 0.99
    wmape_use_torch_f32 = wmape(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f32.reshape(-1)
    )
    print(f"wmape_use_torch_f32 = {wmape_use_torch_f32}")
    assert wmape_use_torch_f32 < 0.02
    mse_use_torch_f32 = paddle.nn.functional.mse_loss(
        input=output_vanilla.reshape(-1),
        label=output_mat_absorbed_use_torch_f32.reshape(-1),
    )
    print(f"mse_use_torch_f32={mse_use_torch_f32}\n")
    cos_use_torch_f16 = paddle.nn.functional.cosine_similarity(
        x1=output_vanilla.reshape(-1),
        x2=output_mat_absorbed_use_torch_f16.reshape(-1),
        axis=0,
    )
    print(f"cos_use_torch_f16 = {cos_use_torch_f16}")
    assert cos_use_torch_f16 > 0.99
    wmape_use_torch_f16 = wmape(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch_f16.reshape(-1)
    )
    print(f"wmape_use_torch_f16 = {wmape_use_torch_f16}")
    assert wmape_use_torch_f16 < 0.03
    mse_use_torch_f16 = paddle.nn.functional.mse_loss(
        input=output_vanilla.reshape(-1),
        label=output_mat_absorbed_use_torch_f16.reshape(-1),
    )
    print(f"mse_use_torch_f16 = {mse_use_torch_f16}\n")
    cos_use_flashinfer = paddle.nn.functional.cosine_similarity(
        x1=output_vanilla.reshape(-1),
        x2=output_mat_absorbed_use_flashinfer.reshape(-1),
        axis=0,
    )
    print(f"cos_use_flashinfer = {cos_use_flashinfer}")
    assert cos_use_flashinfer > 0.99
    wmape_use_flashinfer = wmape(
        output_vanilla.reshape(-1), output_mat_absorbed_use_flashinfer.reshape(-1)
    )
    print(f"wmape_use_flashinfer = {wmape_use_flashinfer}")
    assert wmape_use_flashinfer < 0.02
    mse_use_flashinfer = paddle.nn.functional.mse_loss(
        input=output_vanilla.reshape(-1),
        label=output_mat_absorbed_use_flashinfer.reshape(-1),
    )
    print(f"mse_use_flashinfer = {mse_use_flashinfer}")
