import paddle

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
import flashinfer


def test_mlc_failed_case():
    kv_layout = "HND"
    kv_indptr_1 = paddle.to_tensor(data=[0, 0, 9]).astype(dtype="int32").to(0)
    kv_indices_1 = (
        paddle.to_tensor(data=[3, 4, 5, 6, 7, 8, 9, 10, 11]).astype(dtype="int32").to(0)
    )
    kv_last_page_len_1 = paddle.to_tensor(data=[0, 1]).astype(dtype="int32").to(0)
    num_qo_heads = 32
    num_kv_heads = 32
    page_size = 16
    head_dim = 128
    q = paddle.randn(shape=[2, num_qo_heads, head_dim]).to(0).astype(dtype="float16")
    kv_data = (
        paddle.randn(shape=[12, 2, num_kv_heads, page_size, head_dim])
        .to(0)
        .astype(dtype="float16")
    )
    workspace_buffer = paddle.empty(shape=128 * 1024 * 1024, dtype="int8").to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type="float16",
        q_data_type="float16",
    )
    o_1, lse_1 = wrapper.run_return_lse(q, kv_data)
    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.plan(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type="float16",
        q_data_type="float16",
    )
    o_1_tc, lse_1_tc = wrapper_tensor_cores.run_return_lse(q, kv_data)
    print(lse_1, lse_1_tc)
    print(o_1, o_1_tc)
    assert paddle.allclose(x=lse_1, y=lse_1_tc, rtol=0.001, atol=0.001).item(), ""
    assert paddle.allclose(x=o_1, y=o_1_tc, rtol=0.001, atol=0.001).item(), ""


if __name__ == "__main__":
    test_mlc_failed_case()
