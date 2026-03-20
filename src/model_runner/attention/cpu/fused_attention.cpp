#include "fused_attention.hpp"

#include <utils.hpp>

#include "decode.hpp"
#include "prefill.hpp"

#include <vector>
#include <future>

namespace jllm::ops::cpu {
    void fused_attention(
        std::byte* out,
        std::byte* q, size_t nhead,
        const KVCacheView& cache_view,
        const InputMeta& meta,
        jllmDataType_t dtype
    ) {
        size_t nkvhead = cache_view.shape[1];
        size_t block_size = cache_view.shape[2];
        size_t head_dim = cache_view.shape.back();
        size_t prefix = 0;
        size_t size = jllm::utils::dsize(dtype);
        size_t stride = nhead * head_dim * size;

        for (size_t i = 0; i < meta.block_tables.size(); i++) {
            size_t seq_len = meta.cu_seq_len[i + 1] - prefix;
            size_t pre_len = meta.context_len[i];
            auto block_table = meta.block_tables[i];
            if(seq_len == 1) {
                jllm::ops::cpu::decode(
                    out + prefix * stride, q + prefix * stride,
                    cache_view.k_cache, cache_view.v_cache,
                    nhead, nkvhead, head_dim, seq_len + pre_len, *block_table,
                    block_size, dtype
                );
            }
            else {
                jllm::ops::cpu::prefill(
                    out + prefix * stride, q + prefix * stride,
                    cache_view.k_cache, cache_view.v_cache,
                    nhead, nkvhead, head_dim, pre_len,
                    pre_len + seq_len, *block_table,
                    block_size, dtype
                );
            }
            prefix += seq_len;
        }
    }
}