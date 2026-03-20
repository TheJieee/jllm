#pragma once 
#include "jllm.h"

#include <vector>

namespace jllm::ops::cpu {
    void prefill(
        std::byte* out, std::byte* q,
        std::byte* kcache, std::byte* vcache,
        size_t nhead, size_t nkvhead, size_t head_dim,
        size_t seq_begin, size_t seq_end,
        const std::vector<size_t>& block_table, size_t block_size,
        jllmDataType_t dtype
    );
}