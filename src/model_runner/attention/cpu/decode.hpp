#pragma once
#include "jllm.h"

#include <vector>

namespace jllm::ops::cpu {
    void decode(
        std::byte* out, std::byte* q,
        std::byte* kcache, std::byte* vcache,
        size_t nhead, size_t nkvhead, size_t head_dim,
        size_t seq_len,
        const std::vector<size_t>& block_table, size_t block_size,
        jllmDataType_t dtype
    );
}