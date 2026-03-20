#pragma once
#include "jllm.h"

namespace jllm::ops::cpu {
    void rope(
        std::byte* out,
        const std::byte* in,
        const float* rope_table,
        const int64_t* pos_ids,
        size_t seq_len, size_t nhead, size_t head_dim,
        jllmDataType_t dtype
    );
}