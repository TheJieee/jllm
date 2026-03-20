#pragma once
#include "jllm.h"

namespace jllm::ops::cpu {
void linear_rope(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight,
    const std::byte* bias,
    const int64_t* pos_ids,
    const float* rope_table,
    size_t batch_size,
    size_t nhead,
    size_t in_features,
    size_t out_features,
    jllmDataType_t data_type
);
}