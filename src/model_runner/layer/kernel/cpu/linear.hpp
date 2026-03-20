#pragma once
#include "jllm.h"

#include <cstddef>

namespace jllm::ops::cpu {
void linear(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight,
    const std::byte* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    jllmDataType_t data_type
);
}