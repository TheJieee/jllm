#pragma once
#include "jllm.h"

namespace jllm::ops::cpu {
void gate_up_swiglu(
    std::byte* out,
    const std::byte* in,
    const std::byte* gate_weight,
    const std::byte* up_weight,
    const size_t batch_size,
    const size_t in_features,
    const size_t out_features,
    jllmDataType_t data_type
);
}