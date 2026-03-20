#pragma once
#include "jllm.h"

namespace jllm::ops::cpu {
    void add_norm(
        std::byte* out,
        std::byte* in,
        const std::byte* add_tensor,
        const std::byte* norm_weight,
        size_t batch_size,
        size_t in_features,
        float eps,
        jllmDataType_t data_type
    );
}