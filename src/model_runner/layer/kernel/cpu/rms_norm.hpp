#pragma once
#include "jllm.h"

namespace jllm::ops::cpu {
    void rms_norm(
        std::byte *output,
        const std::byte *input,
        const std::byte *weight,
        jllmDataType_t type,
        size_t batch_size,
        size_t feature_size,
        float epsilon
    );
    } // namespace jllm::ops::cpu
