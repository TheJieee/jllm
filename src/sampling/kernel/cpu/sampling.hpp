#pragma once
#include "jllm.h"

JLLM_BEGIN
namespace ops::cpu {
    void sampling(
        std::byte* out, std::byte* logits,
        float temperature, int topk, float topp,
        size_t batch_size, size_t hidden_size,
        jllmDataType_t dtype
    );
}

JLLM_END