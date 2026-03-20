#pragma once
#include "jllm.h"

#include "../tensor/tensor.hpp"

JLLM_BEGIN
namespace ops {
    void sampling(
        tensor_t out, tensor_t logits,
        float temperature, int topk, float topp
    );
}
JLLM_END