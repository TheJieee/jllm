#pragma once

#include "../tensor/tensor.hpp"

namespace jllm::ops {
    void add_norm(
        tensor_t out,
        tensor_t in,
        const tensor_t add_tensor,
        const tensor_t norm_weight,
        float eps
    );
}