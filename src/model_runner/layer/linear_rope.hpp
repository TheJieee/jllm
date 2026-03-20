#pragma once
#include "jllm.h"

#include "../tensor/tensor.hpp"

namespace jllm::ops {
    void linear_rope(
        tensor_t out,
        const tensor_t in,
        const tensor_t weight,
        const tensor_t bias,
        const tensor_t seq_pos,
        const tensor_t rope_table
    );
}