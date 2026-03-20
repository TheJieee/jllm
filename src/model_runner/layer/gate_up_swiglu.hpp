#pragma once

#include "../tensor/tensor.hpp"

namespace jllm::ops {
    void gate_up_swiglu(
        tensor_t out,
        const tensor_t in,
        const tensor_t gate_weight,
        const tensor_t up_weight
    );
}