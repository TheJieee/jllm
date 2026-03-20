#pragma once

#include "../tensor/tensor.hpp"

namespace jllm::ops {
    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
    void linear(tensor_t out, tensor_t in, tensor_t weight);
    
}