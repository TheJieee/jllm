#pragma once

#include "../../tensor/tensor.hpp"

namespace jllm::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
