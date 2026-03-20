#pragma once
#include "jllm.h"

#include "tensor/tensor.hpp"

namespace jllm::ops {
    void rope(tensor_t out, tensor_t in, tensor_t pos_ids, tensor_t rope_table);
}