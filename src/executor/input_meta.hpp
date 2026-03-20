#pragma once
#include "jllm.h"

#include <vector>
#include "tensor/tensor.hpp"

JLLM_BEGIN

struct InputMeta {
    size_t seq_len;
    std::vector<size_t> cu_seq_len;
    std::vector<size_t> slot_map;
    std::vector<size_t> context_len;
    std::vector<const std::vector<size_t>*> block_tables;
};

struct package {
    tensor_t in;
    std::vector<size_t> seq_ids;
    InputMeta meta;
    float temperature;
    int topk;
    float topp;
};

JLLM_END