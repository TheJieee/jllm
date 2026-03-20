#pragma once
#include "jllm.h"

#include "vector"

JLLM_BEGIN

struct KVCacheView {
    std::byte* k_cache;
    std::byte* v_cache;
    std::vector<size_t> shape; //[num_blocks, nkvhead, block_size, head_dim]
};

JLLM_END