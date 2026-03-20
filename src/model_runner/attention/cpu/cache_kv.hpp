#pragma once
#include "jllm.h"

#include <vector>

JLLM_BEGIN
namespace ops::cpu {
void cache_kv(
    std::byte* k, std::byte* v,             //shape[seq_len, nkvhead, head_dim]
    std::byte* kcache, std::byte* vcache,   //shape[nkvhead, block_size, head_dim]
    const std::vector<size_t>& slot_map,
    size_t block_size,
    size_t nkvhead, size_t head_dim,
    jllmDataType_t dtype
);
}
JLLM_END