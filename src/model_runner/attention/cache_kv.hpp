#pragma once
#include "jllm.h"

#include "../../tensor/tensor.hpp"
#include "../kv_cache_view.hpp"

JLLM_BEGIN
namespace ops {
void cache_kv(
    tensor_t k, tensor_t v,
    KVCacheView& view,
    const std::vector<size_t>& slot_map
);
}
JLLM_END