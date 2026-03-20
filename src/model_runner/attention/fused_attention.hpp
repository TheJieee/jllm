#pragma once
#include "jllm.h"

#include "../../tensor/tensor.hpp"
#include "../kv_cache_view.hpp"
#include "../../executor/input_meta.hpp"

JLLM_BEGIN
namespace ops {
void fused_attention(
    tensor_t out, tensor_t q,
    const KVCacheView& view, 
    const InputMeta& meta
);
}
JLLM_END