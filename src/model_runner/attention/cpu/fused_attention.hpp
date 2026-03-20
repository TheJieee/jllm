#pragma once
#include "jllm.h"

#include "../../../executor/input_meta.hpp"
#include "../../kv_cache_view.hpp"

namespace jllm::ops::cpu {
    void fused_attention(
        std::byte* out,
        std::byte* q, size_t nhead,
        const KVCacheView& cache_view,
        const InputMeta& meta,
        jllmDataType_t dtype
    );
}