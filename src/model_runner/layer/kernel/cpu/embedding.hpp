#pragma once
#include "jllm.h"

namespace jllm::ops::cpu {
    void embedding(
        void* out,
        const void* index,
        const void* weight,
        size_t index_size,
        size_t embedding_dim,
        jllmDataType_t data_type
    );
}