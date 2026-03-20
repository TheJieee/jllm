#include "embedding.hpp"

#include "utils.hpp"
#include <omp.h>

template <typename T>
static void embedding_(
    T* out,
    const int64_t* index,
    const T* weight,
    std::size_t embedding_dim,
    std::size_t index_size
) {
    for (size_t i = 0; i < index_size; ++i) {
        std::memcpy(
            out + i * embedding_dim,
            weight + index[i] * embedding_dim,
            embedding_dim * sizeof(T)
        );
    }
}

namespace jllm::ops::cpu {
    void embedding(
        void* out,
        const void* index,
        const void* weight,
        size_t index_size,
        size_t embedding_dim,
        jllmDataType_t data_type
    ) {
        switch (data_type) {
            case jllmDataType_t::F32:
                embedding_(
                    reinterpret_cast<float*>(out),
                    reinterpret_cast<const int64_t*>(index),
                    reinterpret_cast<const float*>(weight),
                    embedding_dim,
                    index_size
                );
                break;
            case jllmDataType_t::BF16:
                embedding_(
                    reinterpret_cast<jllm::bf16_t*>(out),
                    static_cast<const int64_t*>(index),
                    reinterpret_cast<const jllm::bf16_t*>(weight),
                    embedding_dim,
                    index_size
                );
                break;
            case jllmDataType_t::F16:
                embedding_(
                    reinterpret_cast<jllm::fp16_t*>(out),
                    static_cast<const int64_t*>(index),
                    reinterpret_cast<const jllm::fp16_t*>(weight),
                    embedding_dim,
                    index_size
                );
                break;
            default:
                throw std::runtime_error("Unsupported data type in embedding operation.");
        }
    }
} // namespace llaisys::ops::cpu