#pragma once
#include "jllm.h"

#include "../../tensor/tensor.hpp"
#include <vector>

JLLM_BEGIN

class KVCache {
public:
    KVCache(
        size_t nlayer, size_t nkvhead, size_t head_dim, jllmDataType_t dtype,
        size_t voc_size,
        size_t block_size, size_t num_blocks = 0
    );
    ~KVCache() = default;

    tensor_t get_kblock(size_t layer) {
        return m_kcache[layer];
    }

    tensor_t get_vblock(size_t layer) {
        return m_vcache[layer];
    }

    tensor_t get_logits(size_t block_idx);
    size_t block_size() const;
    size_t get_num_block() const;
    
private:
    std::vector<tensor_t> m_kcache;
    std::vector<tensor_t> m_vcache;
    tensor_t m_logits; //shape[num_block, voc_size]
    size_t m_block_size;
    size_t m_num_block;
};

JLLM_END