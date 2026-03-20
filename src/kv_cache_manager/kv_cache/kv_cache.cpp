#include "kv_cache.hpp"

#include "../../utils.hpp"
#include "log/log.hpp"

JLLM_BEGIN

KVCache::KVCache(
    size_t nlayer, size_t nkvhead, size_t head_dim, jllmDataType_t dtype,
    size_t voc_size,
    size_t block_size, size_t num_blocks
) :m_block_size(block_size), m_num_block(num_blocks) {
    spdlog::info("Initializing KVCache...\n");
    using namespace utils;
    if(num_blocks == 0) {
        size_t available_memory = sysUtils::get_available_memory();
        spdlog::info("system available memory: {}MB.\n", available_memory / (1024.0 * 1024.0));

        size_t block_memory = nlayer * block_size * head_dim * nkvhead * 2 * dsize(dtype);
        m_num_block = num_blocks = (size_t)(available_memory * 0.6 / block_memory); 
    }

    m_logits = Tensor::create({num_blocks, voc_size}, dtype);
    m_kcache.resize(nlayer);
    m_vcache.resize(nlayer);
    for(size_t i = 0; i < nlayer; i++) {
        m_kcache[i] = Tensor::create({num_blocks, nkvhead, block_size, head_dim}, dtype);
        m_vcache[i] = Tensor::create({num_blocks, nkvhead, block_size, head_dim}, dtype);
    }

    size_t used_memory = num_blocks * nlayer * block_size * head_dim * nkvhead * 2 * dsize(dtype);
    spdlog::info(
        "KVCache initialized with {} blocks, using approximately {}MB of memory.\n"
              , num_blocks, used_memory / (1024.0 * 1024.0)
    );
}

tensor_t KVCache::get_logits(size_t block_idx)
{
    auto logit = m_logits->slice(0, block_idx, block_idx + 1)->view({m_logits->shape()[1]});
    return logit;
}

size_t KVCache::block_size() const
{
    return m_block_size;
}

size_t KVCache::get_num_block() const
{
    return m_num_block;
}

JLLM_END