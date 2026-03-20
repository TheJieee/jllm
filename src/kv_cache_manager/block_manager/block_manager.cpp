#include "block_manager.hpp"

JLLM_BEGIN

BlockManager::BlockManager(size_t num_blocks, size_t block_size) : m_block_size(block_size){
    m_free_blocks.resize(num_blocks);
    m_ref_counts.assign(num_blocks, 0);
    m_num_0ref = num_blocks;
    for(size_t i = 0; i < num_blocks; i++) {
        m_free_blocks[i] = i;
    }
}

size_t BlockManager::allocate_block()
{
    if(m_free_blocks.empty()) {
        for(int i = m_ref_counts.size() - 1; i >= 0; i--) {
            if(m_ref_counts[i] == 0) {
                free_block(i);
            }
        }
    }
    size_t block_index = m_free_blocks.back();
    m_free_blocks.pop_back();
    m_ref_counts[block_index] = 1;
    return block_index;
}

size_t BlockManager::allocate_block(size_t hash)
{
    auto it = m_shared_blocks.find(hash);
    size_t ret;
    if(it != m_shared_blocks.end()) {
        ret = it->second;
        m_ref_counts[ret]++;
    }
    else {
        ret = allocate_block();
    }
    return size_t();
}

void BlockManager::del_block(size_t block_index)
{
    m_ref_counts[block_index]--;
    if(m_ref_counts[block_index] <= 0) {
        m_num_0ref++;
    }
}

void BlockManager::free_block(size_t block_index) {
    m_free_blocks.push_back(block_index);
}

void BlockManager::share_block(size_t block_index, size_t hash) {
    m_shared_blocks[hash] = block_index;
}

bool BlockManager::try_reuse(size_t &out, size_t hash) {
    auto it = m_shared_blocks.find(hash);
    if(it != m_shared_blocks.end()) {
        out = it->second;
        m_ref_counts[out]++;
        if(m_ref_counts[out] == 1) {
            m_num_0ref--;
        }
        return true;
    }
    return false;
}

bool BlockManager::can_allocate(const Sequence &seq)
{
    size_t require = (seq.prompt_len() + m_block_size - 1) / m_block_size;
    for(size_t hash : seq.hash_table()) {
        if(m_shared_blocks.find(hash) != m_shared_blocks.end())
        require--;
    }
    return m_free_blocks.size() + m_num_0ref > require;
}

size_t BlockManager::block_size() const
{
    return m_block_size;
}

JLLM_END