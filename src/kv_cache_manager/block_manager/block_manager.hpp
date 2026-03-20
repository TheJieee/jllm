#pragma once
#include "jllm.h"

#include <vector>
#include <map>
#include "../../tensor/tensor.hpp"
#include "sequence.hpp"

JLLM_BEGIN

class BlockManager {
public:
    BlockManager(size_t num_blocks, size_t block_size);

    size_t allocate_block();
    size_t allocate_block(size_t hash);
    void del_block(size_t block_index);
    void share_block(size_t block_index, size_t hash);
    bool try_reuse(size_t& out, size_t hash);
    bool can_allocate(const Sequence& seq);
    size_t block_size() const;
    size_t num_free_blocks() const {
        return m_free_blocks.size();
    }

private:
    void free_block(size_t block_index);
private:
    std::vector<size_t> m_free_blocks;
    std::map<size_t, size_t> m_shared_blocks; // hash -> block index
    std::vector<int> m_ref_counts;
    size_t m_num_0ref;
    size_t m_block_size;
};

JLLM_END