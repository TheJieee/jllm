#pragma once

#include "sequence.hpp"
#include "kv_cache_manager/block_manager/block_manager.hpp"

#include <queue>
#include <mutex>

JLLM_BEGIN

class Scheduler {
public:
    Scheduler(size_t num_blocks, size_t block_size, size_t max_seq_len = 64);
    void push(const Sequence& seq);

    //Remove finished seqs from running queue, free thire blocks, and allocate blocks
    //to running seqs if if they need.
    void update();

    /** Pick seqs from running queue. If having rest, pick seqs from waiting queue
        and allocate blocks.
    */
    std::vector<Sequence*> schedule();
    bool is_free();
private:
private:
    std::queue<Sequence>        m_waiting;
    std::map<size_t, Sequence>  m_running;
    BlockManager                m_manager;
    size_t m_max_seq_len;
    size_t m_chunk_size;
    std::mutex m_mutex;
};

JLLM_END