#include "scheduler.hpp"

#include <algorithm>

JLLM_BEGIN

Scheduler::Scheduler(size_t num_blocks, size_t block_size, size_t max_seq_len)
:m_manager(num_blocks, block_size), m_max_seq_len(max_seq_len), m_chunk_size(block_size) {}

void Scheduler::push(const Sequence &seq)
{

    std::lock_guard lock(m_mutex);
    m_waiting.push(seq);
}

void Scheduler::update()
{
    std::vector<size_t> finished_id;
    for(auto& pair : m_running) {
        auto& seq = pair.second;
        if(seq.is_finished()){
            finished_id.push_back(pair.first);
        }
        else if(seq.num_computed_tokens() >=
         seq.block_table().size() * m_manager.block_size()
        ){
            m_manager.share_block(seq.block_table().back(), seq.hash_table().back());
            seq.allocate_block(m_manager.allocate_block());
        }
    }
    for(size_t id : finished_id) {
        auto& seq = m_running[id];
        for(size_t block_num : seq.block_table()) {
            m_manager.del_block(block_num);
        }
        m_running.erase(id);
    }
}

std::vector<Sequence*> Scheduler::schedule()
{
    size_t seq_len = 0;
    std::vector<Sequence*> ret;
    //processing running queue first.
    for (auto& pair : m_running) {
        auto* pseq = &pair.second;
        ret.push_back(pseq);
        if(pseq->prefill_finished()) seq_len++;
        else{
            seq_len += std::min(m_chunk_size, pseq->remaining_prefill_tokens());
        }
    }
    //pricessing waiting queue.
    std::lock_guard lock(m_mutex);
    while (seq_len <= m_max_seq_len - m_chunk_size) {
        if(!m_waiting.empty()){
            auto& seq = m_waiting.front();
            if(m_manager.can_allocate(seq)){
                //allocate blocks.
                std::vector<size_t> blocks(seq.num_needed_blocks());
                int shared = 0;
                for (int i = 0; 
                    i < seq.hash_table().size() && m_manager.try_reuse(blocks[i], seq.hash_table()[i]); 
                    i++
                ) {
                    shared++;
                }
                for(int i = shared; i < blocks.size(); i++){
                    blocks[i] = m_manager.allocate_block();
                }
                seq.allocate_blocks(std::move(blocks), shared);
                //schedule
                auto id = seq.id();
                seq.set_status(SequenceStatus::running);
                m_running.emplace(id, std::move(seq));
                ret.push_back(&m_running[id]);
                m_waiting.pop();
                seq_len += std::min(m_chunk_size, seq.remaining_prefill_tokens());
            }
            else break;
        }
        else break;
    }

    return ret;
}

bool Scheduler::is_free()
{
    return m_running.empty() && m_waiting.empty();
}

JLLM_END
