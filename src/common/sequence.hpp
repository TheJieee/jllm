#pragma once
#include "jllm.h"

#include <vector>

JLLM_BEGIN
enum class SequenceStatus {
    waiting,
    running,
    swapped,
    finished,
};

class Sequence {
public:
    Sequence() = default;
    Sequence(size_t seq_id, const std::vector<int64_t>& prompt, size_t block_size);

    void set_status(SequenceStatus status);
    bool is_finished() const;
    void add_token(int64_t token_id);
    void add_chunk(size_t chunk_size, int64_t token_id);
    void allocate_blocks(std::vector<size_t>&& block_table, size_t shared_blocks);
    void allocate_block(size_t block_id);
    int64_t get_last_token() const;
    const std::vector<int64_t>& prompt_tokens() const;
    const std::vector<int64_t>& output_tokens() const;
    const std::vector<size_t>&  hash_table()    const;
    const std::vector<size_t>&  block_table()   const;
    size_t num_computed_tokens() const;
    int    num_needed_blocks()   const;
    size_t prompt_len() const;
    size_t id() const;
    bool prefill_finished() const;
    size_t remaining_prefill_tokens() const;

private:
    size_t m_seq_id;
    SequenceStatus m_status{SequenceStatus::waiting};
    std::vector<int64_t> m_prompt;
    std::vector<int64_t> m_generated;
    std::vector<size_t> m_block_table;
    std::vector<size_t> m_hashs;
    size_t m_num_computed_tokens{0};
    size_t m_block_size;
    int m_num_needed_blocks;
};
JLLM_END