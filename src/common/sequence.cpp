#include "sequence.hpp"

#include "../utils.hpp"

JLLM_BEGIN

Sequence::Sequence(size_t seq_id, const std::vector<int64_t> &prompt, size_t block_size)
    :m_seq_id(seq_id), m_prompt(prompt), m_block_size(block_size)
{
    size_t size = prompt.size() / block_size;
    m_num_needed_blocks = size + 1;
    m_hashs.resize(size);
    size_t pre = 0;
    for(size_t i = 0; i < size; i++) {
        size_t tmp = utils::vector_hash(prompt.begin() + i * block_size, prompt.begin() + (i + 1) * block_size);
        m_hashs[i] = utils::bind_hash(pre, tmp);
        pre = tmp;
    }
}

void Sequence::set_status(SequenceStatus status)
{
    m_status = status;
}
bool Sequence::is_finished() const {
    return m_status == SequenceStatus::finished;
}

void Sequence::add_token(int64_t token_id) {
    m_num_computed_tokens++;
    m_generated.push_back(token_id);
    if(m_num_computed_tokens != 0 && m_num_computed_tokens % m_block_size == 0) {
        size_t hash = 0;
        if(m_generated.size() >= m_block_size){
            hash = utils::vector_hash(m_generated.rbegin(), m_generated.rbegin() + m_block_size);
        }
        else {
            hash = utils::vector_hash(m_generated.rbegin(), m_generated.rend());
            size_t remain = m_block_size - m_generated.size();
            size_t hash2 = utils::vector_hash(m_prompt.rbegin(), m_prompt.rbegin() + remain);
            hash = utils::bind_hash(hash, hash2);
        }
        if(!m_hashs.empty())
            hash = utils::bind_hash(hash, m_hashs.back());
        m_hashs.push_back(hash);
        m_num_needed_blocks++;
    }
}

void Sequence::add_chunk(size_t chunk_size, int64_t token_id)
{
    m_num_computed_tokens += chunk_size;
    if(m_num_computed_tokens >= m_prompt.size()){
        add_token(token_id);
    }
}

void Sequence::allocate_blocks(std::vector<size_t> &&block_table, size_t shared_blocks)
{
    m_num_computed_tokens = shared_blocks * m_block_size;
    m_num_needed_blocks -= block_table.size();
    m_block_table = std::move(block_table);
}

void Sequence::allocate_block(size_t block_id)
{
    m_block_table.push_back(block_id);
    m_num_needed_blocks--;
}

int64_t Sequence::get_last_token() const {
    if(m_generated.empty()) {
        return -1; // or some invalid token id
    }
    return m_generated.back();
}

const std::vector<int64_t>& Sequence::prompt_tokens() const {
    return m_prompt;
}
const std::vector<int64_t>& Sequence::output_tokens() const {
    return m_generated;
}

const std::vector<size_t> &Sequence::hash_table() const
{
    return m_hashs;
}

const std::vector<size_t> &Sequence::block_table() const
{
    return m_block_table;
}

size_t Sequence::num_computed_tokens() const {
    return m_num_computed_tokens;
}

int Sequence::num_needed_blocks() const
{
    return m_num_needed_blocks;
}

size_t Sequence::prompt_len() const {
    return m_prompt.size();
}

size_t Sequence::id() const
{
    return m_seq_id;
}

bool Sequence::prefill_finished() const {
    return m_num_computed_tokens >= m_prompt.size();
}

size_t Sequence::remaining_prefill_tokens() const {
    ASSERT(m_num_computed_tokens <= m_prompt.size(), "Only called when process chunked prefill");
    return m_prompt.size() - m_num_computed_tokens;
}

JLLM_END