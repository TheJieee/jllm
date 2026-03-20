#include "executor.hpp"

#include "utils.hpp"
#include "log/log.hpp"


JLLM_BEGIN

Executor::Executor(
    const std::string& model_name, const std::string& model_path,
     size_t block_size, size_t num_blocks, size_t max_seq_len
)   :m_worker(model_name, model_path, max_seq_len, block_size, num_blocks),
    m_max_seq_len(max_seq_len), m_chunk_size(block_size)
{
    const ModelMeta& meta = m_worker.get_model_meta();
    m_output = Tensor::create({max_seq_len}, jllmDataType_t::I64);
    m_input = Tensor::create({max_seq_len}, jllmDataType_t::I64);
}

// struct InputMeta {
//     size_t seq_len;
//     std::vector<size_t> cu_seq_len;
//     std::vector<size_t> slot_map;
//     std::vector<size_t> context_len;
//     std::vector<const std::vector<size_t>*> block_tables;
// };

// struct package {
//     tensor_t in;
//     InputMeta meta;
//     float temperature;
//     int topk;
//     float topp;
// };

void Executor::send(std::vector<Sequence*>& seqs)
{
    //make meta
    auto& cache_shape = m_worker.get_cache_shape();
    size_t block_size = cache_shape[2];
    size_t seq_len = 0;
    InputMeta meta;
    auto& cusl = meta.cu_seq_len;
    auto& sm = meta.slot_map;
    auto& cl = meta.context_len;
    auto& bts = meta.block_tables;
    cusl.resize(seqs.size() + 1);
    cusl[0] = 0;
    cl.resize(seqs.size());
    bts.resize(seqs.size());
    auto p = reinterpret_cast<int64_t*>(m_input->data());
    for(size_t i = 0; i < seqs.size(); i++) {
        bts[i] = &seqs[i]->block_table();
        if(seqs[i]->prefill_finished()){//decode
            cusl[i + 1] = cusl[i] + 1;
            cl[i] = seqs[i]->num_computed_tokens();
            size_t order = cl[i];
            size_t base = bts[i]->at(order / block_size) * block_size;
            size_t offset = seqs[i]->num_computed_tokens() % block_size;
            sm.push_back(base + offset);
            p[seq_len] = seqs[i]->get_last_token();
            seq_len++;
        }
        else {//chunked prefill
            cl[i] = seqs[i]->num_computed_tokens();
            size_t size = std::min(seqs[i]->remaining_prefill_tokens(), m_chunk_size);
            cusl[i + 1] = cusl[i] + size;
            std::memcpy(
                p + seq_len, seqs[i]->prompt_tokens().data() + cl[i],
                size * sizeof(int64_t)
            );
            size_t begin = cl[i] / block_size, offset = cl[i] % block_size;
            const auto& block_table = seqs[i]->block_table();
            for(size_t j = 0; j < size; j++) {
                sm.push_back(block_table[begin] * block_size + offset);
                offset++;
                if(offset == block_size){
                    offset = 0;
                    begin++;
                }
                seq_len++;
            }
        }
    }
    meta.seq_len = seq_len;
    //send
    m_worker.generate(
        m_output->slice(0, 0, seq_len), m_input->slice(0, 0, seq_len), meta,
        m_temperature, m_topk, m_topp
    );
    //write back
    auto& model_meta = m_worker.get_model_meta();
    auto eos = model_meta.eos_token_id;
    auto pout = reinterpret_cast<int64_t*>(m_output->data());
    for(size_t i = 0; i < seqs.size(); i++) {
        int64_t token = pout[cusl[i + 1] - 1];

        if(seqs[i]->prefill_finished())
            seqs[i]->add_token(token);
        else {
            size_t size = std::min(seqs[i]->remaining_prefill_tokens(), m_chunk_size);
            seqs[i]->add_chunk(size, token);
        }
        if(token == eos){
            seqs[i]->set_status(SequenceStatus::finished);
        }
    }
}

const std::vector<size_t> &Executor::kv_cache_shape()
{
    return m_worker.get_cache_shape();
}

JLLM_END