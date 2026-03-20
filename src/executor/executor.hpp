#pragma once

#include <string>
#include <thread>
#include <queue>

#include "input_meta.hpp"
#include "tensor/tensor.hpp"
#include "worker/worker.hpp"
#include "sequence.hpp"

JLLM_BEGIN

class Executor {
public:
    Executor(const std::string& model_name, const std::string& model_path,
        size_t block_size, size_t num_blocks, size_t max_seq_len = 64
    );

    void send(std::vector<Sequence*>& seqs);
    const std::vector<size_t>& kv_cache_shape();
private:
    tensor_t m_output;
    tensor_t m_input;
    Worker m_worker;
    size_t m_max_seq_len;
    size_t m_chunk_size;
    float m_temperature = 1;
    int m_topk = 50;
    float m_topp = 0.775;
};

JLLM_END