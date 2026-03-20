#pragma once

#include <memory>
#include <string>
#include <future>
#include <thread>

#include "../input_meta.hpp"
#include "../../tensor/tensor.hpp"
#include "../../model_runner/model_runner.hpp"
#include "../../kv_cache_manager/kv_cache/kv_cache.hpp"
#include "../utils.hpp"

JLLM_BEGIN

class Worker {
public:
    Worker(const std::string& model_name, const std::string& model_path,
        size_t max_seq_len, size_t block_size, size_t num_blocks
    );
    void generate(
        tensor_t output, tensor_t input, const InputMeta& meta,
        float temperature = 1, size_t topk = 50, float topp = 0.775
    );
    const ModelMeta& get_model_meta() const;
    const std::vector<size_t>& get_cache_shape() const;

private:
    std::shared_ptr<ModelRunner> m_runner;
    std::unique_ptr<KVCache> m_cache;
    tensor_t m_logits;
    const ModelMeta* m_model_meta;
};

JLLM_END