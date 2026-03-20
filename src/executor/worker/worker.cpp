#include "worker.hpp"

#include "model_runner/models/model_factory.hpp"
#include "sampling/sampling.hpp"

JLLM_BEGIN

Worker::Worker(const std::string &model_name, const std::string &model_path, 
    size_t max_seq_len, size_t block_size, size_t num_blocks
)
{
    m_runner = ModelFactory::get_model(model_name);
    KVCache** pass = m_runner->init(model_path);
    m_runner->load_model();
    m_model_meta = &m_runner->get_model_meta();
    m_cache = std::make_unique<KVCache>(
        m_model_meta->nlayer, m_model_meta->nkvhead, m_model_meta->head_dim,
        m_model_meta->dtype, m_model_meta->vocab_size, block_size, num_blocks
    );
    *pass = m_cache.get();
    m_logits = Tensor::create({max_seq_len, m_model_meta->vocab_size}, m_model_meta->dtype);
}

void Worker::generate(
    tensor_t output, tensor_t input, const InputMeta &meta,
    float temperature, size_t topk, float topp
){
    auto logits = m_logits->slice(0, 0, input->shape()[0]);
    m_runner->forward(logits, input, meta);
    const auto& slot_map = meta.slot_map;
    for(size_t i = 0; i < meta.slot_map.size(); i++) {
        auto pos = slot_map[i];
        if(pos % m_cache->block_size() == m_cache->block_size() - 1){
            m_cache->get_logits(pos / m_cache->block_size())->load(
                m_logits->slice(0, i, i + 1)->data()
            );
        }
    }
    ops::sampling(output, logits, temperature, topk, topp);
}

const ModelMeta &Worker::get_model_meta() const
{
    return *m_model_meta;
}

const std::vector<size_t> &Worker::get_cache_shape() const
{
    static std::vector<size_t> shape{
        m_cache->get_num_block(),
        m_model_meta->nkvhead,
        m_cache->block_size(),
        m_model_meta->head_dim
    };
    return shape;
}

JLLM_END