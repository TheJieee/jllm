#pragma once
#include "jllm.h"

#include "../model_runner.hpp"
#include "../kv_cache_view.hpp"
#include <map>
#include <memory>


JLLM_BEGIN

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;   // a.k.a. model.norm.weight
    std::vector<tensor_t> attn_norm_w; // a.k.a. input_layernorm.weight
    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;
    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;
    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;
    std::vector<tensor_t> attn_o_w;
    std::vector<tensor_t> mlp_norm_w; // a.k.a. post_attention_layernorm.weight
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

struct Infer_tensors_buf {
    tensor_t x;
    tensor_t x_norm;
    tensor_t v_;
    tensor_t q_;
    tensor_t q_rope;
    tensor_t k_;
    tensor_t k_rope;
    tensor_t pos_ids;
    tensor_t attn_val;
    tensor_t attn_out;
    tensor_t swiglu_out;
    tensor_t ffn_out;
    tensor_t logits;
    size_t seq_len = 0;
    size_t total_len = 0;
};

class Qwen2 :public ModelRunner {
public:
    Qwen2() = default;
    ~Qwen2() = default;
    KVCache** init(const std::string& path) override;
    void load_model() override;
    void forward(tensor_t out_logits, tensor_t input, const InputMeta& meta) override;
    const ModelMeta& get_model_meta() override;

private:
    Infer_tensors_buf get_(size_t seq_len);
    void load_config();
private:
    ModelMeta m_meta;
    std::string m_path;
    Qwen2Weights m_weights;
    KVCache* m_view{};
    tensor_t m_rope_table;
    Infer_tensors_buf infer_buf_;
};

JLLM_END