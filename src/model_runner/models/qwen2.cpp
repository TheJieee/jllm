#include "qwen2.hpp"

#include "json.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <future>
#include <cmath>
#include <vector>
#include "../../kernel/kernel.h"
#include "../layer/ops.hpp"
#include "../attention/attention.hpp"
#include "../attention/cache_kv.hpp"
#include "log/log.hpp"

#include "model_factory.hpp"

using jllm::Allocator;

JLLM_BEGIN

REGISTER_MODEL(Qwen2, "Qwen2");

using json = nlohmann::json;

KVCache** Qwen2::init(const std::string &path)
{
    m_path = path;
    load_config();
    m_rope_table = Tensor::create({m_meta.max_position_embeddings, m_meta.head_dim}, jllmDataType_t::F32);

    // 初始化 rope_table
    // RoPE 公式：
    // theta_i = rope_theta^(-2i/d)
    // cos_i = cos(m * theta_i)， sin_i = sin(m * theta_i)
    // 其中 m 是位置（seq_pos）， i 从 0 到 d/2-1
    
    float* rope_data = (float*)m_rope_table->data();  // 临时转换为 float 处理
    size_t max_pos = m_meta.max_position_embeddings;
    size_t head_dim = m_meta.head_dim;
    double rope_theta = m_meta.rope_theta;
    
    // 预计算频率倒数：inv_freq = 1.0 / (rope_theta^(2*i/d))
    std::vector<double> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0 / std::pow(rope_theta, 2.0 * i / head_dim);
    }
    
    // 对每个位置计算 rope 值
    // rope_table 总是 F32 格式存储，无论模型 dtype 是什么
    #pragma omp parallel for schedule(static)
    for (size_t seq_pos = 0; seq_pos < max_pos; seq_pos++) {
        float* row_data = rope_data + seq_pos * head_dim;
        for (size_t i = 0; i < head_dim / 2; i++) {
            double theta = (double)seq_pos * inv_freq[i];
            
            // 布局改为：[cos_0, cos_1...cos_{d/2-1}, sin_0, sin_1...sin_{d/2-1}]
            row_data[i] = (float)std::cos(theta);               // 前半部分存 cos
            row_data[i + head_dim / 2] = (float)std::sin(theta); // 后半部分存 sin
        }
    }
    return &m_view;
}

void Qwen2::load_model()
{
    spdlog::info("Loding weights...\n");
    std::string model_path = m_path + "/model.safetensors";
    
    // Load the entire safetensors file using from_file
    storage_t model_storage = Allocator::from_file(model_path.c_str());
    
    // Parse the safetensors header
    // First 8 bytes: header size as little-endian u64
    const uint8_t* data = reinterpret_cast<const uint8_t*>(model_storage->memory());
    
    uint64_t header_size = 0;
    std::memcpy(&header_size, data, 8);
    
    // The header is JSON format
    std::string header_str(reinterpret_cast<const char*>(data + 8), header_size);
    json header = json::parse(header_str);
    
    // Data offset after header
    size_t data_offset = 8 + header_size;
    
    // Initialize weight vectors with correct size
    m_weights.attn_norm_w.resize(m_meta.nlayer);
    m_weights.attn_q_w.resize(m_meta.nlayer);
    m_weights.attn_q_b.resize(m_meta.nlayer);
    m_weights.attn_k_w.resize(m_meta.nlayer);
    m_weights.attn_k_b.resize(m_meta.nlayer);
    m_weights.attn_v_w.resize(m_meta.nlayer);
    m_weights.attn_v_b.resize(m_meta.nlayer);
    m_weights.attn_o_w.resize(m_meta.nlayer);
    m_weights.mlp_norm_w.resize(m_meta.nlayer);
    m_weights.mlp_gate_w.resize(m_meta.nlayer);
    m_weights.mlp_up_w.resize(m_meta.nlayer);
    m_weights.mlp_down_w.resize(m_meta.nlayer);
    
    // Parse each tensor from header and assign to Qwen2Weights
    for (auto& [tensor_name, tensor_info] : header.items()) {
        if (tensor_info.contains("shape") && tensor_info.contains("dtype") && 
            tensor_info.contains("data_offsets")) {
            
            // Get tensor metadata
            std::vector<size_t> shape = tensor_info["shape"].get<std::vector<size_t>>();
            std::string dtype_str = tensor_info["dtype"].get<std::string>();
            
            // Parse data offsets: [start, end]
            auto offsets = tensor_info["data_offsets"].get<std::vector<size_t>>();
            size_t param_offset = data_offset + offsets[0];
            
            // Convert dtype string to jllmDataType_t
            jllmDataType_t dtype = jllmDataType_t::INVALID;
            if (dtype_str == "BF16" || dtype_str == "bfloat16") {
                dtype = jllmDataType_t::BF16;
            } else if (dtype_str == "F32" || dtype_str == "float32") {
                dtype = jllmDataType_t::F32;
            } else if (dtype_str == "F16" || dtype_str == "float16") {
                dtype = jllmDataType_t::F16;
            } else if (dtype_str == "I64" || dtype_str == "int64") {
                dtype = jllmDataType_t::I64;
            } else if (dtype_str == "I32" || dtype_str == "int32") {
                dtype = jllmDataType_t::I32;
            }
            
            // Create tensor_t using Tensor constructor
            tensor_t param_ = std::make_shared<Tensor>(shape, dtype, model_storage, param_offset);
            tensor_t param = Tensor::create(param_->shape(), param_->dtype());
            param->load(param_->data());
            // Map tensor names to Qwen2Weights fields
            if (tensor_name == "model.embed_tokens.weight") {
                m_weights.in_embed = param;
            } else if (tensor_name == "lm_head.weight") {
                m_weights.out_embed = param;
            } else if (tensor_name == "model.norm.weight") {
                m_weights.out_norm_w = param;
            } else {
                // Parse layer-specific parameters
                // Format: model.layers.{layer_id}.{component}
                size_t layer_pos = tensor_name.find("model.layers.");
                if (layer_pos != std::string::npos) {
                    // Extract layer index
                    size_t start = layer_pos + 13; // "model.layers." length
                    size_t end = tensor_name.find(".", start);
                    if (end != std::string::npos) {
                        int layer_id = std::stoi(tensor_name.substr(start, end - start));
                        std::string component = tensor_name.substr(end + 1);
                        
                        if (component.find("input_layernorm.weight") != std::string::npos) {
                            m_weights.attn_norm_w[layer_id] = param;
                        } else if (component.find("self_attn.q_proj.weight") != std::string::npos) {
                            m_weights.attn_q_w[layer_id] = param;
                        } else if (component.find("self_attn.q_proj.bias") != std::string::npos) {
                            m_weights.attn_q_b[layer_id] = param;
                        } else if (component.find("self_attn.k_proj.weight") != std::string::npos) {
                            m_weights.attn_k_w[layer_id] = param;
                        } else if (component.find("self_attn.k_proj.bias") != std::string::npos) {
                            m_weights.attn_k_b[layer_id] = param;
                        } else if (component.find("self_attn.v_proj.weight") != std::string::npos) {
                            m_weights.attn_v_w[layer_id] = param;
                        } else if (component.find("self_attn.v_proj.bias") != std::string::npos) {
                            m_weights.attn_v_b[layer_id] = param;
                        } else if (component.find("self_attn.o_proj.weight") != std::string::npos) {
                            m_weights.attn_o_w[layer_id] = param;
                        } else if (component.find("post_attention_layernorm.weight") != std::string::npos) {
                            m_weights.mlp_norm_w[layer_id] = param;
                        } else if (component.find("mlp.gate_proj.weight") != std::string::npos) {
                            m_weights.mlp_gate_w[layer_id] = param;
                        } else if (component.find("mlp.up_proj.weight") != std::string::npos) {
                            m_weights.mlp_up_w[layer_id] = param;
                        } else if (component.find("mlp.down_proj.weight") != std::string::npos) {
                            m_weights.mlp_down_w[layer_id] = param;
                        } else {
                            throw std::runtime_error("Unknown weight name: " + component);
                        }
                    }
                }
            }
        }
    }
    spdlog::info("Weights load successfully.\n");
}

// struct InputMeta {
    //     size_t seq_len;
    //     std::vector<size_t> cu_seq_len;
    //     std::vector<size_t> slot_map;
    //     std::vector<size_t> context_len;
    //     std::vector<const std::vector<size_t>*> block_tables;
    // };

void Qwen2::forward(tensor_t out_logits, tensor_t input, const InputMeta &meta)
{
    size_t seq_len = meta.seq_len;
    auto& cu_seq_len = meta.cu_seq_len;
    auto& slot_map = meta.slot_map;
    auto& block_tables = meta.block_tables;
    auto& context_len = meta.context_len;
    //Allocate tensor
    auto cache = get_(seq_len);
    auto& x                 = cache.x;
    auto& x_norm            = cache.x_norm;
    auto& v_                = cache.v_;
    auto& q_                = cache.q_;
    auto& k_                = cache.k_;
    auto& q_rope            = cache.q_rope;
    auto& k_rope            = cache.k_rope;
    auto& pos_ids           = cache.pos_ids;
    auto& attn_val          = cache.attn_val;
    auto& attn_out          = cache.attn_out;
    auto& swiglu_out        = cache.swiglu_out;
    auto& ffn_out           = cache.ffn_out;

    //embedding
    ops::embedding(x, input, m_weights.in_embed);

    KVCacheView view;
    view.shape = {
        m_view->get_num_block(),
        m_meta.nkvhead,
        m_view->block_size(),
        m_meta.head_dim,
    };

    auto p_pos_ids = (int64_t*)pos_ids->data();
    for(size_t i = 0; i < cu_seq_len.size() - 1; i++) {
        for(size_t j = cu_seq_len[i]; j < cu_seq_len[i + 1]; j++) {
            p_pos_ids[j] = context_len[i] + j - cu_seq_len[i];
        }
    }
    int a = 1 + 1;
    for(size_t i = 0; i < m_meta.nlayer; i++) {
        view.k_cache = m_view->get_kblock(i)->data();
        view.v_cache = m_view->get_vblock(i)->data();

        if(i == 0) {
            ops::rms_norm(x_norm, x, m_weights.attn_norm_w[0], m_meta.rms_norm_eps);
        }
        else {
            ops::add_norm(
                x_norm,
                x, attn_out,
                m_weights.attn_norm_w[i], m_meta.rms_norm_eps
            );
        }

        //attention
        ops::linear(v_, x_norm, m_weights.attn_v_w[i], m_weights.attn_v_b[i]);
        ops::linear(k_, x_norm, m_weights.attn_k_w[i]);
        ops::linear(q_, x_norm, m_weights.attn_q_w[i], m_weights.attn_q_b[i]);

        ops::rope(q_rope, q_, pos_ids, m_rope_table);
        ops::rope(k_rope, k_, pos_ids, m_rope_table);

        ops::cache_kv(k_rope, v_, view, slot_map);
        ops::fused_attention(attn_val, q_rope, view, meta);
        ops::linear(attn_out, attn_val, m_weights.attn_o_w[i]);

        ops::add_norm(x_norm, attn_out, x, m_weights.mlp_norm_w[i], m_meta.rms_norm_eps);
        //FFN
        ops::gate_up_swiglu(swiglu_out, x_norm, m_weights.mlp_gate_w[i], m_weights.mlp_up_w[i]);
        ops::linear(x, swiglu_out, m_weights.mlp_down_w[i]);
    }
    ops::add_norm(x_norm, x, attn_out, m_weights.out_norm_w, m_meta.rms_norm_eps);
    ops::linear(out_logits, x_norm, m_weights.out_embed);
}

const ModelMeta &Qwen2::get_model_meta()
{
    return m_meta;
}

Infer_tensors_buf Qwen2::get_(size_t seq_len)
{
    infer_buf_.total_len += seq_len;
   
    Infer_tensors_buf cache;
    if (infer_buf_.seq_len < seq_len) {
        infer_buf_.x                 = Tensor::create({seq_len, m_meta.hidden_size}, m_meta.dtype);
        infer_buf_.x_norm            = Tensor::create({seq_len, m_meta.hidden_size}, m_meta.dtype);
        infer_buf_.v_                = Tensor::create({seq_len, m_meta.nkvhead, m_meta.head_dim}, m_meta.dtype);
        infer_buf_.q_                = Tensor::create({seq_len, m_meta.nhead, m_meta.head_dim}, m_meta.dtype);
        infer_buf_.k_                = Tensor::create({seq_len, m_meta.nkvhead, m_meta.head_dim}, m_meta.dtype);
        infer_buf_.q_rope            = Tensor::create({seq_len, m_meta.nhead, m_meta.head_dim}, m_meta.dtype);
        infer_buf_.k_rope            = Tensor::create({seq_len, m_meta.nkvhead, m_meta.head_dim}, m_meta.dtype);
        infer_buf_.pos_ids           = Tensor::create({seq_len}, jllmDataType_t::I64);
        infer_buf_.attn_val          = Tensor::create({seq_len, m_meta.hidden_size}, m_meta.dtype);
        infer_buf_.attn_out          = Tensor::create({seq_len, m_meta.hidden_size}, m_meta.dtype);
        infer_buf_.swiglu_out        = Tensor::create({seq_len, m_meta.intermediate_size}, m_meta.dtype);
        infer_buf_.ffn_out           = Tensor::create({seq_len, m_meta.hidden_size}, m_meta.dtype);
        infer_buf_.seq_len = seq_len;
        cache = infer_buf_;
    }else {
        cache.x = infer_buf_.x->slice(0, 0, seq_len);
        cache.x_norm = infer_buf_.x_norm->slice(0, 0, seq_len);
        cache.v_ = infer_buf_.v_->slice(0, 0, seq_len);
        cache.q_ = infer_buf_.q_rope->slice(0, 0, seq_len);
        cache.k_ = infer_buf_.k_rope->slice(0, 0, seq_len);
        cache.q_rope = infer_buf_.q_rope->slice(0, 0, seq_len);
        cache.k_rope = infer_buf_.k_rope->slice(0, 0, seq_len);
        cache.pos_ids = infer_buf_.pos_ids->slice(0, 0, seq_len);
        cache.attn_val = infer_buf_.attn_val->slice(0, 0, seq_len);
        cache.attn_out = infer_buf_.attn_out->slice(0, 0, seq_len);
        cache.swiglu_out = infer_buf_.swiglu_out->slice(0, 0, seq_len);
        cache.ffn_out = infer_buf_.ffn_out->slice(0, 0, seq_len);

    }
    return cache;
}

void Qwen2::load_config()
{
    std::ifstream file(m_path + "/config.json");
    if(!file) throw std::runtime_error("Fail to open model config file.");
    json config = json::parse(file);
    m_meta.dtype = jllmDataType_t::BF16;
    m_meta.eos_token_id = config["eos_token_id"];
    m_meta.hidden_size = config["hidden_size"];
    m_meta.intermediate_size = config["intermediate_size"];
    m_meta.max_position_embeddings = config["max_position_embeddings"];
    m_meta.name = config["model_type"];
    m_meta.nhead = config["num_attention_heads"];
    m_meta.nkvhead = config["num_key_value_heads"];
    m_meta.nlayer = config["num_hidden_layers"];
    m_meta.vocab_size = config["vocab_size"];
    m_meta.rms_norm_eps = config["rms_norm_eps"];
    m_meta.rope_theta = config["rope_theta"];
    m_meta.head_dim = m_meta.hidden_size / m_meta.nhead;
    m_meta.vhead_dim = m_meta.head_dim;
}

JLLM_END