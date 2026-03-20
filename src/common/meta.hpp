#pragma once
#include "jllm.h"

#include <string>

JLLM_BEGIN

struct ModelMeta {
    std::string name;
    size_t hidden_size;
    size_t intermediate_size;
    size_t nhead, nkvhead;
    size_t head_dim, vhead_dim;
    size_t vocab_size;
    size_t nlayer;
    size_t eos_token_id;
    size_t max_position_embeddings;
    float rms_norm_eps;
    float rope_theta;
    jllmDataType_t dtype;
};

JLLM_END