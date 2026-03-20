#pragma once
#include "jllm.h"

#include "meta.hpp"
#include "../tensor/tensor.hpp"
#include "../executor/input_meta.hpp"
#include "kv_cache_manager/kv_cache/kv_cache.hpp"
#include <string>

JLLM_BEGIN

class ModelRunner {
public:
    virtual KVCache** init(const std::string& path) = 0;
    virtual void load_model() = 0;
    virtual void forward(tensor_t out_logits, tensor_t input, const InputMeta& meta) = 0;
    virtual const ModelMeta& get_model_meta() = 0;

    ModelRunner() = default;
    virtual ~ModelRunner() {}
};

JLLM_END