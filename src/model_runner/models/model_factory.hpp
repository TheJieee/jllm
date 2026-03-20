#pragma once
#include "jllm.h"

#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <functional>
#include <pybind11/pybind11.h>
#include "../model_runner.hpp"

#define REGISTER_MODEL(model_class, model_name)         \
    void bind_##model_class(pybind11::module_ &m) {     \
        ModelFactory::registerModel(model_name, []() {  \
            return std::make_shared<model_class>();     \
        });                                             \
    }
JLLM_BEGIN

class ModelFactory {
public:
    using Creator = std::function<std::shared_ptr<ModelRunner>()>;

    static ModelFactory& get() {
        static ModelFactory s_instance;
        return s_instance;
    }

    static std::shared_ptr<ModelRunner> get_model(const std::string& name) {
        auto& instance = get();
        auto it = instance.models.find(name);
        if(it == instance.models.end()){
            throw std::runtime_error("Unsupported model: " + name);
        }
        return it->second();
    }

    static void registerModel(const std::string& name, Creator creator) {
        get().models[name] = creator;
    }

    ModelFactory(const ModelFactory&) = delete;
    ModelFactory(ModelFactory&&) = delete;
private:
    ModelFactory() = default;
    std::map<std::string, Creator> models;
};

JLLM_END