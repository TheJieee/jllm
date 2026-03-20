#include "add_norm.hpp"

#include "../../utils.hpp"

#include "kernel/cpu/add_norm.hpp"

namespace jllm::ops {
    void add_norm(
        tensor_t out,
        tensor_t in,
        const tensor_t add_tensor,
        const tensor_t norm_weight,
        float eps
    ) {
        CHECK_SAME_DEVICE(out, in, add_tensor, norm_weight);
        if(
            in->shape().back() != add_tensor->shape().back() || 
            in->shape().back() != norm_weight->shape().back()
        ) EXCEPTION_SHAPE_MISMATCH;
    
        if (out->deviceType() == jllmDeviceType_t::CPU) {
            return cpu::add_norm(
                out->data(),
                in->data(),
                add_tensor->data(),
                norm_weight->data(),
                in->shape()[0],
                in->shape().back(),
                eps,
                in->dtype()
            );
        }
        TO_BE_IMPLEMENTED();
    }
}