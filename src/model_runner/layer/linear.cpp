#include "linear.hpp"

#include "../../utils.hpp"

#include "kernel/cpu/linear.hpp"

namespace jllm::ops {
    void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_ARGUMENT(
            in->shape().back() == weight->shape().back(),
            "Input dimension does not match weight dimension."
        );
    
        size_t batch_size = 1;
        for (size_t i = 0; i < in->ndim() - 1; ++i) {
            batch_size *= in->shape()[i];
        }
        if (out->deviceType() == jllmDeviceType_t::CPU) {
            return cpu::linear(
                out->data(),
                in->data(),
                weight->data(),
                bias->data(),
                batch_size,
                in->shape().back(),
                weight->shape()[weight->ndim() - 2],
                in->dtype()
            );
        }
        TO_BE_IMPLEMENTED();
    }
    
    void linear(tensor_t out, tensor_t in, tensor_t weight) {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_ARGUMENT(
            in->shape().back() == weight->shape().back(),
            "Input dimension does not match weight dimension."
        );
    
        size_t batch_size = 1;
        for (size_t i = 0; i < in->ndim() - 1; ++i) {
            batch_size *= in->shape()[i];
        }
        if (out->deviceType() == jllmDeviceType_t::CPU) {
            return cpu::linear(
                out->data(),
                in->data(),
                weight->data(),
                nullptr,
                batch_size,
                in->shape().back(),
                weight->shape()[weight->ndim() - 2],
                in->dtype()
            );
        }
        TO_BE_IMPLEMENTED();
    }
}