#include "rms_norm.hpp"

#include "utils.hpp"

#include "kernel/cpu/rms_norm.hpp"

namespace jllm::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    size_t feature_size = in->shape().back();
    size_t batch_size = in->shape().front();

    // Call CPU implementation
    if (in->deviceType() == jllmDeviceType_t::CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            in->dtype(),
            batch_size,
            feature_size,
            eps);
    }
    //TODO: Add more device implementations here
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
