#include "embedding.hpp"

#include "utils.hpp"

#include "kernel/cpu/embedding.hpp"

namespace jllm::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(
        index->dtype() == jllmDataType_t::I64,
        "Index tensor must be of type INT64."
    );

    if(out->deviceType() == jllmDeviceType_t::CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            index->numel(),
            weight->shape().back(),
            out->dtype()
        );
    }
}
} // namespace llaisys::ops