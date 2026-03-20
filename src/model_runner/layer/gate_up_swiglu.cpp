#include "gate_up_swiglu.hpp"

#include "../../utils.hpp"

#include "kernel/cpu/gate_up_swiglu.hpp"

namespace jllm::ops {
    void gate_up_swiglu(
        tensor_t out, tensor_t in, tensor_t gate_weight, tensor_t up_weight) {
        CHECK_SAME_DEVICE(out, in, gate_weight, up_weight);
        CHECK_ARGUMENT(
            in->shape().back() == gate_weight->shape().back(),
            "Input dimension does not match gate weight dimension."
        );
        CHECK_ARGUMENT(
            gate_weight->shape()[0] == up_weight->shape()[0],
            "Gate weight and up weight dimensions do not match."
        );

        size_t batch_size = 1;
        for (size_t i = 0; i < in->ndim() - 1; ++i) {
            batch_size *= in->shape()[i];
        }
        if (out->deviceType() == jllmDeviceType_t::CPU) {
            return cpu::gate_up_swiglu(
                out->data(),
                in->data(),
                gate_weight->data(),
                up_weight->data(),
                batch_size,
                in->shape().back(),
                up_weight->shape()[up_weight->ndim() - 2],
                in->dtype()
            );
        }
        TO_BE_IMPLEMENTED();
    }
}