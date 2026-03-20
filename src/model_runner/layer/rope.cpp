#include "rope.hpp"

#include "kernel/cpu/rope.hpp"

#include "utils.hpp"

JLLM_BEGIN

void ops::rope(tensor_t out, tensor_t in, tensor_t pos_ids, tensor_t rope_table)
{
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    ASSERT(rope_table->shape().front() >= pos_ids->shape().front(), "RoPe table too short.");
    CHECK_ARGUMENT(pos_ids->shape().front() == out->shape().front(), "Sequence length mismatch.");

    if(out->deviceType() == jllmDeviceType_t::CPU) {
        return ops::cpu::rope(
            out->data(), in->data(), 
            reinterpret_cast<float*>(rope_table->data()),
            reinterpret_cast<int64_t*>(pos_ids->data()),
            out->shape()[0],
            out->shape()[1],
            out->shape()[2],
            out->dtype()
        );
    }
    TO_BE_IMPLEMENTED();
}

JLLM_END