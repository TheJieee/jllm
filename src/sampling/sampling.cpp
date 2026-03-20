#include "sampling.hpp"

#include "../utils.hpp"

#include "kernel/cpu/sampling.hpp"

JLLM_BEGIN

void ops::sampling(tensor_t out, tensor_t logits, float temperature, int topk, float topp)
{
    CHECK_SAME_DEVICE(out, logits);
    CHECK_ARGUMENT(
        logits->ndim() >= 1,
        "logits must be at least 1D tensor."
    );
    CHECK_ARGUMENT(
        out->ndim() >= 1,
        "out must be at least 1D tensor."
    );
    
    if (logits->deviceType() == jllmDeviceType_t::CPU) {
        size_t batch_size = logits->shape().front();
        size_t vocab_size = logits->shape().back();
        
        return cpu::sampling(
            out->data(),
            logits->data(),
            temperature,
            topk,
            topp,
            batch_size,
            vocab_size,
            logits->dtype()
        );
    }
    TO_BE_IMPLEMENTED();
}

JLLM_END