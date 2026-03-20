#include "fused_attention.hpp"

#include "../../utils/check.hpp"
#include "../../executor/input_meta.hpp"
#include "cpu/fused_attention.hpp"

JLLM_BEGIN
namespace ops {
void fused_attention(
    tensor_t out, tensor_t q,
    const KVCacheView& view, 
    const InputMeta& meta
) {
    size_t nhead = q->shape()[1];
    size_t nkvhead = view.shape[1];
    size_t head_dim = view.shape.back();
    // Parameter validation
    CHECK_SAME_DEVICE(out, q);
    CHECK_SAME_DTYPE(out->dtype(), q->dtype());
    
    CHECK_ARGUMENT(
        out->ndim() >= 2,
        "Output tensor must have at least 2 dimensions"
    );
    
    CHECK_ARGUMENT(
        q->ndim() >= 2,
        "Query tensor must have at least 2 dimensions"
    );
    
    // Head dimension check
    CHECK_ARGUMENT(
        q->shape().back() == head_dim,
        "Query last dimension must match head_dim"
    );
    
    // Get device type and process accordingly
    if (out->deviceType() == jllmDeviceType_t::CPU) {
        return jllm::ops::cpu::fused_attention(
            out->data(),
            q->data(), nhead,
            view,
            meta,
            out->dtype()
        );
    }
    
    TO_BE_IMPLEMENTED();
}
}
JLLM_END