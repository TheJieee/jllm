#include "cache_kv.hpp"

#include "cpu/cache_kv.hpp"
#include "utils.hpp"

JLLM_BEGIN
namespace ops {
void cache_kv(
    tensor_t k, tensor_t v,
    KVCacheView& view,
    const std::vector<size_t>& slot_map
) {
    CHECK_ARGUMENT(
        k->shape() == v->shape(),
        "shape mismatch"
    );

    if(k->deviceType() == jllmDeviceType_t::CPU) {
        return cpu::cache_kv(
            k->data(), v->data(),
            view.k_cache, view.v_cache, slot_map,
            view.shape[2], view.shape[1], view.shape[3],
            k->dtype()
        );
    }
    TO_BE_IMPLEMENTED();
}
}
JLLM_END