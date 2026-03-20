#include "linear_rope.hpp"

#include "../../utils.hpp"

#include "kernel/cpu/linear_rope.hpp"

namespace jllm::ops {
    void linear_rope(
        tensor_t out, //shape[seq_len, num_head, head_dim]
        const tensor_t in, //shape[seq_len, hidden_size]
        const tensor_t weight, //shape[nhead*head_dim, hidden_size] or [hidden_size, nhead*head_dim]
        const tensor_t bias, 
        const tensor_t seq_pos,
        const tensor_t rope_table //shape[max_pos, head_dim]
    ) {
        CHECK_SAME_DEVICE(out, in, weight, bias, seq_pos, rope_table);
        CHECK_ARGUMENT(
            in->shape().back() == weight->shape().back(),
            "Input dimension does not match weight dimension."
        );
        CHECK_ARGUMENT(
            seq_pos->ndim() == 1 && seq_pos->dtype() == jllmDataType_t::I64,
            "seq_pos must be a 1D tensor of int64."
        );
        
        // 提取参数
        size_t seq_len = in->shape()[0];
        size_t hidden_size = in->shape()[1];  // in: [seq_len, hidden_size]
        size_t nhead = out->shape()[1];       // out: [seq_len, nhead, head_dim]
        size_t head_dim = out->shape().back();// 
        
        CHECK_ARGUMENT(
            hidden_size == weight->shape().back() && 
            nhead * head_dim == weight->shape()[weight->ndim() - 2],
            "Weight shape does not match: should be [nhead*head_dim, hidden_size]"
        );
    
        if (out->deviceType() == jllmDeviceType_t::CPU) {
            return cpu::linear_rope(
                out->data(),
                in->data(),
                weight->data(),
                bias->data(),
                reinterpret_cast<const int64_t*>(seq_pos->data()),
                reinterpret_cast<const float*>(rope_table->data()),
                seq_len,        // batch_size
                nhead,          // nhead
                hidden_size,    // in_features
                head_dim,       // out_features
                in->dtype()
            );
        }
        TO_BE_IMPLEMENTED();
    }
}