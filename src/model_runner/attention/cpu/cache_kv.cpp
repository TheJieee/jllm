#include "cache_kv.hpp"

#include "../../../utils.hpp"
#include <cstring>
#include <immintrin.h>
#include <vector>

template<typename T>
static void cache_kv_(
    T* k, T* v,             //shape[seq_len, nkvhead, head_dim]
    T* kcache, T* vcache,   //shape[nkvhead, block_size, head_dim]
    const std::vector<size_t>& slot_map,
    size_t block_size,
    size_t nkvhead, size_t head_dim
 ) {
    const size_t num_tokens = slot_map.size();
    const size_t head_size_bytes = head_dim * sizeof(T);
    const size_t row_stride = nkvhead * head_dim;

    // 提前计算常量步长，减少循环内乘法
    const size_t block_stride = nkvhead * block_size * head_dim;
    const size_t head_stride = block_size * head_dim;

    #pragma omp parallel for schedule(static) if(num_tokens > 1)
    for (size_t i = 0; i < num_tokens; i++) {
        const size_t slot_id = slot_map[i];
        const size_t block_id = slot_id / block_size;
        const size_t offset_in_block = slot_id % block_size;

        // 当前 Token 在输入 k/v 中的起始位置
        const T* src_k_token = k + i * row_stride;
        const T* src_v_token = v + i * row_stride;

        for (size_t h = 0; h < nkvhead; h++) {
            // 计算 Cache 中的目标地址
            // 基础偏移 = Block偏移 + Head偏移 + Slot在Block内的偏移
            size_t dst_base_off = (block_id * block_stride) + (h * head_stride) + (offset_in_block * head_dim);
            
            T* dst_k_ptr = kcache + dst_base_off;
            T* dst_v_ptr = vcache + dst_base_off;
            
            const T* src_k_ptr = src_k_token + h * head_dim;
            const T* src_v_ptr = src_v_token + h * head_dim;

            // --- AVX2 向量化拷贝 (按字节处理以适配不同 T) ---
            size_t b = 0;
            // 每次搬运 32 字节 (16个 BF16 或 8个 FP32)
            for (; b + 31 < head_size_bytes; b += 32) {
                __m256i m_k = _mm256_loadu_si256((const __m256i*)((const char*)src_k_ptr + b));
                __m256i m_v = _mm256_loadu_si256((const __m256i*)((const char*)src_v_ptr + b));
                
                _mm256_storeu_si256((__m256i*)((char*)dst_k_ptr + b), m_k);
                _mm256_storeu_si256((__m256i*)((char*)dst_v_ptr + b), m_v);
            }

            // 处理不足 32 字节的残余部分
            for (; b < head_size_bytes; b++) {
                ((char*)dst_k_ptr)[b] = ((const char*)src_k_ptr)[b];
                ((char*)dst_v_ptr)[b] = ((const char*)src_v_ptr)[b];
            }
        }
    }
}

// 假设 T 是 uint16_t 或专门的 __bf16 类型
template <typename T>
static void cache_kv_bf16_avx2(
    const T* k, const T* v,          // [seq_len, nkvhead, head_dim]
    T* kcache, T* vcache,            // [num_blocks, nkvhead, block_size, head_dim]
    const std::vector<size_t>& slot_map,
    size_t block_size,
    size_t nkvhead, size_t head_dim
) {
    const size_t num_tokens = slot_map.size();
    const size_t row_stride = nkvhead * head_dim;
    const size_t block_stride = nkvhead * block_size * head_dim;
    const size_t head_cache_stride = block_size * head_dim;

    // 并行化 Token 维度。如果是 Decoding 阶段 (num_tokens=1)，OMP 会自动跳过开销
    #pragma omp parallel for schedule(static) if(num_tokens > 1)
    for (size_t i = 0; i < num_tokens; i++) {
        const size_t slot_id = slot_map[i];
        const size_t block_id = slot_id / block_size;
        const size_t offset_in_block = slot_id % block_size;

        // 源 Token 起始位置
        const T* __restrict__ src_k_token = k + i * row_stride;
        const T* __restrict__ src_v_token = v + i * row_stride;

        // 目标 Block 基地址
        T* __restrict__ dst_k_block = kcache + block_id * block_stride;
        T* __restrict__ dst_v_block = vcache + block_id * block_stride;

        for (size_t h = 0; h < nkvhead; h++) {
            // 计算当前 Head 在 Cache 中的线性位置
            T* __restrict__ dst_k_ptr = dst_k_block + h * head_cache_stride + offset_in_block * head_dim;
            T* __restrict__ dst_v_ptr = dst_v_block + h * head_cache_stride + offset_in_block * head_dim;
            
            const T* __restrict__ src_k_ptr = src_k_token + h * head_dim;
            const T* __restrict__ src_v_ptr = src_v_token + h * head_dim;

            size_t d = 0;
            // AVX2 主循环：一次搬运 16 个 BF16 (32字节)
            for (; d + 15 < head_dim; d += 16) {
                __m256i v_k = _mm256_loadu_si256((const __m256i*)(src_k_ptr + d));
                __m256i v_v = _mm256_loadu_si256((const __m256i*)(src_v_ptr + d));
                
                _mm256_storeu_si256((__m256i*)(dst_k_ptr + d), v_k);
                _mm256_storeu_si256((__m256i*)(dst_v_ptr + d), v_v);
            }

            // 处理剩余不足 16 个元素的情况
            for (; d < head_dim; d++) {
                dst_k_ptr[d] = src_k_ptr[d];
                dst_v_ptr[d] = src_v_ptr[d];
            }
        }
    }
}


JLLM_BEGIN
namespace ops::cpu {
void cache_kv(
    std::byte *k, std::byte *v, std::byte *kcache, std::byte *vcache,
    const std::vector<size_t> &slot_map, size_t block_size,
    size_t nkvhead, size_t head_dim,
    jllmDataType_t dtype
){
    switch (dtype) {
    case jllmDataType_t::F32:
        cache_kv_(
            reinterpret_cast<float*>(k), reinterpret_cast<float*>(v),
            reinterpret_cast<float*>(kcache), reinterpret_cast<float*>(vcache),
            slot_map, block_size, nkvhead, head_dim
        );
        break;
    case jllmDataType_t::F16:
        cache_kv_(
            reinterpret_cast<fp16_t*>(k), reinterpret_cast<fp16_t*>(v),
            reinterpret_cast<fp16_t*>(kcache), reinterpret_cast<fp16_t*>(vcache),
            slot_map, block_size, nkvhead, head_dim
        );
        break;
    case jllmDataType_t::BF16:
        cache_kv_bf16_avx2(
            reinterpret_cast<uint16_t*>(k), reinterpret_cast<uint16_t*>(v),
            reinterpret_cast<uint16_t*>(kcache), reinterpret_cast<uint16_t*>(vcache),
            slot_map, block_size, nkvhead, head_dim
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for cache_kv");
    }
}
}
JLLM_END