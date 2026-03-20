#include "rope.hpp"

#include "utils.hpp"

#include <immintrin.h>
#include <omp.h>
#include <stdexcept>

// 辅助函数：加载 8 个 BF16 并转换为 FP32 (同你之前的逻辑)
static inline __m256 load_bf16_to_fp32(const void* ptr) {
    __m128i raw = _mm_loadu_si128((const __m128i*)ptr);
    __m256i u32 = _mm256_cvtepu16_epi32(raw);
    return _mm256_castsi256_ps(_mm256_slli_epi32(u32, 16));
}

// 辅助函数：将 FP32 向量截断并存回 8 个 BF16
static inline void store_fp32_to_bf16(void* ptr, __m256 f32_vec) {
    __m256i i32 = _mm256_castps_si256(f32_vec);
    __m256i shifted = _mm256_srli_epi32(i32, 16); 
    __m128i lo = _mm256_castsi256_si128(shifted);
    __m128i hi = _mm256_extracti128_si256(shifted, 1);
    __m128i packed = _mm_packus_epi32(lo, hi); 
    _mm_storeu_si128((__m128i*)ptr, packed);
}

void rope_optimized_bf16(
    std::byte* out,
    const std::byte* in,
    const float* rope_table,
    const int64_t* pos_ids,
    size_t seq_len, size_t nhead, size_t head_dim
) {
    const size_t half = head_dim / 2;
    auto* in_bf16 = (const uint16_t*)in;
    auto* out_bf16 = (uint16_t*)out;

    // 只有在 Prefill 阶段 (seq_len > 1) 开启并行才有显著收益
    #pragma omp parallel for collapse(2) if(seq_len * nhead > 4)
    for (size_t s = 0; s < seq_len; s++) {
        for (size_t h = 0; h < nhead; h++) {
            // 计算当前 Head 的数据起始位置
            const size_t offset = (s * nhead + h) * head_dim;
            const uint16_t* t_in = in_bf16 + offset;
            uint16_t* t_out = out_bf16 + offset;

            // 获取位置索引及对应的 cos/sin 表起始位置
            const size_t seq_pos = (size_t)pos_ids[s];
            const float* cur_cos = rope_table + seq_pos * head_dim;
            const float* cur_sin = cur_cos + half;

            size_t d = 0;
            // 向量化主循环：每次处理 8 对元素 (即 8 个 x1 和 8 个 x2)
            for (; d + 8 <= half; d += 8) {
                // 1. 加载数据
                __m256 x1 = load_bf16_to_fp32(t_in + d);
                __m256 x2 = load_bf16_to_fp32(t_in + d + half);
                __m256 cos_v = _mm256_loadu_ps(cur_cos + d);
                __m256 sin_v = _mm256_loadu_ps(cur_sin + d);

                // 2. 旋转计算: 
                // out1 = x1 * cos - x2 * sin
                // out2 = x1 * sin + x2 * cos
                __m256 res1 = _mm256_fmsub_ps(x1, cos_v, _mm256_mul_ps(x2, sin_v));
                __m256 res2 = _mm256_fmadd_ps(x1, sin_v, _mm256_mul_ps(x2, cos_v));

                // 3. 写回结果
                store_fp32_to_bf16(t_out + d, res1);
                store_fp32_to_bf16(t_out + d + half, res2);
            }

            // 4. 标量收尾 (处理 head_dim 不能被 16 整除的情况)
            for (; d < half; d++) {
                float f_x1 = jllm::utils::cast<float>(t_in[d]);
                float f_x2 = jllm::utils::cast<float>(t_in[d + half]);
                float c = cur_cos[d];
                float s_val = cur_sin[d];

                reinterpret_cast<jllm::bf16_t*>(t_out)[d] = jllm::utils::cast<jllm::bf16_t>(f_x1 * c - f_x2 * s_val);
                reinterpret_cast<jllm::bf16_t*>(t_out)[d + half] = jllm::utils::cast<jllm::bf16_t>(f_x1 * s_val + f_x2 * c);
            }
        }
    }
}

namespace jllm::ops::cpu {
    void rope(
        std::byte* out,
        const std::byte* in,
        const float* rope_table,
        const int64_t* pos_ids,
        size_t seq_len, size_t nhead, size_t head_dim,
        jllmDataType_t dtype
    ) {
        switch (dtype)
        {
        case jllmDataType_t::BF16:
            return rope_optimized_bf16(
                out, in, rope_table, pos_ids,
                seq_len, nhead, head_dim
            );
        default:
            throw std::invalid_argument("Unsupported date type.");
        }
    }
}