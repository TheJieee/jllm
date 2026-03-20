#include "gate_up_swiglu.hpp"

#include "utils.hpp"
#include <omp.h>
#include <cmath>
#include <immintrin.h>
#include <algorithm>

template <typename T>
static void gate_up_swiglu_(
    T* out,
    const T* in,
    const T* gate_weight,
    const T* up_weight,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    using namespace jllm::utils;
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        const T* in_batch = in + b * in_features;
        T* out_batch = out + b * out_features;

        for (size_t o = 0; o < out_features; o++) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            const T* gate_weight_ = gate_weight + o * in_features;
            const T* up_weight_ = up_weight + o * in_features;
            
            for (size_t i = 0; i < in_features; i++) {
                gate_sum += cast<float>(in_batch[i]) * cast<float>(gate_weight_[i]);
                up_sum += cast<float>(in_batch[i]) * cast<float>(up_weight_[i]);
            }
            float val = gate_sum / (1.0f + std::exp(-gate_sum)) * up_sum;
            out_batch[o] = cast<T>(val);
        }
    }
}

// 将 8 个 BF16 转换为 8 个 FP32
inline __m256 bf16_to_fp32_avx2(const uint16_t* src) {
    // 加载 128 位 (8 个 uint16)
    __m128i raw = _mm_loadu_si128((const __m128i*)src);
    // 将 16 位整数扩展为 32 位整数 (高位补 0)
    __m256i wide = _mm256_cvtepu16_epi32(raw);
    // 左移 16 位，使原始 16 位进入 FP32 的高位部分 (Exponent + Mantissa)
    return _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16));
}

// 快速求和累加
inline float _mm256_reduce_add_ps(__m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// 假设 T 是 uint16_t (代表 __bf16)
static void gate_up_swiglu_bf16_avx2(
    uint16_t* out, const uint16_t* in, const uint16_t* gate_weight, const uint16_t* up_weight,
    size_t batch_size, size_t in_features, size_t out_features
) {
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        const uint16_t* in_ptr = in + b * in_features;
        uint16_t* out_ptr = out + b * out_features;

        for (size_t o = 0; o < out_features; o++) {
            const uint16_t* gw_ptr = gate_weight + o * in_features;
            const uint16_t* uw_ptr = up_weight + o * in_features;

            __m256 v_gate_sum = _mm256_setzero_ps();
            __m256 v_up_sum = _mm256_setzero_ps();

            size_t i = 0;
            for (; i + 7 < in_features; i += 8) {
                __m256 v_in = bf16_to_fp32_avx2(in_ptr + i);
                __m256 v_gw = bf16_to_fp32_avx2(gw_ptr + i);
                __m256 v_uw = bf16_to_fp32_avx2(uw_ptr + i);

                // 使用 FMA 指令加速点积
                v_gate_sum = _mm256_fmadd_ps(v_in, v_gw, v_gate_sum);
                v_up_sum = _mm256_fmadd_ps(v_in, v_uw, v_up_sum);
            }

            float gate_sum = _mm256_reduce_add_ps(v_gate_sum);
            float up_sum = _mm256_reduce_add_ps(v_up_sum);

            // 处理尾部数据
            for (; i < in_features; i++) {
                // 模拟 BF16 转 FP32: 位移或强制转换
                auto bf16_to_f32 = [](uint16_t val) {
                    uint32_t res = (uint32_t)val << 16;
                    return *(float*)&res;
                };
                gate_sum += bf16_to_f32(in_ptr[i]) * bf16_to_f32(gw_ptr[i]);
                up_sum += bf16_to_f32(in_ptr[i]) * bf16_to_f32(uw_ptr[i]);
            }

            // Swish + Up
            float silu_gate = gate_sum / (1.0f + std::exp(-gate_sum));
            float final_val = silu_gate * up_sum;

            // FP32 转回 BF16: 简单截断或舍入 (此处采用简单截断)
            uint32_t res_u32 = *(uint32_t*)&final_val;
            out_ptr[o] = (uint16_t)(res_u32 >> 16);
        }
    }
}

namespace jllm::ops::cpu {
    void gate_up_swiglu(
        std::byte* out,
        const std::byte* in,
        const std::byte* gate_weight,
        const std::byte* up_weight,
        size_t batch_size,
        size_t in_features,
        size_t out_features,
        jllmDataType_t data_type
    ) {
    switch (data_type) {
    case jllmDataType_t::F32:
        return gate_up_swiglu_(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(gate_weight),
            reinterpret_cast<const float*>(up_weight),
            batch_size,
            in_features,
            out_features
        );
    case jllmDataType_t::F16:
        return gate_up_swiglu_(
            reinterpret_cast<jllm::fp16_t*>(out),
            reinterpret_cast<const jllm::fp16_t*>(in),
            reinterpret_cast<const jllm::fp16_t*>(gate_weight),
            reinterpret_cast<const jllm::fp16_t*>(up_weight),
            batch_size,
            in_features,
            out_features
        );
    case jllmDataType_t::BF16:
        return gate_up_swiglu_bf16_avx2(
            reinterpret_cast<uint16_t*>(out),
            reinterpret_cast<const uint16_t*>(in),
            reinterpret_cast<const uint16_t*>(gate_weight),
            reinterpret_cast<const uint16_t*>(up_weight),
            batch_size,
            in_features,
            out_features
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(data_type);
    }
        
}
}