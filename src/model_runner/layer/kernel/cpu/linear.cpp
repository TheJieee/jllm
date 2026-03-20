#include "linear.hpp"

#include "jllm.h"
#include "utils.hpp"
#include <omp.h>
#include <immintrin.h>

// 优化的水平加法 - AVX2
static inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
// Float32 版本 - 极致优化 with AVX2 FMA
static void linear_optimized_float(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            const float* in_batch = in + b * in_features;
            float* out_batch = out + b * out_features;
            const float* weight_ = weight + o * in_features;
            __m256 acc = _mm256_setzero_ps();
            
            size_t i = 0;
            // 向量化部分：处理 8 个元素
            for (; i + 8 <= in_features; i += 8) {
                __m256 v = _mm256_loadu_ps(in_batch + i);
                __m256 w = _mm256_loadu_ps(weight_ + i);
                acc = _mm256_fmadd_ps(v, w, acc);
            }
            
            // 优化的水平加法
            float sum = hsum_avx2(acc);
            sum += bias ? bias[o] : 0.0f;
            
            // 处理剩余元素
            for (; i < in_features; i++) {
                sum += in_batch[i] * weight_[i];
            }
            out_batch[o] = sum;
        }
    }
}

// 批量转换 bf16 到 float 的辅助函数
static inline void bf16_to_float_avx2(float* out, const jllm::bf16_t* in, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i bf16_vals = _mm_loadu_si128((__m128i*)(in + i));
        __m256i bf16_expanded = _mm256_cvtepu16_epi32(bf16_vals);
        __m256i float_bits = _mm256_slli_epi32(bf16_expanded, 16);
        __m256 result = _mm256_castsi256_ps(float_bits);
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < count; i++) {
        out[i] = jllm::utils::cast<float>(in[i]);
    }
}

static void linear_optimized_bf16(
    jllm::bf16_t* out,
    const jllm::bf16_t* in,
    const jllm::bf16_t* weight,
    const jllm::bf16_t* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    using namespace jllm::utils;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            const uint16_t* in_ptr = reinterpret_cast<const uint16_t*>(in + b * in_features);
            const uint16_t* weight_ptr = reinterpret_cast<const uint16_t*>(weight + o * in_features);
            
            __m256 acc = _mm256_setzero_ps();
            size_t i = 0;

            // 向量化：每次处理 8 个 bf16 (刚好填满一个 __m256)
            for (; i + 8 <= in_features; i += 8) {
                // 1. 加载 8 个 uint16 到 128 位寄存器
                __m128i v_raw = _mm_loadu_si128((__m128i*)(in_ptr + i));
                __m128i w_raw = _mm_loadu_si128((__m128i*)(weight_ptr + i));

                // 2. 将 uint16 零扩展到 uint32 (8个)
                __m256i v32 = _mm256_cvtepu16_epi32(v_raw);
                __m256i w32 = _mm256_cvtepu16_epi32(w_raw);

                // 3. 左移 16 位得到 FP32 格式 (BF16 的定义就是 FP32 的高16位)
                __m256 v_fp32 = _mm256_castsi256_ps(_mm256_slli_epi32(v32, 16));
                __m256 w_fp32 = _mm256_castsi256_ps(_mm256_slli_epi32(w32, 16));

                // 4. FMA 累加
                acc = _mm256_fmadd_ps(v_fp32, w_fp32, acc);
            }

            // 水平求和 + Bias
            float sum = hsum_avx2(acc);
            if (bias) sum += cast<float>(bias[o]);

            // 剩余标量部分处理
            for (; i < in_features; i++) {
                sum += cast<float>(in[b * in_features + i]) * cast<float>(weight[o * in_features + i]);
            }

            out[b * out_features + o] = cast<jllm::bf16_t>(sum);
        }
    }
}

// 标量版本
template <typename T>
static void linear_(
    T* out,
    const T* in,
    const T* weight,
    const T* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    using namespace jllm::utils;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t o = 0; o < out_features; o++) {
            const T* in_batch = in + b * in_features;
            T* out_batch = out + b * out_features;
            float sum = bias ? cast<float>(bias[o]) : 0.0f;
            const T* weight_ = weight + o * in_features;
            
            for (size_t i = 0; i < in_features; i++) {
                sum += cast<float>(in_batch[i]) * cast<float>(weight_[i]);
            }
            out_batch[o] = cast<T>(sum);
        }
    }
}


namespace jllm::ops::cpu {
void linear(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight,
    const std::byte* bias,
    size_t batch_size,
    size_t in_features,
    size_t out_features,
    jllmDataType_t data_type
) {
    switch (data_type) {
    case jllmDataType_t::F32:
        return linear_optimized_float(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            reinterpret_cast<const float*>(bias),
            batch_size,
            in_features,
            out_features
        );
    case jllmDataType_t::F16:
        return linear_(
            reinterpret_cast<jllm::fp16_t*>(out),
            reinterpret_cast<const jllm::fp16_t*>(in),
            reinterpret_cast<const jllm::fp16_t*>(weight),
            reinterpret_cast<const jllm::fp16_t*>(bias),
            batch_size,
            in_features,
            out_features
        );
    case jllmDataType_t::BF16:
        return linear_optimized_bf16(
            reinterpret_cast<jllm::bf16_t*>(out),
            reinterpret_cast<const jllm::bf16_t*>(in),
            reinterpret_cast<const jllm::bf16_t*>(weight),
            reinterpret_cast<const jllm::bf16_t*>(bias),
            batch_size,
            in_features,
            out_features
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(data_type);
    }
        
}
}