#include "rms_norm.hpp"

#include "utils.hpp"
#include <omp.h>
#include <immintrin.h>
#include <cmath>

// 优化的水平加法 - AVX2
static inline float hsum_avx2(__m256 v) {
    __m256 hsum = _mm256_permute2f128_ps(v, v, 1);
    hsum = _mm256_add_ps(v, hsum);
    hsum = _mm256_hadd_ps(hsum, hsum);
    hsum = _mm256_hadd_ps(hsum, hsum);
    return _mm256_cvtss_f32(hsum);
}

// Helper: 转换 BF16 到 FP32 (8 个元素)
static inline void bf16_to_float_8(float* out, const jllm::bf16_t* in) {
    const uint16_t* in_u16 = (const uint16_t*)in;
    for (int j = 0; j < 8; j++) {
        uint32_t bits = ((uint32_t)in_u16[j]) << 16;
        out[j] = *(float*)(&bits);
    }
}

// Helper: 转换 FP32 到 BF16 (8 个元素)
static inline void float_to_bf16_8(jllm::bf16_t* out, const float* in) {
    uint16_t* out_u16 = (uint16_t*)out;
    for (int j = 0; j < 8; j++) {
        uint32_t bits = *(uint32_t*)(&in[j]);
        out_u16[j] = (uint16_t)(bits >> 16);
    }
}

// RMS Norm F32 版本 - 使用 AVX2 优化
static void rms_norm_f32(
    float* output,
    const float* input,
    const float* weight,
    size_t batch_size,
    size_t feature_size,
    float epsilon
) {
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        const float* in_batch = input + b * feature_size;
        float* out_batch = output + b * feature_size;
        
        // 计算 mean(x^2)
        __m256 sum_sq = _mm256_setzero_ps();
        size_t i = 0;
        
        // 向量化计算平方和
        for (; i + 8 <= feature_size; i += 8) {
            __m256 v = _mm256_loadu_ps(in_batch + i);
            sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
        }
        
        // 水平加法
        float mean_square = hsum_avx2(sum_sq);
        
        // 处理残差
        for (; i < feature_size; i++) {
            mean_square += in_batch[i] * in_batch[i];
        }
        mean_square /= feature_size;
        
        // 计算 RMS 倒数
        float rms_inv = 1.0f / std::sqrt(mean_square + epsilon);
        __m256 rms_vec = _mm256_set1_ps(rms_inv);
        
        // 应用归一化和权重
        i = 0;
        for (; i + 8 <= feature_size; i += 8) {
            __m256 v = _mm256_loadu_ps(in_batch + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(v, rms_vec), w);
            _mm256_storeu_ps(out_batch + i, result);
        }
        
        for (; i < feature_size; i++) {
            out_batch[i] = in_batch[i] * rms_inv * weight[i];
        }
    }
}

// RMS Norm F16 版本 - 简单标量版本
static void rms_norm_f16(
    jllm::fp16_t* output,
    const jllm::fp16_t* input,
    const jllm::fp16_t* weight,
    size_t batch_size,
    size_t feature_size,
    float epsilon
) {
    using namespace jllm::utils;
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        const jllm::fp16_t* in_batch = input + b * feature_size;
        jllm::fp16_t* out_batch = output + b * feature_size;
        
        // 计算 mean(x^2)
        float mean_square = 0.0f;
        for (size_t i = 0; i < feature_size; i++) {
            float val = cast<float>(in_batch[i]);
            mean_square += val * val;
        }
        mean_square /= feature_size;
        
        // 计算 RMS 倒数
        float rms_inv = 1.0f / std::sqrt(mean_square + epsilon);
        
        // 应用归一化和权重
        for (size_t i = 0; i < feature_size; i++) {
            float val = cast<float>(in_batch[i]);
            float w = cast<float>(weight[i]);
            out_batch[i] = cast<jllm::fp16_t>(val * rms_inv * w);
        }
    }
}

// RMS Norm BF16 版本 - 优化版本（不使用 AVX512）
static void rms_norm_bf16(
    jllm::bf16_t* output,
    const jllm::bf16_t* input,
    const jllm::bf16_t* weight,
    size_t batch_size,
    size_t feature_size,
    float epsilon
) {
    using namespace jllm::utils;
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        const jllm::bf16_t* in_batch = input + b * feature_size;
        jllm::bf16_t* out_batch = output + b * feature_size;
        
        // 计算 mean(x^2) - 使用向量化加速
        __m256 sum_sq_vec = _mm256_setzero_ps();
        size_t i = 0;
        
        // 向量化计算平方和（每次处理 8 个 BF16 -> FP32）
        for (; i + 8 <= feature_size; i += 8) {
            float vals[8];
            bf16_to_float_8(vals, in_batch + i);
            
            __m256 v = _mm256_loadu_ps(vals);
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
        }
        
        // 水平加法
        float mean_square = hsum_avx2(sum_sq_vec);
        
        // 处理残差元素
        for (; i < feature_size; i++) {
            float val = cast<float>(in_batch[i]);
            mean_square += val * val;
        }
        mean_square /= feature_size;
        
        // 计算 RMS 倒数
        float rms_inv = 1.0f / std::sqrt(mean_square + epsilon);
        __m256 rms_vec = _mm256_set1_ps(rms_inv);
        
        // 应用归一化和权重
        i = 0;
        for (; i + 8 <= feature_size; i += 8) {
            float in_vals[8], w_vals[8];
            bf16_to_float_8(in_vals, in_batch + i);
            bf16_to_float_8(w_vals, weight + i);
            
            __m256 v = _mm256_loadu_ps(in_vals);
            __m256 w = _mm256_loadu_ps(w_vals);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(v, rms_vec), w);
            
            float result_f32[8];
            _mm256_storeu_ps(result_f32, result);
            float_to_bf16_8(out_batch + i, result_f32);
        }
        
        // 处理残差元素
        for (; i < feature_size; i++) {
            float val = cast<float>(in_batch[i]);
            float w = cast<float>(weight[i]);
            out_batch[i] = cast<jllm::bf16_t>(val * rms_inv * w);
        }
    }
}

namespace jllm::ops::cpu {
    void rms_norm(
        std::byte *output,
        const std::byte *input,
        const std::byte *weight,
        jllmDataType_t type,
        size_t batch_size,
        size_t feature_size,
        float epsilon
    ) {
        switch (type) {
        case jllmDataType_t::F32:
            return rms_norm_f32(
                reinterpret_cast<float *>(output),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(weight),
                batch_size,
                feature_size,
                epsilon
            );
        case jllmDataType_t::F16:
            return rms_norm_f16(
                reinterpret_cast<jllm::fp16_t *>(output),
                reinterpret_cast<const jllm::fp16_t *>(input),
                reinterpret_cast<const jllm::fp16_t *>(weight),
                batch_size,
                feature_size,
                epsilon
            );
        case jllmDataType_t::BF16:
            return rms_norm_bf16(
                reinterpret_cast<jllm::bf16_t *>(output),
                reinterpret_cast<const jllm::bf16_t *>(input),
                reinterpret_cast<const jllm::bf16_t *>(weight),
                batch_size,
                feature_size,
                epsilon
            );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}

    