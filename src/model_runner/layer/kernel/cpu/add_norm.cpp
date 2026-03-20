#include "add_norm.hpp"

#include "utils.hpp"
#include <omp.h>
#include <immintrin.h>
#include <cmath>

template <typename T>
void add_norm_(
    T* out,
    T* in,
    const T* add_tensor,
    const T* norm_weight,
    size_t batch_size,
    size_t in_features,
    float eps
) {
    using namespace jllm::utils;
    for(size_t b = 0; b < batch_size; b++) {
        T* in_batch = in + b * in_features;
        const T* add_batch = add_tensor + b * in_features;
        T* out_batch = out + b * in_features;

        // Add
        for (size_t i = 0; i < in_features; i++) {
            in_batch[i] = out_batch[i] = cast<T>(cast<float>(in_batch[i]) + cast<float>(add_batch[i]));
        }
        // RMS Norm
        float mean_square = 0.0f;
        for (size_t i = 0; i < in_features; i++) {
            float val = cast<float>(out_batch[i]);
            mean_square += val * val;
        }
        mean_square /= in_features;
        float norm_factor = 1.0f / std::sqrt(mean_square + eps);
        for (size_t i = 0; i < in_features; i++) {
            out_batch[i] = cast<T>(cast<float>(out_batch[i]) * cast<float>(norm_weight[i]) * norm_factor);
        }
    }
}



// 高性能 BF16 -> FP32 (8路)
inline __m256 bf168_to_f328(const uint16_t* ptr) {
    __m128i raw = _mm_loadu_si128((const __m128i*)ptr);
    __m256i wide = _mm256_cvtepu16_epi32(raw);
    return _mm256_castsi256_ps(_mm256_slli_epi32(wide, 16));
}

// 高性能 FP32 -> BF16 (8路，含 AVX2 Lane 修正)
inline void f328_to_bf168(uint16_t* ptr, __m256 fv) {
    __m256i i32 = _mm256_castps_si256(fv);
    // 提取高 16 位
    __m256i shifted = _mm256_srli_epi32(i32, 16);
    // 跨 Lane 打包并修正顺序 (0xD8 = 11 01 10 00)
    __m256i packed = _mm256_packus_epi32(shifted, shifted);
    __m256i permuted = _mm256_permute4x64_epi64(packed, 0xD8);
    _mm_storeu_si128((__m128i*)ptr, _mm256_castsi256_si128(permuted));
}

// AVX2 向量内累加
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

// 辅助函数：正确转换 BF16 位模式到 FP32 (标量版)
inline float bf16_to_f32_scalar(uint16_t h) {
    uint32_t f = static_cast<uint32_t>(h) << 16;
    float res;
    std::memcpy(&res, &f, sizeof(float));
    return res;
}

// 辅助函数：FP32 转换回 BF16 位模式 (标量版)
inline uint16_t f32_to_bf16_scalar(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return static_cast<uint16_t>(i >> 16);
}

template <typename T>
void add_norm_bf16_avx2_optimized(
    T* out, T* in, const T* add_tensor, const T* norm_weight,
    size_t batch_size, size_t in_features, float eps
) {
    int max_threads = omp_get_max_threads();
    // 依然需要 b_temp 来存储中间的 FP32 结果，避免第二遍 Pass 重复从内存读并转换 BF16
    float* temp_sum_all = (float*)aligned_alloc(32, max_threads * in_features * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        int tid = omp_get_thread_num();
        uint16_t* in_ptr = (uint16_t*)(in + b * in_features); // 注意：这里变为非 const，用于写回
        const uint16_t* add_ptr = (const uint16_t*)(add_tensor + b * in_features);
        const uint16_t* weight_ptr = (const uint16_t*)norm_weight;
        uint16_t* out_ptr = (uint16_t*)(out + b * in_features);
        
        float* b_temp = temp_sum_all + (tid * in_features); 

        __m256 m_sq_vec = _mm256_setzero_ps();
        size_t i = 0;

        // --- Pass 1: Add + Store Residual + SquareSum ---
        for (; i + 7 < in_features; i += 8) {
            __m256 v_in = bf168_to_f328(in_ptr + i);
            __m256 v_add = bf168_to_f328(add_ptr + i);
            __m256 v_sum = _mm256_add_ps(v_in, v_add);
            
            // 1. 写回残差 (Residual Out): in = in + add
            f328_to_bf168(in_ptr + i, v_sum);
            
            // 2. 存入临时 Buffer 供后面 Norm 使用 (避免重复转换)
            _mm256_storeu_ps(b_temp + i, v_sum);
            
            // 3. 累加平方和
            m_sq_vec = _mm256_fmadd_ps(v_sum, v_sum, m_sq_vec);
        }

        // 处理标量剩余部分
        float m_sq = _mm256_reduce_add_ps(m_sq_vec);
        for (; i < in_features; i++) {
            float s = bf16_to_f32_scalar(in_ptr[i]) + bf16_to_f32_scalar(add_ptr[i]);
            
            in_ptr[i] = f32_to_bf16_scalar(s); // 写回残差
            b_temp[i] = s;
            m_sq += s * s;
        }

        // 计算 RMS 因子
        float inv_rms = 1.0f / std::sqrt(m_sq / in_features + eps);
        __m256 v_inv_rms = _mm256_set1_ps(inv_rms);

        // --- Pass 2: Norm + Store to Out ---
        i = 0;
        for (; i + 7 < in_features; i += 8) {
            __m256 v_val = _mm256_loadu_ps(b_temp + i);
            __m256 v_w = bf168_to_f328(weight_ptr + i);
            
            // Out = (Sum * Weight) * inv_rms
            __m256 res = _mm256_mul_ps(_mm256_mul_ps(v_val, v_w), v_inv_rms);
            f328_to_bf168(out_ptr + i, res);
        }

        for (; i < in_features; i++) {
            float res_f32 = b_temp[i] * bf16_to_f32_scalar(weight_ptr[i]) * inv_rms;
            out_ptr[i] = f32_to_bf16_scalar(res_f32);
        }
    }
    free(temp_sum_all);
}

namespace jllm::ops::cpu {
    void add_norm(
        std::byte* out,
        std::byte* in,
        const std::byte* add_tensor,
        const std::byte* norm_weight,
        size_t batch_size,
        size_t in_features,
        float eps,
        jllmDataType_t data_type
    ) {
    switch (data_type) {
    case jllmDataType_t::F32:
        return add_norm_(
            reinterpret_cast<float*>(out),
            reinterpret_cast<float*>(in),
            reinterpret_cast<const float*>(add_tensor),
            reinterpret_cast<const float*>(norm_weight),
            batch_size,
            in_features,
            eps
        );
    case jllmDataType_t::F16:
        return add_norm_(
            reinterpret_cast<jllm::fp16_t*>(out),
            reinterpret_cast<jllm::fp16_t*>(in),
            reinterpret_cast<const jllm::fp16_t*>(add_tensor),
            reinterpret_cast<const jllm::fp16_t*>(norm_weight),
            batch_size,
            in_features,
            eps
        );
    case jllmDataType_t::BF16:
        return add_norm_bf16_avx2_optimized(
            reinterpret_cast<jllm::bf16_t*>(out),
            reinterpret_cast<jllm::bf16_t*>(in),
            reinterpret_cast<const jllm::bf16_t*>(add_tensor),
            reinterpret_cast<const jllm::bf16_t*>(norm_weight),
            batch_size,
            in_features,
            eps
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(data_type);
    }
}
}