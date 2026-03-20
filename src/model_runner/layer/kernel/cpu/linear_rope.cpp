#include "linear_rope.hpp"

#include "jllm.h"
#include "utils.hpp"
#include <omp.h>
#include <immintrin.h>


// AVX2 水平求和
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

// Float32 极致优化版本 - AVX2
static void linear_rope_optimized_float(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const int64_t* pos_ids,
    const float* rope_table,
    size_t batch_size,
    size_t nhead,
    size_t in_features,
    size_t out_features
) {
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size * nhead; b++) {
        const float* in_batch = in + b * in_features;
        float* out_batch = out + b * out_features;

        // 矩阵乘法优化 - 每个输出元素
        for (size_t o = 0; o < out_features; o++) {
            const float* weight_ = weight + o * in_features;
            __m256 acc = _mm256_setzero_ps();
            
            size_t i = 0;
            // 向量化部分：处理 8 个元素
            for (; i + 8 <= in_features; i += 8) {
                __m256 v = _mm256_loadu_ps(in_batch + i);
                __m256 w = _mm256_loadu_ps(weight_ + i);
                acc = _mm256_fmadd_ps(v, w, acc);
            }
            
            // 使用优化的水平加法
            float sum = hsum_avx2(acc);
            sum += bias ? bias[o] : 0.0f;
            
            // 处理剩余元素
            for (; i < in_features; i++) {
                sum += in_batch[i] * weight_[i];
            }
            out_batch[o] = sum;
        }

        // RoPE 旋转位置编码优化 - 使用 SSE128 处理 4 对
        size_t seq_pos = pos_ids[b / nhead];
        size_t half = out_features / 2;
        
        size_t i = 0;
        for (; i + 4 <= half; i += 4) {
            // 加载 x1[i:i+4] 和 x2[i:i+4]
            __m128 x1_vals = _mm_loadu_ps(&out_batch[i]);
            __m128 x2_vals = _mm_loadu_ps(&out_batch[i + half]);
            
            // 加载 rope 表中的 cos/sin 值：[cos0, sin0, cos1, sin1, cos2, sin2, cos3, sin3]
            const float* rope_ptr = &rope_table[seq_pos * out_features + i * 2];
            
            // 提取 cos 值：cos0, cos1, cos2, cos3 （索引 0, 2, 4, 6）
            __m128 cos_vals = _mm_setr_ps(rope_ptr[0], rope_ptr[2], rope_ptr[4], rope_ptr[6]);
            // 提取 sin 值：sin0, sin1, sin2, sin3 （索引 1, 3, 5, 7）
            __m128 sin_vals = _mm_setr_ps(rope_ptr[1], rope_ptr[3], rope_ptr[5], rope_ptr[7]);
            
            // 计算 RoPE: out1 = x1*cos - x2*sin
            __m128 out1 = _mm_sub_ps(_mm_mul_ps(x1_vals, cos_vals), _mm_mul_ps(x2_vals, sin_vals));
            // 计算 RoPE: out2 = x1*sin + x2*cos
            __m128 out2 = _mm_add_ps(_mm_mul_ps(x1_vals, sin_vals), _mm_mul_ps(x2_vals, cos_vals));
            
            // 存储结果
            _mm_storeu_ps(&out_batch[i], out1);
            _mm_storeu_ps(&out_batch[i + half], out2);
        }
        
        // 处理剩余的标量部分
        for (; i < half; i++) {
            float cos_val = rope_table[seq_pos * out_features + i * 2];
            float sin_val = rope_table[seq_pos * out_features + i * 2 + 1];
            float x1 = out_batch[i];
            float x2 = out_batch[i + half];
            out_batch[i] = x1 * cos_val - x2 * sin_val;
            out_batch[i + half] = x1 * sin_val + x2 * cos_val;
        }
    }
}

// 1. BF16 向量加载并解包为 FP32 (处理 8 个元素)
static inline __m256 load_bf16_as_fp32(const jllm::bf16_t* ptr) {
    __m128i raw = _mm_loadu_si128((__m128i*)ptr); // 加载 8 个 uint16
    __m256i u32 = _mm256_cvtepu16_epi32(raw);     // 扩展到 8 个 uint32
    return _mm256_castsi256_ps(_mm256_slli_epi32(u32, 16)); // 左移 16 位变 FP32
}

// 2. FP32 向量压缩并存储为 BF16 (处理 8 个元素)
static inline void store_fp32_as_bf16(jllm::bf16_t* ptr, __m256 f32_vec) {
    __m256i i32 = _mm256_castps_si256(f32_vec);
    // 简单的截断转换（若需更高精度可实现 Round-to-nearest-even）
    __m256i shifted = _mm256_srli_epi32(i32, 16); 
    __m128i lo = _mm256_castsi256_si128(shifted);
    __m128i hi = _mm256_extracti128_si256(shifted, 1);
    __m128i packed = _mm_packus_epi32(lo, hi); // 重新打包为 8 个 uint16
    _mm_storeu_si128((__m128i*)ptr, packed);
}

void linear_rope_optimized_bf16(
    jllm::bf16_t* out,           // [batch_size, nhead, out_features]
    const jllm::bf16_t* in,      // [batch_size, nhead, in_features]
    const jllm::bf16_t* weight,  // [out_features, in_features]
    const jllm::bf16_t* bias,    // [out_features]
    const int64_t* pos_ids,      // [batch_size]
    const float* rope_table,     // [max_pos, out_features], 布局: [cos...sin...]
    size_t batch_size,
    size_t nhead,
    size_t in_features,
    size_t out_features
) {
    using namespace jllm::utils;

    // 并行化处理每个 Batch 和每个 Head
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t bh_idx = b * nhead + h;
            const jllm::bf16_t* in_ptr = in + bh_idx * in_features;
            jllm::bf16_t* out_ptr = out + bh_idx * out_features;
            
            // --- 第一阶段：线性层计算 (Matrix-Vector Multiplication) ---
            for (size_t o = 0; o < out_features; o++) {
                const jllm::bf16_t* w_row = weight + o * in_features;
                __m256 acc = _mm256_setzero_ps();
                
                size_t i = 0;
                for (; i + 8 <= in_features; i += 8) {
                    __m256 v = load_bf16_as_fp32(in_ptr + i);
                    __m256 w = load_bf16_as_fp32(w_row + i);
                    acc = _mm256_fmadd_ps(v, w, acc);
                }
                
                float sum = hsum_avx2(acc);
                // 标量收尾
                for (; i < in_features; i++) {
                    sum += cast<float>(in_ptr[i]) * cast<float>(w_row[i]);
                }
                if (bias) sum += cast<float>(bias[o]);
                out_ptr[o] = cast<jllm::bf16_t>(sum);
            }

            // --- 第二阶段：RoPE 旋转位置编码 ---
            size_t seq_pos = (size_t)pos_ids[b];
            size_t half = out_features / 2;
            const float* cur_rope_cos = rope_table + seq_pos * out_features;
            const float* cur_rope_sin = cur_rope_cos + half; // 对应 [all_cos, all_sin] 布局

            size_t r = 0;
            for (; r + 8 <= half; r += 8) {
                // 加载前半段 x1 和后半段 x2
                __m256 x1 = load_bf16_as_fp32(out_ptr + r);
                __m256 x2 = load_bf16_as_fp32(out_ptr + r + half);

                // 加载预计算好的 cos 和 sin
                __m256 cos_v = _mm256_loadu_ps(cur_rope_cos + r);
                __m256 sin_v = _mm256_loadu_ps(cur_rope_sin + r);

                // 旋转逻辑: 
                // r1 = x1 * cos - x2 * sin
                // r2 = x1 * sin + x2 * cos
                __m256 res1 = _mm256_fmsub_ps(x1, cos_v, _mm256_mul_ps(x2, sin_v));
                __m256 res2 = _mm256_fmadd_ps(x1, sin_v, _mm256_mul_ps(x2, cos_v));

                // 写回
                store_fp32_as_bf16(out_ptr + r, res1);
                store_fp32_as_bf16(out_ptr + r + half, res2);
            }

            // 标量收尾 (处理 half 不被 8 整除的情况)
            for (; r < half; r++) {
                float x1 = cast<float>(out_ptr[r]);
                float x2 = cast<float>(out_ptr[r + half]);
                float cos_val = cur_rope_cos[r];
                float sin_val = cur_rope_sin[r];
                out_ptr[r] = cast<jllm::bf16_t>(x1 * cos_val - x2 * sin_val);
                out_ptr[r + half] = cast<jllm::bf16_t>(x1 * sin_val + x2 * cos_val);
            }
        }
    }
}

void linear_rope_naive_bf16(
    jllm::bf16_t* out,           
    const jllm::bf16_t* in,      // 形状 [batch_size, hidden_size]
    const jllm::bf16_t* weight,  // 形状 [nhead * head_dim, hidden_size]
    const jllm::bf16_t* bias,    
    const int64_t* pos_ids,      
    const float* rope_table,     
    size_t batch_size,
    size_t nhead,
    size_t in_features,          // 即 hidden_size (3072)
    size_t out_features          // 即 head_dim (128)
) {
    for (size_t b = 0; b < batch_size; b++) {
        // 每个 Token (b) 的输入起始地址是固定的，不随 Head 变化
        const jllm::bf16_t* in_token_ptr = in + b * in_features; 

        for (size_t h = 0; h < nhead; h++) {
            // 每个 Head (h) 的输出起始地址
            jllm::bf16_t* out_ptr = out + (b * nhead + h) * out_features;
            
            // --- 第一阶段：线性层 (Q 或 K 或 V) ---
            for (size_t o = 0; o < out_features; o++) {
                // 核心修正：权重行索引 = (当前 Head 索引 * 每个 Head 的维度) + 当前维度索引
                size_t weight_row_idx = h * out_features + o;
                const jllm::bf16_t* w_row = weight + weight_row_idx * in_features;
                
                float sum = 0.0f;
                for (size_t i = 0; i < in_features; i++) {
                    sum += jllm::utils::cast<float>(in_token_ptr[i]) * jllm::utils::cast<float>(w_row[i]);
                }
                if (bias) sum += jllm::utils::cast<float>(bias[weight_row_idx]);
                out_ptr[o] = jllm::utils::cast<jllm::bf16_t>(sum);
            }

            // --- 第二阶段：RoPE ---
            size_t seq_pos = (size_t)pos_ids[b];
            size_t half = out_features / 2;
            const float* cur_rope_cos = rope_table + seq_pos * out_features;
            const float* cur_rope_sin = cur_rope_cos + half;

            for (size_t r = 0; r < half; r++) {
                float x1 = jllm::utils::cast<float>(out_ptr[r]);
                float x2 = jllm::utils::cast<float>(out_ptr[r + half]);
                
                float res1 = x1 * cur_rope_cos[r] - x2 * cur_rope_sin[r];
                float res2 = x1 * cur_rope_sin[r] + x2 * cur_rope_cos[r];
                
                out_ptr[r] = jllm::utils::cast<jllm::bf16_t>(res1);
                out_ptr[r + half] = jllm::utils::cast<jllm::bf16_t>(res2);
            }
        }
    }
}

namespace jllm::ops::cpu {
void linear_rope(
    std::byte* out,
    const std::byte* in,
    const std::byte* weight,
    const std::byte* bias,
    const int64_t* pos_ids, 
    const float* rope_table,
    size_t batch_size,
    size_t nhead,
    size_t in_features,
    size_t out_features,
    jllmDataType_t data_type
) {
    switch (data_type) {
    case jllmDataType_t::F32:
        return linear_rope_optimized_float(
            reinterpret_cast<float*>(out),
            reinterpret_cast<const float*>(in),
            reinterpret_cast<const float*>(weight),
            reinterpret_cast<const float*>(bias),
            pos_ids,
            rope_table,
            batch_size,
            nhead,
            in_features,
            out_features
        );
    case jllmDataType_t::BF16:
        return linear_rope_naive_bf16(
            reinterpret_cast<jllm::bf16_t*>(out),
            reinterpret_cast<const jllm::bf16_t*>(in),
            reinterpret_cast<const jllm::bf16_t*>(weight),
            reinterpret_cast<const jllm::bf16_t*>(bias),
            pos_ids,
            rope_table,
            batch_size,
            nhead,
            in_features,
            out_features
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(data_type);
    }
        
}
}