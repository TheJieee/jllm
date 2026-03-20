#include "decode.hpp"

#include "../../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <stdexcept>

template<typename T>
void decode_(
    T* out, T* q, //shape[1, nhead, head_dim]
    T* kcache, T* vcache, //shape[nkvhead, block_size, head_dim]
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_len,
    const std::vector<size_t>& block_table, size_t block_size
) {
    using namespace jllm::utils;
    std::vector<float> A(seq_len, 0);
    size_t bs = nkvhead * block_size * head_dim;
    float scale = static_cast<float>(1.0 / std::sqrt(head_dim));

    for(size_t h = 0; h < nhead; h++) {
        T* q_base = q + h * head_dim;
        size_t kv_id = h / (nhead / nkvhead);
        float max = -INFINITY;
        for(size_t i = 0; i < seq_len; i++) {
            size_t offset = block_table[i / block_size] * bs;
            size_t block_offset = i % block_size;
            T* k_base = kcache + offset + (kv_id * block_size + block_offset) * head_dim;
            float sum = 0;
            for(size_t j = 0; j < head_dim; j++) {
                sum += cast<float>(q_base[j]) * cast<float>(k_base[j]);
            }
            sum *= scale;
            A[i] = sum;
            max = std::max(max, sum);
        }
        float sum = 0;
        for(auto& val : A) {
            val = std::exp(val - max);
            sum += val;
        }
        for (auto& val : A) val /= sum;
        
        T* out_base = out + h * head_dim;
        std::vector<float> tmp_out(head_dim, 0.0f);
        for(size_t i = 0; i < seq_len; i++) {
            size_t b_idx = block_table[i / block_size];
            size_t b_offset = i % block_size;
            T* v_ptr = vcache + b_idx * (nkvhead * block_size * head_dim) + kv_id * (block_size * head_dim) + b_offset * head_dim;
            
            for(size_t d = 0; d < head_dim; d++)
                tmp_out[d] += A[i] * cast<float>(v_ptr[d]);
        }
        for(size_t d = 0; d < head_dim; d++) out_base[d] = cast<T>(tmp_out[d]);
    }
}


// 辅助函数：将 8 个 BF16 (uint16_t) 快速转为 8 个 float
inline __m256 bf168_to_f328(const uint16_t* ptr) {
    __m128i raw = _mm_loadu_si128((__m128i*)ptr); // 加载 128 bit
    __m256i i32 = _mm256_cvtepu16_epi32(raw);      // 扩展到 256 bit (uint32)
    return _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16)); // 左移 16 位即成 FP32
}

template<typename T>
void decode_optimized(
    T* out, T* q, // [nhead, head_dim]
    T* kcache, T* vcache, // [num_blocks, nkvhead, block_size, head_dim]
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_len,
    const std::vector<size_t>& block_table, size_t block_size
) {
    size_t gqa_ratio = nhead / nkvhead;
    size_t bs_stride = nkvhead * block_size * head_dim; // 一个 block 的物理跨度
    float inv_scale = static_cast<float>(1.0 / std::sqrt(head_dim));

    #pragma omp parallel for schedule(static)
    for (size_t h = 0; h < nhead; h++) {
        size_t kv_id = h / gqa_ratio;
        T* q_ptr = q + h * head_dim;
        std::vector<float> scores(seq_len);

        // --- 1. 计算 QK 点积 (AVX2) ---
        float max_score = -INFINITY;
        for (size_t i = 0; i < seq_len; i++) {
            size_t b_idx = block_table[i / block_size];
            size_t b_offset = i % block_size;
            T* k_ptr = kcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;

            __m256 sum_vec = _mm256_setzero_ps();
            for (size_t d = 0; d < head_dim; d += 8) {
                __m256 q_vec = bf168_to_f328((uint16_t*)(q_ptr + d));
                __m256 k_vec = bf168_to_f328((uint16_t*)(k_ptr + d));
                sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
            }
            
            // 水平求和 (Horizontal Sum)
            __m128 hi  = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo  = _mm256_castps256_ps128(sum_vec);
            lo = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(lo);
            __m128 sums = _mm_add_ps(lo, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float score = _mm_cvtss_f32(sums) * inv_scale;
            scores[i] = score;
            if (score > max_score) max_score = score;
        }

        // --- 2. Softmax ---
        float exp_sum = 0.0f;
        for (size_t i = 0; i < seq_len; i++) {
            scores[i] = std::exp(scores[i] - max_score);
            exp_sum += scores[i];
        }
        float inv_exp_sum = 1.0f / exp_sum;
        for (size_t i = 0; i < seq_len; i++) scores[i] *= inv_exp_sum;

        // --- 3. 计算 Weighted Sum (V) ---
        // 关键：为了访存连续，外层循环为 seq_len，内层为 head_dim
        std::vector<float> acc(head_dim, 0.0f);
        for (size_t i = 0; i < seq_len; i++) {
            float weight = scores[i];
            __m256 w_vec = _mm256_set1_ps(weight); // 广播权重
            
            size_t b_idx = block_table[i / block_size];
            size_t b_offset = i % block_size;
            T* v_ptr = vcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;

            for (size_t d = 0; d < head_dim; d += 8) {
                __m256 v_vec = bf168_to_f328((uint16_t*)(v_ptr + d));
                __m256 acc_vec = _mm256_loadu_ps(&acc[d]);
                acc_vec = _mm256_fmadd_ps(w_vec, v_vec, acc_vec);
                _mm256_storeu_ps(&acc[d], acc_vec);
            }
        }

        // 写回结果并转回 T
        for (size_t d = 0; d < head_dim; d++) {
            out[h * head_dim + d] = jllm::utils::cast<T>(acc[d]); 
        }
    }
}

void jllm::ops::cpu::decode(
    std::byte *out, std::byte *q, std::byte *kcache, std::byte *vcache, 
    size_t nhead, size_t nkvhead, size_t head_dim, size_t seq_len, 
    const std::vector<size_t> &block_table, size_t block_size,
    jllmDataType_t dtype
){
    switch (dtype) {
    case jllmDataType_t::F32:
        decode_<float>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(q),
            reinterpret_cast<float*>(kcache), reinterpret_cast<float*>(vcache),
            nhead, nkvhead, head_dim, seq_len, block_table, block_size
        );
        break;
    case jllmDataType_t::F16:
        decode_<fp16_t>(
            reinterpret_cast<fp16_t*>(out), reinterpret_cast<fp16_t*>(q),
            reinterpret_cast<fp16_t*>(kcache), reinterpret_cast<fp16_t*>(vcache),
            nhead, nkvhead, head_dim, seq_len, block_table, block_size
        );
        break;
    case jllmDataType_t::BF16:
        decode_optimized<bf16_t>(
            reinterpret_cast<bf16_t*>(out), reinterpret_cast<bf16_t*>(q),
            reinterpret_cast<bf16_t*>(kcache), reinterpret_cast<bf16_t*>(vcache),
            nhead, nkvhead, head_dim, seq_len, block_table, block_size
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for decode");
    }
}
