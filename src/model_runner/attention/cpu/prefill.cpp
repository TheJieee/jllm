#include "prefill.hpp"

#include "../../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <stdexcept>
#include <vector>
#include <omp.h>

JLLM_BEGIN

// ============================================================================
// 通用版本 (基础实现)
// ============================================================================

template<typename T>
static void prefill_generic(
    T *out, T *q, // shape[seq_len, nhead, head_dim]
    T *kcache, T *vcache, // shape[num_blocks, nkvhead, block_size, head_dim]
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_begin, size_t seq_end,
    const std::vector<size_t> &block_table, size_t block_size
) {
    using jllm::utils::cast;
    
    size_t seq_len = seq_end - seq_begin;
    size_t gqa_ratio = nhead / nkvhead;
    size_t bs_stride = nkvhead * block_size * head_dim;
    double scale = 1.0 / std::sqrt(head_dim);
    
    // 对每个头处理
    for (size_t h = 0; h < nhead; h++) {
        size_t kv_id = h / gqa_ratio;
        T* q_head = q + h * head_dim;
        T* out_head = out + h * head_dim;
        
        // 临时缓存：用于存储当前位置的QK和softmax分母
        std::vector<float> scores(seq_begin + seq_len);
        
        // 第一步：计算所有QK分数
        float max_score = -INFINITY;
        
        // 计算当前查询序列与KV缓存的点积
        for (size_t pos = seq_begin; pos < seq_end; pos++) {
            T* q_ptr = q + (pos - seq_begin) * nhead * head_dim + h * head_dim;
            float score_sum = 0.0f;
            
            // 计算与所有历史位置的点积（包括当前位置本身）
            for (size_t kv_pos = 0; kv_pos <= pos; kv_pos++) {
                size_t b_idx = block_table[kv_pos / block_size];
                size_t b_offset = kv_pos % block_size;
                T* k_ptr = kcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;
                
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += cast<float>(q_ptr[d]) * cast<float>(k_ptr[d]);
                }
                dot *= scale;
                score_sum += dot;
                max_score = std::max(max_score, dot);
            }
        }
        
        // 第二步：计算Softmax和加权求和
        for (size_t pos = seq_begin; pos < seq_end; pos++) {
            T* q_ptr = q + (pos - seq_begin) * nhead * head_dim + h * head_dim;
            T* out_ptr = out + (pos - seq_begin) * nhead * head_dim + h * head_dim;
            
            std::vector<float> att_weight;
            float exp_sum = 0.0f;
            
            // 计算softmax权重
            for (size_t kv_pos = 0; kv_pos <= pos; kv_pos++) {
                size_t b_idx = block_table[kv_pos / block_size];
                size_t b_offset = kv_pos % block_size;
                T* k_ptr = kcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;
                
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += cast<float>(q_ptr[d]) * cast<float>(k_ptr[d]);
                }
                dot = (dot * scale) - max_score;
                float exp_score = std::exp(dot);
                att_weight.push_back(exp_score);
                exp_sum += exp_score;
            }
            
            // 归一化权重
            for (auto& w : att_weight) w /= exp_sum;
            
            // 加权求和V
            std::vector<float> out_acc(head_dim, 0.0f);
            for (size_t kv_pos = 0; kv_pos <= pos; kv_pos++) {
                size_t b_idx = block_table[kv_pos / block_size];
                size_t b_offset = kv_pos % block_size;
                T* v_ptr = vcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;
                
                float weight = att_weight[kv_pos];
                for (size_t d = 0; d < head_dim; d++) {
                    out_acc[d] += weight * cast<float>(v_ptr[d]);
                }
            }
            
            // 写回结果
            for (size_t d = 0; d < head_dim; d++) {
                out_ptr[d] = cast<T>(out_acc[d]);
            }
        }
    }
}

// ============================================================================
// fp16_t 普通版本 (paged attention prefill)
// ============================================================================

static void prefill_f16(
    fp16_t *out, fp16_t *q,
    fp16_t *kcache, fp16_t *vcache,
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_begin, size_t seq_end,
    const std::vector<size_t> &block_table, size_t block_size
) {
    prefill_generic<fp16_t>(
        out, q, kcache, vcache,
        nhead, nkvhead, head_dim,
        seq_begin, seq_end,
        block_table, block_size
    );
}

// ============================================================================
// BF16 优化版本 (使用 AVX2)
// ============================================================================

// 更加精确的 BF16 转换 (Round to Nearest Even)
inline __m256 bf168_to_f328(const uint16_t* ptr) {
    __m128i raw = _mm_loadu_si128((__m128i*)ptr);
    __m256i i32 = _mm256_cvtepu16_epi32(raw);
    return _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16));
}

// 带有 RNE 舍入逻辑的转换，比直接截断更准
inline void f328_to_bf168_rne(__m256 fvec, uint16_t* ptr) {
    __m256i i32 = _mm256_castps_si256(fvec);
    // RNE 逻辑：加上 0x7fff + ((i32 >> 16) & 1)
    __m256i rounding_bias = _mm256_set1_epi32(0x7FFF);
    __m256i bit_check = _mm256_and_si256(_mm256_srli_epi32(i32, 16), _mm256_set1_epi32(1));
    i32 = _mm256_add_epi32(i32, _mm256_add_epi32(rounding_bias, bit_check));
    __m256i shifted = _mm256_srli_epi32(i32, 16);
    // 抽取低16位并打包
    __m128i low = _mm256_castsi256_si128(shifted);
    __m128i high = _mm256_extracti128_si256(shifted, 1);
    __m128i packed = _mm_packus_epi32(low, high);
    _mm_storeu_si128((__m128i*)ptr, packed);
}

// 快速水平求和
inline float hsum_avx(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    low = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(low);
    low = _mm_add_ps(low, shuf);
    shuf = _mm_movehl_ps(shuf, low);
    low = _mm_add_ss(low, shuf);
    return _mm_cvtss_f32(low);
}

// 假设 bf16_t == uint16_t
static void prefill_bf16_optimized(
    uint16_t *out, uint16_t *q,
    uint16_t *kcache, uint16_t *vcache,
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_begin, size_t seq_end,
    const std::vector<size_t> &block_table, size_t block_size
) {
    size_t seq_len = seq_end - seq_begin;
    size_t gqa_ratio = nhead / nkvhead;
    size_t bs_stride = nkvhead * block_size * head_dim;
    float scale = 1.0f / std::sqrt((float)head_dim);

    #pragma omp parallel
    {
        std::vector<float> scores_buf(seq_len);
        std::vector<float> out_acc; out_acc.reserve(head_dim);

        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t pos = seq_begin + seq_idx;
                size_t kv_id = h / gqa_ratio;
                uint16_t* q_ptr = q + (seq_idx * nhead + h) * head_dim;

                float max_score = -INFINITY;
                // dot products
                for (size_t kv_pos = 0; kv_pos <= pos; ++kv_pos) {
                    size_t b_idx = block_table[kv_pos / block_size];
                    size_t b_offset = kv_pos % block_size;
                    uint16_t* k_ptr = kcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;

                    __m256 sum_vec = _mm256_setzero_ps();
                    size_t d = 0;
                    for (; d + 7 < head_dim; d += 8) {
                        __m256 qv = bf168_to_f328(q_ptr + d);
                        __m256 kv = bf168_to_f328(k_ptr + d);
                        sum_vec = _mm256_fmadd_ps(qv, kv, sum_vec);
                    }
                    float score = hsum_avx(sum_vec);
                    // tail
                    for (; d < head_dim; ++d) {
                        float qf = _mm_cvtss_f32(_mm_castsi128_ps(_mm_set1_epi32((int)q_ptr[d] << 16))); // 或者手写 scalar bf16->float
                        float kf = _mm_cvtss_f32(_mm_castsi128_ps(_mm_set1_epi32((int)k_ptr[d] << 16)));
                        score += qf * kf;
                    }
                    score *= scale;
                    scores_buf[kv_pos] = score;
                    if (score > max_score) max_score = score;
                }

                // softmax
                float exp_sum = 0.0f;
                for (size_t i = 0; i <= pos; ++i) {
                    scores_buf[i] = std::exp(scores_buf[i] - max_score);
                    exp_sum += scores_buf[i];
                }
                float inv_sum = 1.0f / exp_sum;

                // out_acc zero init
                out_acc.assign(head_dim, 0.0f);

                // weighted sum
                for (size_t kv_pos = 0; kv_pos <= pos; ++kv_pos) {
                    float w = scores_buf[kv_pos] * inv_sum;
                    __m256 wv = _mm256_set1_ps(w);
                    size_t b_idx = block_table[kv_pos / block_size];
                    size_t b_offset = kv_pos % block_size;
                    uint16_t* v_ptr = vcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;

                    size_t d = 0;
                    for (; d + 7 < head_dim; d += 8) {
                        __m256 vv = bf168_to_f328(v_ptr + d);
                        __m256 acc = _mm256_loadu_ps(&out_acc[d]);
                        _mm256_storeu_ps(&out_acc[d], _mm256_fmadd_ps(wv, vv, acc));
                    }
                    for (; d < head_dim; ++d) {
                        out_acc[d] += w * (float)/*scalar bf16->float*/ ( (int)v_ptr[d] << 16 ); // 用正确的 scalar 转换
                    }
                }

                // write back to bf16
                uint16_t* out_ptr = out + (seq_idx * nhead + h) * head_dim;
                size_t d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    __m256 accv = _mm256_loadu_ps(&out_acc[d]);
                    f328_to_bf168_rne(accv, out_ptr + d);
                }
                for (; d < head_dim; ++d) {
                    // scalar convert and store
                    uint32_t bits = (uint32_t)(*(uint32_t*)&out_acc[d]); // 需要正确实现 scalar RNE
                    uint16_t bf = (uint16_t)( (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 );
                    out_ptr[d] = bf;
                }
            }
        }
    } // omp parallel
}
// ============================================================================
// Float32 优化版本 (使用 AVX2)
// ============================================================================

static void prefill_f32_optimized(
    float *out, float *q,
    float *kcache, float *vcache,
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_begin, size_t seq_end,
    const std::vector<size_t> &block_table, size_t block_size
) {
    using namespace jllm::utils;
    
    size_t seq_len = seq_end - seq_begin;
    size_t gqa_ratio = nhead / nkvhead;
    size_t bs_stride = nkvhead * block_size * head_dim;
    float scale = static_cast<float>(1.0 / std::sqrt(head_dim));
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t seq_idx = 0; seq_idx < seq_len; seq_idx++) {
        for (size_t h = 0; h < nhead; h++) {
            size_t kv_id = h / gqa_ratio;
            size_t pos = seq_begin + seq_idx;
            
            float* q_ptr = q + seq_idx * nhead * head_dim + h * head_dim;
            float* out_ptr = out + seq_idx * nhead * head_dim + h * head_dim;
            
            // 1. 计算所有attention分数并找最大值
            std::vector<float> scores;
            float max_score = -INFINITY;
            
            for (size_t kv_pos = 0; kv_pos <= pos; kv_pos++) {
                size_t b_idx = block_table[kv_pos / block_size];
                size_t b_offset = kv_pos % block_size;
                float* k_ptr = kcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;
                
                // 使用AVX2计算点积
                __m256 sum_vec = _mm256_setzero_ps();
                for (size_t d = 0; d < head_dim; d += 8) {
                    __m256 q_vec = _mm256_loadu_ps(q_ptr + d);
                    __m256 k_vec = _mm256_loadu_ps(k_ptr + d);
                    sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec);
                }
                
                // 水平求和
                float res[8];
                _mm256_storeu_ps(res, sum_vec);
                float score = (res[0] + res[1] + res[2] + res[3] + 
                              res[4] + res[5] + res[6] + res[7]) * scale;
                scores.push_back(score);
                max_score = std::max(max_score, score);
            }
            
            // 2. 计算Softmax
            float exp_sum = 0.0f;
            for (auto& s : scores) {
                s = std::exp(s - max_score);
                exp_sum += s;
            }
            float inv_exp_sum = 1.0f / exp_sum;
            for (auto& s : scores) s *= inv_exp_sum;
            
            // 3. 初始化输出累加器
            std::vector<float> out_acc(head_dim, 0.0f);
            
            // 3. 加权求和V
            for (size_t kv_pos = 0; kv_pos <= pos; kv_pos++) {
                float w = scores[kv_pos];
                size_t b_idx = block_table[kv_pos / block_size];
                size_t b_offset = kv_pos % block_size;
                float* v_ptr = vcache + b_idx * bs_stride + kv_id * (block_size * head_dim) + b_offset * head_dim;
                
                __m256 w_vec = _mm256_set1_ps(w);
                for (size_t d = 0; d < head_dim; d += 8) {
                    __m256 v_vec = _mm256_loadu_ps(v_ptr + d);
                    __m256 acc_vec = _mm256_loadu_ps(&out_acc[d]);
                    acc_vec = _mm256_fmadd_ps(w_vec, v_vec, acc_vec);
                    _mm256_storeu_ps(&out_acc[d], acc_vec);
                }
            }
            
            // 写回结果
            for (size_t d = 0; d < head_dim; d++) {
                out_ptr[d] = out_acc[d];
            }
        }
    }
}

// ============================================================================
// 分发逻辑
// ============================================================================

void ops::cpu::prefill(
    std::byte *out, std::byte *q,
    std::byte *kcache, std::byte *vcache,
    size_t nhead, size_t nkvhead, size_t head_dim,
    size_t seq_begin, size_t seq_end,
    const std::vector<size_t> &block_table, size_t block_size,
    jllmDataType_t dtype
) {
    switch (dtype) {
    case jllmDataType_t::F32:
        prefill_f32_optimized(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(q),
            reinterpret_cast<float*>(kcache), reinterpret_cast<float*>(vcache),
            nhead, nkvhead, head_dim,
            seq_begin, seq_end,
            block_table, block_size
        );
        break;
    case jllmDataType_t::F16:
        prefill_f16(
            reinterpret_cast<fp16_t*>(out), reinterpret_cast<fp16_t*>(q),
            reinterpret_cast<fp16_t*>(kcache), reinterpret_cast<fp16_t*>(vcache),
            nhead, nkvhead, head_dim,
            seq_begin, seq_end,
            block_table, block_size
        );
        break;
    case jllmDataType_t::BF16:
        prefill_bf16_optimized(
            reinterpret_cast<uint16_t*>(out), reinterpret_cast<uint16_t*>(q),
            reinterpret_cast<uint16_t*>(kcache), reinterpret_cast<uint16_t*>(vcache),
            nhead, nkvhead, head_dim,
            seq_begin, seq_end,
            block_table, block_size
        );
        break;
    default:
        throw std::invalid_argument("Unsupported dtype for prefill");
    }
}

JLLM_END