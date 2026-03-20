#include "sampling.hpp"

#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <vector>
#include <omp.h>

template<typename T>
static void sampling_(
    int64_t* out, T* logits,
    float temperature, int topk, float topp,
    size_t batch_size, size_t vocab_size
) {
    using jllm::utils::cast;
    static std::mt19937 gen(std::random_device{}());

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batch_size; ++b) {
        T* cur_logits = logits + b * vocab_size;
        
        // 1. 温度缩放 (Temperature Scaling)
        std::vector<std::pair<float, int>> probs(vocab_size);
        float inv_temp = 1.0f / (temperature > 0 ? temperature : 1.0f);
        
        for (int i = 0; i < (int)vocab_size; ++i) {
            probs[i] = {std::exp(cast<float>(cur_logits[i]) * inv_temp), i};
        }

        // 2. Top-K 过滤
        if (topk > 0 && topk < (int)vocab_size) {
            std::nth_element(probs.begin(), probs.begin() + topk, probs.end(), 
                             std::greater<std::pair<float, int>>());
            probs.resize(topk);
        }

        // 3. 排序以便计算 Top-P 和归一化
        std::sort(probs.begin(), probs.end(), std::greater<std::pair<float, int>>());

        // 4. 归一化概率
        float sum = 0;
        for (auto& p : probs) sum += p.first;
        for (auto& p : probs) p.first /= sum;

        // 5. Top-P (Nucleus) 过滤
        if (topp < 1.0f && topp > 0.0f) {
            float cumulative_prob = 0.0f;
            int last_idx = 0;
            for (; last_idx < (int)probs.size(); ++last_idx) {
                cumulative_prob += probs[last_idx].first;
                if (cumulative_prob >= topp) break;
            }
            probs.resize(last_idx + 1);
            // 重新归一化
            float new_sum = 0;
            for (auto& p : probs) new_sum += p.first;
            for (auto& p : probs) p.first /= new_sum;
        }

        // 6. 采样
        std::vector<float> p_values;
        for (auto& p : probs) p_values.push_back(p.first);
        std::discrete_distribution<int> dist(p_values.begin(), p_values.end());
        
        out[b] = probs[dist(gen)].second;
    }
}


JLLM_BEGIN

void ops::cpu::sampling(
    std::byte* out, std::byte* logits,
    float temperature, int topk, float topp,
    size_t batch_size, size_t vocab_size,
    jllmDataType_t dtype
){
    switch (dtype)
    {
    case jllmDataType_t::F32:
        return sampling_(
            reinterpret_cast<int64_t*>(out), reinterpret_cast<float*>(logits),
            temperature, topk, topp, batch_size, vocab_size
        );
    case jllmDataType_t::F16:
        return sampling_(
            reinterpret_cast<int64_t*>(out), reinterpret_cast<fp16_t*>(logits),
            temperature, topk, topp, batch_size, vocab_size
        );
    case jllmDataType_t::BF16:
        return sampling_(
            reinterpret_cast<int64_t*>(out), reinterpret_cast<bf16_t*>(logits),
            temperature, topk, topp, batch_size, vocab_size
        );
    default:
        throw std::invalid_argument("Unsupported data type.");
    }
}

JLLM_END