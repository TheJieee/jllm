#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>
#include "../model_runner/layer/kernel/cpu/gate_up_swiglu.hpp"
#include "utils.hpp"

using namespace jllm;
using namespace jllm::ops::cpu;

// 性能测试框架
class GateUpSwiGLUBenchmark {
public:
    struct BenchmarkConfig {
        size_t batch_size = 1;
        size_t in_features = 4096;
        size_t out_features = 11008;
        size_t num_warmup = 5;
        size_t num_iterations = 10;
    };

    static void run_benchmark(const BenchmarkConfig& config) {
        std::cout << "\n=== BF16 Gate-Up-SwiGLU 性能测试 ===" << std::endl;
        std::cout << "配置:" << std::endl;
        std::cout << "  批次大小: " << config.batch_size << std::endl;
        std::cout << "  输入维度: " << config.in_features << std::endl;
        std::cout << "  输出维度: " << config.out_features << std::endl;
        std::cout << "  预热次数: " << config.num_warmup << std::endl;
        std::cout << "  测试次数: " << config.num_iterations << std::endl;

        // 分配内存
        size_t in_size = config.batch_size * config.in_features;
        size_t out_size = config.batch_size * config.out_features;
        size_t weight_size = config.out_features * config.in_features;

        std::vector<bf16_t> in(in_size);
        std::vector<bf16_t> out(out_size);
        std::vector<bf16_t> gate_weight(weight_size);
        std::vector<bf16_t> up_weight(weight_size);

        // 随机初始化
        initialize_random(in.data(), in.size());
        initialize_random(gate_weight.data(), gate_weight.size());
        initialize_random(up_weight.data(), up_weight.size());

        // 预热
        std::cout << "\n预热中..." << std::endl;
        for (size_t i = 0; i < config.num_warmup; i++) {
            gate_up_swiglu(
                reinterpret_cast<std::byte*>(out.data()),
                reinterpret_cast<const std::byte*>(in.data()),
                reinterpret_cast<const std::byte*>(gate_weight.data()),
                reinterpret_cast<const std::byte*>(up_weight.data()),
                config.batch_size,
                config.in_features,
                config.out_features,
                jllmDataType_t::BF16
            );
        }

        // 主要测试
        std::cout << "运行测试..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < config.num_iterations; i++) {
            gate_up_swiglu(
                reinterpret_cast<std::byte*>(out.data()),
                reinterpret_cast<const std::byte*>(in.data()),
                reinterpret_cast<const std::byte*>(gate_weight.data()),
                reinterpret_cast<const std::byte*>(up_weight.data()),
                config.batch_size,
                config.in_features,
                config.out_features,
                jllmDataType_t::BF16
            );
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 计算性能指标
        double total_ops = static_cast<double>(config.num_iterations) * 
                          config.batch_size * config.out_features * config.in_features * 2.0;  // 乘+加
        double total_elements = static_cast<double>(config.num_iterations) * out_size;
        double avg_time_ms = static_cast<double>(duration.count()) / config.num_iterations;
        double gflops = total_ops / (duration.count() * 1e6);
        double gbps = (total_elements * sizeof(bf16_t) * 3) / (duration.count() * 1e6);  // 读3份权重

        // 输出结果
        std::cout << "\n=== 结果 ===" << std::endl;
        std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
        std::cout << "平均耗时/迭代: " << avg_time_ms << " ms" << std::endl;
        std::cout << "吞吐量: " << gflops << " GFLOPS" << std::endl;
        std::cout << "内存带宽: " << gbps << " GB/s" << std::endl;

        // 理论峰值计算
        print_theoretical_peak();
    }

    static void print_theoretical_peak() {
        // 假设 3.5 GHz CPU, AVX2 (8 FP32 per cycle), 2 FMA per cycle
        double cpu_freq_ghz = 3.5;
        double peak_gflops = cpu_freq_ghz * 8 * 2;  // 56 GFLOPS for single core
        
        int num_cores = omp_get_max_threads();
        double peak_gflops_total = peak_gflops * num_cores;

        std::cout << "\n=== 理论峰值 ===" << std::endl;
        std::cout << "CPU频率: " << cpu_freq_ghz << " GHz" << std::endl;
        std::cout << "核心数: " << num_cores << std::endl;
        std::cout << "单核峰值: " << peak_gflops << " GFLOPS" << std::endl;
        std::cout << "全核峰值: " << peak_gflops_total << " GFLOPS" << std::endl;
    }

private:
    static void initialize_random(bf16_t* data, size_t size) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < size; i++) {
            data[i] = utils::cast<bf16_t>(dis(gen));
        }
    }
};

// 使用示例
int main() {
    GateUpSwiGLUBenchmark::BenchmarkConfig config{
        .batch_size = 1,
        .in_features = 4096,
        .out_features = 11008,
        .num_warmup = 5,
        .num_iterations = 10
    };

    GateUpSwiGLUBenchmark::run_benchmark(config);

    // 运行多个不同配置的测试
    std::cout << "\n\n=== 推理序列长度扩展性测试 ===" << std::endl;
    for (size_t batch : {1, 2, 4, 8}) {
        config.batch_size = batch;
        GateUpSwiGLUBenchmark::run_benchmark(config);
    }

    return 0;
}
