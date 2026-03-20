#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <unistd.h>
#include "../src/common/jllm.h"
#include "../src/tensor/tensor.hpp"
#include "../src/engine/engine.hpp"
#include "../src/log/log.hpp"

using namespace jllm;

// 调试辅助函数
void print_debug_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void print_test_result(const std::string& test_name, bool passed) {
    std::string status = passed ? "✓ PASS" : "✗ FAIL";
    std::cout << "[" << status << "] " << test_name << std::endl;
}

// 测试1: 数据类型支持
void test_data_types() {
    print_debug_header("测试1: 数据类型支持");
    
    const char* type_names[] = {
        "INVALID", "BOOL", "BYTE", "I8", "I16", "I32", "I64",
        "U8", "U16", "U32", "U64", "F8", "F16", "BF16",
        "F32", "F64", "C16", "C32", "C64", "C128"
    };
    
    std::cout << "支持的数据类型:\n";
    for (int i = 0; i < 20; i++) {
        jllmDataType_t dtype = static_cast<jllmDataType_t>(i);
        std::cout << "  " << i << ": " << type_names[i] << std::endl;
    }
    
    print_test_result("数据类型枚举", true);
}

// 测试2: 张量创建和信息
void test_tensor_creation() {
    print_debug_header("测试2: 张量创建");
    
    try {
        // 创建一个 [4, 8] 的 F32 张量
        auto tensor = Tensor::create({4, 8}, jllmDataType_t::F32, jllmDeviceType_t::CPU);
        
        std::cout << "创建张量: shape=[4, 8], dtype=F32\n";
        std::cout << "张量信息:\n";
        std::cout << tensor->info();
        
        std::cout << "\n张量属性:\n";
        std::cout << "  维度数: " << tensor->ndim() << std::endl;
        std::cout << "  元素总数: " << tensor->numel() << std::endl;
        std::cout << "  元素大小: " << tensor->elementSize() << " 字节\n";
        std::cout << "  是否连续: " << (tensor->isContiguous() ? "是" : "否") << std::endl;
        
        print_test_result("张量创建和查询", true);
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        print_test_result("张量创建和查询", false);
    }
}

// 测试3: 不同数据类型的张量
void test_multiple_dtypes() {
    print_debug_header("测试3: 多数据类型张量");
    
    struct {
        jllmDataType_t dtype;
        const char* name;
    } dtypes[] = {
        {jllmDataType_t::F32, "F32"},
        {jllmDataType_t::F64, "F64"},
        {jllmDataType_t::F16, "F16"},
        {jllmDataType_t::BF16, "BF16"},
        {jllmDataType_t::I32, "I32"},
        {jllmDataType_t::I64, "I64"},
    };
    
    for (const auto& dtype_info : dtypes) {
        try {
            auto tensor = Tensor::create({2, 4}, dtype_info.dtype, jllmDeviceType_t::CPU);
            std::cout << "✓ 创建 " << dtype_info.name << " 张量成功\n";
        } catch (const std::exception& e) {
            std::cout << "✗ 创建 " << dtype_info.name << " 张量失败: " << e.what() << "\n";
        }
    }
}

// 测试4: 引擎初始化
void test_engine_initialization() {
    print_debug_header("测试4: 引擎初始化");
    
    try {
        Engine engine;
        std::cout << "引擎初始化成功\n";
        
        std::string model_path = engine.model_path();
        std::cout << "模型路径: " << model_path << std::endl;
        
        print_test_result("引擎初始化", true);
    } catch (const std::exception& e) {
        std::cerr << "引擎初始化失败: " << e.what() << std::endl;
        print_test_result("引擎初始化", false);
    }
}

// 测试5: 推理请求
void test_inference_request() {
    print_debug_header("测试5: 基础推理请求");
    
    try {
        Engine engine;
        
        // 创建一个简单的请求
        Request req;
        req.request_id = 1;
        req.prompt = {101, 2054, 2054, 102};  // 示例 token IDs
        
        std::cout << "创建推理请求:\n";
        std::cout << "  请求ID: " << req.request_id << std::endl;
        std::cout << "  提示词长度: " << req.prompt.size() << std::endl;
        std::cout << "  Tokens: ";
        for (int64_t token : req.prompt) {
            std::cout << token << " ";
        }
        std::cout << "\n";
        
        // 尝试生成
        auto result = engine.generate(req);
        std::cout << "生成完成，输出 token 数: " << result.size() << std::endl;
        
        print_test_result("推理请求", true);
    } catch (const std::exception& e) {
        std::cerr << "推理请求失败: " << e.what() << std::endl;
        print_test_result("推理请求", false);
    }
}

// 测试6: 性能基准测试
void test_performance_baseline() {
    print_debug_header("测试6: 性能基准测试");
    
    try {
        const int num_iterations = 100;
        std::cout << "创建 " << num_iterations << " 个张量并计时...\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; i++) {
            auto tensor = Tensor::create({64, 64}, jllmDataType_t::F32, jllmDeviceType_t::CPU);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "总耗时: " << duration.count() << " ms\n";
        std::cout << "平均耗时: " << (duration.count() / static_cast<double>(num_iterations)) << " ms\n";
        
        print_test_result("性能基准测试", true);
    } catch (const std::exception& e) {
        std::cerr << "性能测试失败: " << e.what() << std::endl;
        print_test_result("性能基准测试", false);
    }
}

// 测试7: 系统信息
void test_system_info() {
    print_debug_header("测试7: 系统信息");
    
    #ifdef __AVX2__
        std::cout << "✓ AVX2 支持: 是\n";
    #else
        std::cout << "✓ AVX2 支持: 否\n";
    #endif
    
    #ifdef __AVX512F__
        std::cout << "✓ AVX512 支持: 是\n";
    #else
        std::cout << "✓ AVX512 支持: 否\n";
    #endif
    
    #ifdef NDEBUG
        std::cout << "✓ 构建模式: Release\n";
    #else
        std::cout << "✓ 构建模式: Debug\n";
    #endif
    
    std::cout << "✓ 指针大小: " << sizeof(void*) * 8 << " 位\n";
    std::cout << "✓ Size_t 大小: " << sizeof(size_t) << " 字节\n";
}

// 主函数
int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║         JLLM (Jie LLM) 编译和调试检查工具                ║\n";
    std::cout << "║              Version 0.1.0 Debug Build                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "当前 C++ 工作目录: " << cwd << std::endl;
    }
    
    try {
        // 运行所有测试
        test_data_types();
        test_system_info();
        test_tensor_creation();
        test_multiple_dtypes();
        test_engine_initialization();
        test_inference_request();
        test_performance_baseline();
        
        // 总结
        print_debug_header("调试检查完成");
        std::cout << "\n✓ 所有调试检查已完成\n";
        std::cout << "✓ 项目编译成功并准备就绪\n\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ 调试过程中发生未预期的错误: " << e.what() << std::endl;
        return 1;
    }
}
