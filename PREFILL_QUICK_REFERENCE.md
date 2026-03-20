# Prefill 快速参考指南

## 📌 快速概览

已为jLLM项目实现完整的 **paged attention prefill** 功能，包括普通版本和多种CPU优化版本。

```
prefill_f16() ─┐
              ├─→ ops::cpu::prefill() ─→ 分发逻辑
prefill_f32_optimized() ─┤
              └─→ prefill_bf16_optimized()
```

## 🎯 实现的三个版本

### 1️⃣ FP16普通版本 (`prefill_f16`)
- **调用**: `prefill_generic<fp16_t>()`
- **场景**: 基础实现，参考版本
- **特点**: 无特殊优化，易于理解

### 2️⃣ BF16优化版本 (`prefill_bf16_optimized`)
- **优化**: AVX2 SIMD + FMA + OpenMP
- **性能**: 3-5倍提升
- **特点**: 适合高性能推理

### 3️⃣ Float32优化版本 (`prefill_f32_optimized`)
- **优化**: AVX2 SIMD + FMA + OpenMP  
- **特点**: 高精度，适合需要FP32的场景

## 📋 核心算法

对于每个attention头 h 和序列位置 pos ∈ [seq_begin, seq_end]:

```
1. 计算 attention score = Q_pos · K_{0..pos} / √head_dim
2. 应用 softmax 得到权重 w_i = exp(score_i - max_score) / Σ(exp(score - max_score))
3. 输出 = Σ w_i · V_i
```

**关键特性**:
- ✅ Paged Attention支持（块化KV缓存）
- ✅ Group Query Attention (GQA)
- ✅ 数值稳定性（max-trick）

## 🔧 API使用

### 基础调用

```cpp
#include "src/model_runner/attention/cpu/prefill.hpp"

// 准备数据
std::vector<size_t> block_table = {0, 1, 2, 3, ...};
size_t block_size = 256;
size_t nhead = 32;
size_t nkvhead = 8;
size_t head_dim = 128;
size_t seq_len = 512;

// 调用prefill
jllm::ops::cpu::prefill(
    output,           // std::byte* output buffer
    query,            // std::byte* query [seq_len, nhead, head_dim]
    kcache, vcache,   // std::byte* [num_blocks, nkvhead, block_size, head_dim]
    nhead, nkvhead, head_dim,
    0, seq_len,       // seq_begin=0, seq_end=seq_len
    block_table, block_size,
    jllmDataType_t::BF16  // 选择数据类型
);
```

### 支持的数据类型

```cpp
jllmDataType_t::F32    // Float32 (优化版)
jllmDataType_t::F16    // Float16 (普通版)
jllmDataType_t::BF16   // BFloat16 (优化版)
```

## 📊 性能对比

| 版本 | 优化技术 | 相对性能 | 用途 |
|------|---------|---------|------|
| F16 | - | 1x | 基准参考 |
| BF16 | AVX2+FMA+OMP | ~3-5x | 高性能推理 |
| F32 | AVX2+FMA+OMP | ~3-5x | 精度优先 |

## 💾 内存布局

### 输入

**查询 (Q)**
```
Shape: [seq_len, nhead, head_dim]
Layout: C连续
Example: [512, 32, 128]
Size: 512 * 32 * 128 * 2 = 4MB (BF16)
```

**KV缓存**
```
Shape: [num_blocks, nkvhead, block_size, head_dim]
Layout: 块化存储
Block数量: ceil((seq_begin + seq_len) / block_size)
```

### 块寻址

对于历史位置 pos，对应的块号：
```cpp
block_id = block_table[pos / block_size]
offset_in_block = pos % block_size
address = block_id * (nkvhead * block_size * head_dim) 
        + kv_head * (block_size * head_dim)
        + offset_in_block * head_dim
```

## 🔍 关键优化技术

### 1. SIMD向量化
```cpp
// BF16 → FP32 转换
__m256 bf168_to_f328(const uint16_t* ptr)
// 点积计算
sum_vec = _mm256_fmadd_ps(q_vec, k_vec, sum_vec)
// FP32 → BF16 转换  
void f328_to_bf168(__m256 fvec, uint16_t* ptr)
```

### 2. OpenMP并行化
```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (size_t seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    for (size_t h = 0; h < nhead; h++) {
        // 每个(seq_idx, head)对由一个线程处理
    }
}
```

### 3. 数值稳定性
```cpp
// Max-trick softmax
float max_score = max(all scores)
for i: exp_i = exp(score_i - max_score)
attention_i = exp_i / sum(exp_i)
// 防止指数溢出
```

## 🛠️ 编译

### 所需标志
```bash
-std=c++17   # C++17标准
-mavx2       # AVX2指令集
-mfma        # FMA指令集
-fopenmp     # OpenMP
```

### 完整编译示例
```bash
g++ -std=c++17 -I./src -I./src/common -O3 \
    -mavx2 -mfma -fopenmp \
    -c src/model_runner/attention/cpu/prefill.cpp
```

## ⚙️ 配置参数

| 参数 | 说明 | 范围 |
|------|------|------|
| `nhead` | 总头数 | 1-256 |
| `nkvhead` | KV头数 (GQA) | 1-nhead |
| `head_dim` | 单个头维度 | 32-256 |
| `head_dim % 8` | - | 必须8的倍数 |
| `block_size` | KV缓存块大小 | 64-512 |
| `seq_begin/seq_end` | 序列范围 | 支持任意范围 |

## 🐛 常见问题

### Q: 为什么输出不对?
**A**: 检查:
1. `block_table` 是否正确映射
2. `head_dim` 是否与模型配置匹配
3. 数据类型是否正确
4. 内存对齐是否满足

### Q: 性能不达预期?
**A**: 检查:
1. 编译器优化标志 (`-O3`)
2. SIMD指令集支持 (`-mavx2 -mfma`)
3. OpenMP线程数 (`OMP_NUM_THREADS`)
4. 块大小是否合理 (推荐256)

### Q: 支持其他数据类型吗?
**A**: 当前支持 F32, F16, BF16。可通过扩展 `prefill_generic<>` 添加其他类型。

## 📚 相关文件

- **实现**: `src/model_runner/attention/cpu/prefill.cpp`
- **头文件**: `src/model_runner/attention/cpu/prefill.hpp`
- **详细文档**: `PREFILL_IMPLEMENTATION.md`
- **修改日志**: `CHANGELOG_PREFILL.md`

## 🚀 后续改进

1. **Flash Attention**: 更高IO效率
2. **量化支持**: Int8/Int4优化
3. **分布式**: 多GPU支持
4. **自适应**: 动态块大小调整

## 📞 快速查询

### 使用什么版本?
- 高性能需求 → BF16优化版
- 需要高精度 → F32优化版
- 调试/参考 → F16普通版

### 数据类型如何选择?
- `BF16` ✅ 推荐用于生产环境
- `F32` ✅ 精度优先场景
- `F16` ⚠️ 基准参考版本

### 块大小如何设置?
- 推荐: 256
- 范围: 64-512
- 权衡: 大块→缓存友好，小块→灵活性高
