# Paged Attention Prefill 实现文档

## 概述
为 jLLM 项目实现了完整的 paged attention prefill 功能，支持多种数据类型和CPU优化。

## 文件位置
- **头文件**: `src/model_runner/attention/cpu/prefill.hpp`
- **实现**: `src/model_runner/attention/cpu/prefill.cpp`

## 实现架构

### 1. 通用版本 - `prefill_generic<T>()`
**用途**: 为任意数据类型提供基础实现

**特性**:
```cpp
template<typename T>
static void prefill_generic(
    T *out, T *q,                          // 输出和查询
    T *kcache, T *vcache,                 // KV缓存
    size_t nhead, size_t nkvhead,         // 头数
    size_t head_dim,                      // 头维度
    size_t seq_begin, size_t seq_end,     // 序列范围
    const std::vector<size_t> &block_table, // 块表
    size_t block_size                     // 块大小
)
```

**算法流程**:
1. 对每个attention头 h:
   - 对序列中每个位置 pos ∈ [seq_begin, seq_end]:
     - 计算 Q_pos 与 K_{0..pos} 的点积 (QK^T)
     - 应用缩放因子 (1/√head_dim)
     - 使用max-trick进行softmax计算(数值稳定)
     - 加权求和对应的V值
     - 写回输出

**优化点**:
- 支持GQA (Group Query Attention): `kv_id = h / (nhead/nkvhead)`
- Paged attention支持: 通过 `block_table` 进行块位置映射
- 使用 `std::vector<float>` 进行精度累积

### 2. FP16 普通版本 - `prefill_f16()`
直接调用 `prefill_generic<fp16_t>()`，提供基础实现。

### 3. BF16 优化版本 - `prefill_bf16_optimized()`
**CPU优化策略**:

```cpp
// AVX2向量化处理
inline __m256 bf168_to_f328(const uint16_t* ptr)  // BF16→FP32转换
inline void f328_to_bf168(__m256 fvec, uint16_t* ptr)  // FP32→BF16转换
```

**优化技术**:
- **SIMD向量化**: 使用AVX2进行8个FP32的并行处理
- **FMA指令**: `_mm256_fmadd_ps()` 用于快速乘法累加
- **并行化**: OpenMP `collapse(2)` 对 (seq_idx, head) 进行并行化
- **缓存优化**: 局部 `vector<float>` 累加器避免频繁内存访问

**关键优化**:
```cpp
#pragma omp parallel for collapse(2) schedule(static)
for (size_t seq_idx = 0; seq_idx < seq_len; seq_idx++) {
    for (size_t h = 0; h < nhead; h++) {
        // 每个线程独立处理一个(seq_idx, head)对
        // 避免线程同步开销
    }
}
```

### 4. Float32 优化版本 - `prefill_f32_optimized()`
**特性**:
- 直接使用 `_mm256_loadu_ps()` 加载FP32
- 与BF16版本类似的优化策略
- 更高的精度，适合某些应用场景

### 5. 分发逻辑 - `ops::cpu::prefill()`
根据数据类型分发到对应的实现:

```cpp
switch (dtype) {
    case jllmDataType_t::F32:
        prefill_f32_optimized(...);
        break;
    case jllmDataType_t::F16:
        prefill_f16(...);
        break;
    case jllmDataType_t::BF16:
        prefill_bf16_optimized(...);
        break;
    default:
        throw std::invalid_argument("Unsupported dtype");
}
```

## 性能优化总结

| 版本 | 优化方法 | 场景 |
|------|---------|------|
| generic | 无特殊优化 | 基础参考实现 |
| f16 | 精简型 | FP16推理基线 |
| bf16 | AVX2+FMA+OpenMP | 高性能计算 |
| f32 | AVX2+FMA+OpenMP | 精度优先 |

## 内存布局

### 输入输出
- `q`: shape `[seq_len, nhead, head_dim]`
- `out`: shape `[seq_len, nhead, head_dim]`

### KV缓存（Paged）
- `kcache, vcache`: shape `[num_blocks, nkvhead, block_size, head_dim]`
- `block_table`: 长度 >= `(seq_end + block_size - 1) / block_size`

### 块寻址公式
对于位置 pos:
```cpp
block_idx = block_table[pos / block_size]
block_offset = pos % block_size
address = block_idx * (nkvhead * block_size * head_dim) 
        + kv_id * (block_size * head_dim)
        + block_offset * head_dim
```

## 数值稳定性

采用**Max-Trick**确保Softmax稳定性:
```cpp
// 1. 先找最大分数
float max_score = max(all scores)

// 2. 使用max-trick计算softmax
for i: exp_i = exp(score_i - max_score)
norm = sum(exp_i)
attention_i = exp_i / norm
```

这避免了指数溢出问题。

## 编译要求

### 需要的编译器标志
```bash
-std=c++17
-mavx2        # AVX2指令集
-mfma         # FMA指令集
-fopenmp      # OpenMP支持
```

### 编译验证
```bash
g++ -std=c++17 -I./src -I./src/common -O3 \
    -mavx2 -mfma -fPIC \
    -c src/model_runner/attention/cpu/prefill.cpp \
    -o prefill.o
```

## 支持的配置

- ✅ **数据类型**: FP32, FP16, BF16
- ✅ **Attention机制**: 标准Multi-Head/Group Query Attention
- ✅ **Paged KV缓存**: 动态块管理
- ✅ **并行化**: OpenMP多线程
- ✅ **SIMD优化**: AVX2+FMA

## 与Decode的比较

| 特性 | Prefill | Decode |
|------|---------|--------|
| 输入序列长度 | seq_end - seq_begin | 1 (单个位置) |
| 访问模式 | 三角形（可并行化） | 顺序（难以并行化） |
| 内存访问 | 连续性优化/缓存友好 | 随机访问 |
| 适用场景 | 初始编码 | 自回归解码 |

## 已知限制

1. 不支持Flash Attention变体（可后续扩展）
2. 块大小必须为常数（动态大小需修改寻址逻辑）
3. 暂不支持多GPU分布式

## 后续优化方向

1. **Flash Attention**: I/O高效算法，减少内存读写
2. **int8量化**: 进一步降低内存带宽需求
3. **NUMA感知**: 在NUMA系统上优化数据局部性
4. **自适应块大小**: 根据硬件选择最优块大小
