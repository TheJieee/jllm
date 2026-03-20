# 🚀 JLLM-Engine

>  CPU 大模型推理引擎

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-orange.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![AVX2](https://img.shields.io/badge/SIMD-AVX2-red.svg)](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

---

## 简介

**JLLM-Engine** 是一款大语言模型推理服务系统，后端以纯 C++ 编写，通过 [pybind11](https://github.com/pybind/pybind11) 向 Python 层暴露 `Engine` 与 `AsyncEngine` 两套接口。

系统架构参考 [vLLM](https://github.com/vllm-project/vllm)，在 CPU 平台上复现并拓展了其核心调度与内存管理特性，同时融入多项面向 CPU 的专项加速优化，实现了生产级别的推理吞吐与延迟表现。

---

## 核心特性

### 🧠 内存与调度（参考 vLLM 架构）

| 特性 | 描述 |
|------|------|
| **Paged Attention** | KV Cache 以固定大小的 Page 为单位进行分配与管理，彻底消除内存碎片，大幅提升并发请求数量上限 |
| **Chunked Prefill** | 将超长 Prefill 请求拆分为若干 Chunk 分批处理，与 Decode 阶段请求混合调度，降低首 Token 延迟（TTFT），提升整体吞吐 |
| **Prefix Caching** | 对相同系统提示或历史上下文的 KV Cache 进行哈希索引与复用，命中缓存时跳过对应 Prefill 计算，显著降低重复请求的计算开销 |

### ⚡ CPU 专项加速

| 特性 | 描述 |
|------|------|
| **算子融合（Kernel Fusion）** | 将 LayerNorm、激活函数、矩阵乘等相邻算子在编译期/运行期合并为单一 Kernel，减少中间内存读写与函数调用开销 |
| **AVX2 SIMD 向量化** | 核心计算路径（GEMM、点积、Softmax 等）使用 AVX2 指令集手写或自动向量化，单核吞吐相比标量实现提升显著 |
| **多线程并行** | 基于线程池实现算子级与请求级双层并行，充分利用多核 CPU 算力；支持 NUMA 感知绑核策略，降低跨 NUMA 访存延迟 |

### 🐍 Python 接口

通过 pybind11 提供两套使用模式：

- **`Engine`** — 同步接口，适用于单次推理、调试与低延迟场景
- **`AsyncEngine`** — 异步接口，基于协程驱动的连续批处理（Continuous Batching），适用于高并发在线服务

---

## 快速开始
### 编译安装

```bash
# 克隆仓库
git clone https://github.com/TheJiee/jllm.git
cd jllm

# 编译（使用全部核心）
xmake
```
详细指南请阅读[README.md](python/README.md)
## 参与贡献

欢迎提交 Issue 与 Pull Request！ 

主要贡献方向：
- 支持gpu推理
- 新模型架构适配
- 更多 SIMD 指令集支持（AVX-512、AMX）
- INT8 / BF16 量化推理
- 分布式多机推理

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
