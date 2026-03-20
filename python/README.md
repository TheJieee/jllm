# JLLM Python Wrapper

Python友好的JLLM (Jintian's Large Language Model) 推理引擎封装层。

## 特性

- **简洁的Python API**: 为pybind11导出的C++接口提供Pythonic的包装
- **类型提示**: 完整的类型注解支持IDE自动补全和类型检查
- **错误处理**: 详细的错误消息和异常处理
- **异步支持**: `AsyncEngine`用于批量推理任务
- **文档完善**: 详细的类和方法文档

## 安装

1. 构建C++扩展:
```bash
cd /home/jie/learning/ai_camp_2026/jllm
xmake
```

2. Python包已在`build/lib/`中生成

## 快速开始

### 基本使用

```python
from jllm import Engine, Request

# 创建引擎
engine = Engine()

# 创建请求
request = Request(prompt=[1, 2, 3, 4, 5], request_id=1)

# 生成输出
output = engine.generate(request)
print(output)  # [token1, token2, ...]
```

### 异步批量推理

```python
from jllm import AsyncEngine, Request

# 创建异步引擎
async_engine = AsyncEngine()
async_engine.set_up()

# 推送多个请求
requests = [
    Request(prompt=[1, 2, 3], request_id=1),
    Request(prompt=[4, 5, 6], request_id=2),
]

for req in requests:
    async_engine.push(req)

# 获取结果
while async_engine.has_output():
    request_id, tokens = async_engine.get_one()
    print(f"Request {request_id}: {tokens}")

# 或一次获取所有结果
results = async_engine.get_all()
for request_id, tokens in results:
    print(f"Request {request_id}: {tokens}")
```

### 配置引擎

```python
from jllm import Config

config = Config(
    cache_num_block=256,     # 缓存块数量
    cache_block_size=2048    # 每块大小
)
```

## API 文档

### 类: Request

请求对象，包含输入token和请求ID。

**属性:**
- `prompt: List[int]` - 输入token列表
- `request_id: int` - 请求唯一标识符

**示例:**
```python
request = Request(prompt=[1, 2, 3], request_id=42)
```

### 类: Config

引擎配置对象。

**属性:**
- `cache_num_block: int` - KV缓存块数量
- `cache_block_size: int` - KV缓存块大小

### 类: Engine

同步推理引擎。

**方法:**
- `generate(request: Request) -> List[int]` - 从请求生成输出token
- `step() -> None` - 执行一步推理

### 类: AsyncEngine

异步推理引擎，支持批量请求。

**方法:**
- `set_up() -> None` - 初始化引擎（必须先调用）
- `push(request: Request) -> int` - 推送请求到队列，返回request_id
- `has_output() -> bool` - 检查是否有可用的结果
- `get_one() -> Optional[Tuple[int, List[int]]]` - 获取一个结果
- `get_all() -> List[Tuple[int, List[int]]]` - 获取所有可用结果
- `pending_count() -> int` - 获取待处理请求数

## 文件结构

```
python/
├── __init__.py              # 包初始化
├── jllm/
│   ├── __init__.py          # jllm包初始化，暴露主要接口
│   ├── _engine_wrapper.py   # C++接口的Python包装
│   ├── _jllm_engine.so      # pybind11生成的共享库
│   └── kernel.py            # 内核特定的实现
└── talk.py                  # 使用示例
```

## 开发

### 从源码安装（开发模式）

```bash
cd /home/jie/learning/ai_camp_2026/jllm
pip install -e .
```

### 运行示例

```bash
cd python
python talk.py
```

### 运行测试

```bash
python -m pytest tests/  # 如果有测试文件
```

## 故障排除

### "Failed to import _jllm_engine.so"

确保已编译C++扩展：
```bash
cd /home/jie/learning/ai_camp_2026/jllm
xmake
```

.so文件应该在`build/lib/jllm_engine.so`或通过xmake复制到`python/jllm/`。

### 导入错误

检查Python路径包含包目录：
```python
import sys
sys.path.insert(0, '/path/to/python')
```

## 贡献

欢迎提交问题和拉取请求！

## 许可证

参见项目根目录的LICENSE文件。
