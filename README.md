# 🌙 DeepDream 梦境生成助手 — LangGraph 重构版

基于 **LangGraph + PyTorch** 的工程化重构版本，由千问大模型驱动的智能梦境生成助手。

> **当前版本**：`2.1.0` ｜ **Python**：`>=3.10` ｜ **License**：GPL-3.0-or-later

______________________________________________________________________

## ✨ 功能特性

- **5 种梦境生成**：正常梦、美梦、噩梦、迷雾梦、疯狂梦
- **智能对话**：基于千问大模型（`dashscope` 官方 SDK）的自然语言交互
- **知识科普**：梦境心理学 + DeepDream 算法原理
- **实用工具**：天气查询、科学计算器
- **工程化架构**：LangGraph 状态图 + 分层设计

## 🏗️ 项目结构

```
deepdream_project/
├── pyproject.toml            # 项目元数据与依赖声明（PEP 621）
├── requirements.txt          # 依赖清单（pip 直装时使用）
├── main.py                   # 主程序入口（Gradio 界面 + CLI 入口）
├── uploads/                  # 上传图片目录
├── dream_outputs/            # 梦境输出目录
└── deepdream/
    ├── __init__.py
    └── src/
        ├── __init__.py
        ├── core/             # 核心模块
        │   ├── __init__.py
        │   └── graph.py      # LangGraph 图定义
        ├── node/             # 节点实现
        │   ├── __init__.py
        │   ├── llm_node.py   # LLM 节点（千问官方 SDK）
        │   ├── tool_node.py  # 工具执行节点
        │   └── output_node.py # 输出节点
        ├── agent/            # Agent 逻辑
        │   ├── __init__.py
        │   └── agent.py      # DreamAgent 类
        ├── tools/            # 工具模块
        │   ├── __init__.py
        │   ├── weather.py    # 天气查询工具
        │   ├── calculator.py # 科学计算器工具
        │   └── dream_gen.py  # 梦境生成 + 知识科普工具
        ├── deepdream/        # DeepDream 算法（PyTorch）
        │   ├── __init__.py
        │   └── generator.py  # DeepDreamGenerator 类
        └── config/           # 配置模块
            ├── __init__.py
            └── settings.py   # 梦境配置（保留原有 5 种配置）
```

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
| --- | --- |
| Python | `>=3.10` |
| 操作系统 | Linux / macOS / Windows |
| 硬件 | NVIDIA GPU（CUDA）+ 8GB 显存可流畅运行；无 GPU 时自动降级为 CPU |
| 外部依赖 | `pytesseract` 需要系统安装 Tesseract OCR；`opencv-python` 需要系统图像库 |

### 1. 安装

```bash
git clone https://github.com/DakeGinger/deepdream-langgraph.git
cd deepdream-langgraph

# 方式 A：可编辑安装（推荐，含 CLI 入口）
pip install -e .

# 方式 B：仅安装运行依赖
pip install -r requirements.txt

# 开发依赖（测试、格式化、导入排序）
pip install -e ".[dev]"
```

### 2. 配置 API Key

在项目根目录创建 `.env`（参考 `.env.example`）：

```env
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

或在终端直接导出：

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. 运行程序

```bash
# 方式 A：pip 安装后使用 CLI 入口
deepdream

# 方式 B：模块运行
python main.py
```

浏览器访问 `http://localhost:7860`。

> **端口说明**：Gradio 默认端口为 `7860`。本项目使用 Gradio 作为 Web 界面。

## 🎨 梦境类型说明

| 类型 | 中文名称 | 特点 |
| --- | --- | --- |
| `normal` | 正常梦 | 基础效果，平衡真实与梦幻 |
| `sweet` | 美梦 | 柔和梦幻，高斯模糊滤镜 |
| `nightmare` | 噩梦 | 恐怖锐利，锐化滤镜 |
| `mist` | 迷雾梦 | 朦胧模糊，层次感强 |
| `crazy` | 疯狂梦 | 极致迷幻，视觉冲击强烈 |

## 🔧 技术栈

| 组件 | 版本 / 说明 | 来源 |
| --- | --- | --- |
| LangGraph | `>=1.2.0`，状态图编排 | `pyproject.toml` |
| LangChain | `>=1.3.0` + `langchain-core>=0.3.0` | `pyproject.toml` |
| PyTorch | `>=2.0.0` + `torchvision>=0.15.0` | `pyproject.toml` |
| DeepDream backbone | InceptionV3（预训练，约 100MB） | 首次运行自动下载 |
| LLM | `dashscope>=1.20.0`（千问官方 SDK） | `pyproject.toml` |
| Web UI | `gradio>=4.0.0` | `pyproject.toml` |
| 图像处理 | `Pillow>=9.0.0`、`numpy>=1.24.0` | `pyproject.toml` |

## 💡 使用示例

### 生成梦境

1. 上传一张图片
1. 输入：`帮我生成一个美梦`
1. 等待生成，查看结果

### 查询天气

输入：`查询北京的天气`

### 科学计算

输入：`计算 sin(pi/2) + sqrt(16) + log(e)`

### 知识科普

输入：`给我讲讲 DeepDream 算法原理`

## ⚙️ 性能说明

- **推荐配置**：RTX 3060 Mobile + R7 5800H 可流畅运行
- **GPU 加速**：自动检测 CUDA，有 GPU 时自动使用
- **CPU 模式**：无 GPU 时自动降级为 CPU 运行（速度较慢）
- **内存占用**：约 2–4GB（取决于图片大小和梦境类型）
- **瓶颈提示**：单张 800×800 图片的保存耗时仅数毫秒，真正的瓶颈在 GPU 迭代（单次约 120 秒量级）。因此图片 I/O 不值得引入异步文件库，将其丢进 `asyncio.to_thread` 或 `concurrent.futures.ThreadPoolExecutor` 避免阻塞事件循环即可。

> 以上性能数字来自单卡开发环境的粗略观测，受图片尺寸、迭代次数与梦境类型影响，不可作为承诺值引用。

## 🧪 开发工作流

代码质量工具由 `pyproject.toml` 统一配置，提交前请保持全绿。

| 工具 | 配置来源 | 说明 |
| --- | --- | --- |
| `black` | `[tool.black]` | `line-length = 100`，`target-version = ['py310']` |
| `isort` | `[tool.isort]` | `profile = "black"`，`line_length = 100` |
| `pytest` | `[project.optional-dependencies.dev]` | `pytest>=7.0` |

```bash
# 格式化
black .
isort .

# 运行测试
pytest
```

## 📝 注意事项

1. 首次运行会自动下载 InceptionV3 预训练模型（约 100MB）
1. 疯狂梦模式计算量较大，建议使用 GPU
1. 图片尺寸会自动缩放到配置的最大尺寸，以保证性能
1. 生成的图片保存在 `dream_outputs/` 目录下

## 🤝 贡献

欢迎 Issue 和 PR。提交前请确保：

- `pytest` 通过
- `black` / `isort` 无格式改动
- 新增功能附带测试

## 📄 License

本项目以 **GNU GPL v3.0 或更高版本（GPL-3.0-or-later）** 发布。

- 完整许可证文本见仓库根目录 `LICENSE` 文件
- 修改本项目代码时，须保留原作者署名并标注修改之处
- 本软件按「原样」提供，不附带任何明示或暗示的担保

______________________________________________________________________

Made with ❤️ by DakeGinger
