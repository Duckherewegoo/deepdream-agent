# 🌙 DeepDream 梦境生成助手 - LangGraph 重构版

基于 **LangGraph + PyTorch** 的工程化重构版本，使用千问大模型驱动的智能梦境生成助手。

## ✨ 功能特性

- **5种梦境生成**：正常梦、美梦、噩梦、迷雾梦、疯狂梦
- **智能对话**：基于千问大模型的自然语言交互
- **知识科普**：梦境心理学 + DeepDream 算法原理
- **实用工具**：天气查询、科学计算器
- **工程化架构**：LangGraph 状态图 + 分层设计

## 🏗️ 项目结构

```
deepdream_project/
├── project.toml              # 项目配置（pyproject.toml 格式）
├── requirements.txt          # 依赖清单
├── main.py                   # 主程序入口（Gradio 界面）
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
            └── settings.py   # 梦境配置（保留原有5种配置）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

设置环境变量：

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. 运行程序

```bash
python main.py
```

然后在浏览器中打开 `http://localhost:7860`

## 🎨 梦境类型说明

| 类型 | 中文名称 | 特点 |
|------|----------|------|
| normal | 正常梦 | 基础效果，平衡真实与梦幻 |
| sweet | 美梦 | 柔和梦幻，高斯模糊滤镜 |
| nightmare | 噩梦 | 恐怖锐利，锐化滤镜 |
| mist | 迷雾梦 | 朦胧模糊，层次感强 |
| crazy | 疯狂梦 | 极致迷幻，视觉冲击强烈 |

## 🔧 技术栈

- **LangGraph 1.2.6**：状态图编排
- **LangChain 1.3.11**：Runnable 体系
- **PyTorch + InceptionV3**：DeepDream 算法实现
- **dashscope 官方 SDK**：千问大模型调用
- **Gradio**：Web 界面
- **Pillow**：图像处理（替代 matplotlib）

## 💡 使用示例

### 生成梦境

1. 上传一张图片
2. 输入："帮我生成一个美梦"
3. 等待生成，查看结果

### 查询天气

输入："查询北京的天气"

### 科学计算

输入："计算 sin(pi/2) + sqrt(16) + log(e)"

### 知识科普

输入："给我讲讲 DeepDream 算法原理"

## ⚙️ 性能说明

- **推荐配置**：RTX 3060 Mobile + R7 5800H 可流畅运行
- **GPU 加速**：自动检测 CUDA，有 GPU 时自动使用
- **CPU 模式**：无 GPU 时自动降级为 CPU 运行（速度较慢）
- **内存占用**：约 2-4GB（取决于图片大小和梦境类型）

## 📝 注意事项

1. 首次运行会自动下载 InceptionV3 预训练模型（约 100MB）
2. 疯狂梦模式计算量较大，建议使用 GPU
3. 图片尺寸会自动缩放到配置的最大尺寸，以保证性能
4. 生成的图片保存在 `dream_outputs/` 目录下

## 📄 License

MIT License
