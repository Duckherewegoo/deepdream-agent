# 🌙 DeepDream 梦境生成助手

> 基于 LangGraph + PyTorch 的智能梦境生成系统 — 让 AI 为你绘制超现实主义梦境

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [核心模块说明](#核心模块说明)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [性能优化](#性能优化)
- [扩展计划](#扩展计划)
- [商业化路线](#商业化路线)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

**DeepDream 梦境生成助手** 是一个基于 LangGraph 构建的智能 Agent 系统，它将 Google 经典的 DeepDream 算法与现代大语言模型相结合，创造出独特的交互式梦境生成体验。

用户只需上传一张图片并输入一个简单的指令（如"生成一个美梦"），系统便会：
1. 🧠 **理解意图** — 通过关键词匹配识别梦境类型
2. 🎨 **生成梦境** — 基于 InceptionV3 的 DeepDream 算法渲染图像
3. 💬 **描述梦境** — 调用通义千问 LLM 为生成的图像撰写诗意描述

### 技术亮点

- **Agentic Workflow**：基于 LangGraph 的状态机编排，支持复杂的条件路由
- **多尺度生成**：经典的 Octave 金字塔策略，从粗到细逐级优化
- **元数据驱动**：每个生成结果自动归档 `.meta.json`，支持一键复现
- **工具扩展**：内置科学计算器、天气查询等工具，LLM 可自主调用
- **实时可观测**：日志拦截 + Gradio Timer 轮询，用户可见每一步执行状态

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🌙 **5 种梦境类型** | 正常梦、美梦、噩梦、迷雾梦、疯狂梦，每种均有独立的超参数配置 |
| 🧠 **LLM 描述生成** | 基于图像元数据自动生成诗意化的梦境描述（40-60 字） |
| 🔁 **一键复现** | 读取 `.meta.json`，用相同参数在新图片上复现梦境 |
| 📊 **实时日志** | UI 上实时显示算法执行进度，透明化黑盒过程 |
| 🛠️ **工具系统** | 科学计算器、天气查询（JWT + EdDSA 认证），支持 LLM 自主调用 |
| 💾 **元数据归档** | 每次生成自动保存参数、耗时、设备信息到 `.meta.json` |
| 🚀 **单例优化** | DeepDream 生成器全局单例，模型只加载一次 |

---

## 快速开始

### 方式一：本地运行（推荐开发）

#### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourname/deepdream-agent.git
cd deepdream-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

2. 配置环境变量
bash

# 设置 DashScope API Key（通义千问）
export DASHSCOPE_API_KEY="your-api-key-here"

# 可选：调试模式
export DREAM_DEBUG_MODE="true"

3. 启动服务
bash

python main.py

访问 http://127.0.0.1:7860 即可开始体验。
方式二：Docker 部署（推荐生产）
1. 构建镜像

创建 Dockerfile：
dockerfile

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（PyTorch + 图像处理）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p uploads dream_outputs

# 暴露 Gradio 端口
EXPOSE 7860

# 启动命令
CMD ["python", "main.py"]

2. 一键部署脚本

创建 deploy.sh：
bash

#!/bin/bash
# deploy.sh

# 构建镜像
docker build -t deepdream-agent .

# 运行容器
docker run -d \
    --name deepdream \
    -p 7860:7860 \
    -e DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY}" \
    -v $(pwd)/uploads:/app/uploads \
    -v $(pwd)/dream_outputs:/app/dream_outputs \
    --gpus all \  # 如有 GPU
    --restart unless-stopped \
    deepdream-agent

echo "🌙 DeepDream Agent 已启动，访问 http://localhost:7860"

bash

# 执行部署
chmod +x deploy.sh
./deploy.sh

3. Docker Compose（推荐）

创建 docker-compose.yml：
yaml

version: '3.8'

services:
  deepdream:
    build: .
    ports:
      - "7860:7860"
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
      - DREAM_DEBUG_MODE=${DREAM_DEBUG_MODE:-false}
    volumes:
      - ./uploads:/app/uploads
      - ./dream_outputs:/app/dream_outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

bash

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down

项目架构
整体架构图
text

┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户交互层 (main.py)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Gradio UI    │  │ 日志拦截器   │  │ Timer 轮询   │  │ 状态展示     │  │
│  │ (Blocks)     │  │ DreamLogger  │  │ (0.5s)       │  │ (Meta Card)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ agent.chat(user_text, user_image)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          服务封装层 (agent.py)                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  DreamAgent                                                        │  │
│  │  ✅ __init__() → 创建目录 + 设置全局状态 + 初始化 LLM              │  │
│  │  ✅ save_uploaded_image() → UUID 命名 + 保存图片                   │  │
│  │  ✅ chat() → 调用 run_graph() + 提取结果                           │  │
│  │  ✅ chat_with_path() → 路径版本的接口                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ run_graph()
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        流程调度层 (core/graph.py)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  LangGraph 状态图                                                   │  │
│  │  ┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌──────────┐    │  │
│  │  │ 入口     │───▶│ 意图识别    │───▶│ 条件边   │───▶│ LLM 节点 │    │  │
│  │  │ (intent) │    │ (intent_node)│    │ (路由)   │    │(llm_node)│    │  │
│  │  └─────────┘    └─────────────┘    └────┬────┘    └────┬─────┘    │  │
│  │                                          │              │           │  │
│  │                                          ▼              │           │  │
│  │                                    ┌─────────────┐     │           │  │
│  │                                    │ 梦境生成    │     │           │  │
│  │                                    │(dream_gen_  │     │           │  │
│  │                                    │   node)     │     │           │  │
│  │                                    └─────────────┘     │           │  │
│  │                                          │              │           │  │
│  │                                          ▼              ▼           │  │
│  │                                    ┌─────────────────────────┐     │  │
│  │                                    │    输出节点            │     │  │
│  │                                    │   (output_node)        │     │  │
│  │                                    └─────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐     ┌─────────────────────┐     ┌───────────────────────┐
│  算法层       │     │  认知层             │     │  工具层               │
│  generator.py │     │  llm_node.py        │     │  tool_node.py         │
│  ✅ InceptionV3│     │  ✅ 通义千问       │     │  ✅ 工具注册表       │
│  ✅ 多尺度生成│     │  ✅ 降级保底       │     │  ✅ 三层异常防护     │
│  ✅ 梯度归一化│     │  ✅ 语气生成       │     │  ✅ 审计日志         │
│  ✅ 元数据归档│     │                     │     │                       │
└───────────────┘     └─────────────────────┘     └──────────┬────────────┘
                                                             │
                                    ┌────────────────────────┼────────────────────────┐
                                    │                        │                        │
                                    ▼                        ▼                        ▼
                            ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
                            │ 科学计算器    │    │ 梦境生成工具  │    │ 天气查询工具  │
                            │ calculator.py │    │ dream_gen.py  │    │ weather.py    │
                            │ ✅ 沙箱 eval  │    │ ✅ 单例模式   │    │ ✅ JWT 认证   │
                            │ ✅ 白名单     │    │ ✅ 双重重试   │    │ ✅ 链式 API   │
                            └───────────────┘    └───────────────┘    └───────────────┘

目录结构
text

deepdream-agent/
├── deepdream/                         # 核心包
│   ├── src/
│   │   ├── agent/
│   │   │   └── agent.py               # Agent 封装层
│   │   ├── config/
│   │   │   └── settings.py            # 全局配置（关键词、超参数、层映射）
│   │   ├── core/
│   │   │   └── graph.py               # LangGraph 状态图定义
│   │   ├── deepdream/
│   │   │   └── generator.py           # DeepDream 核心算法（PyTorch）
│   │   ├── node/
│   │   │   ├── intent_node.py         # 意图识别节点
│   │   │   ├── dream_gen_node.py      # 梦境生成节点
│   │   │   ├── llm_node.py            # LLM 调用节点（通义千问）
│   │   │   ├── tool_node.py           # 工具执行节点
│   │   │   └── output_node.py         # 输出节点
│   │   ├── state/
│   │   │   └── state.py               # AgentState 类型定义
│   │   ├── tools/
│   │   │   ├── calculator.py          # 科学计算器工具
│   │   │   ├── dream_gen.py           # 梦境生成工具（LLM 驱动）
│   │   │   └── weather.py             # 天气查询工具
│   │   └── utils/
│   │       └── meta_utils.py          # 元数据加载工具
│   └── __init__.py
├── main.py                             # 入口文件（Gradio UI）
├── requirements.txt                    # Python 依赖
├── Dockerfile                          # Docker 镜像
├── docker-compose.yml                  # Docker Compose 编排
├── deploy.sh                           # 一键部署脚本
├── README.md                           # 项目文档
└── LICENSE                             # MIT 许可证

核心模块说明
1. Agent 封装层 (agent.py)
方法	功能	返回值
__init__(upload_dir, output_dir)	初始化 Agent，创建目录，设置全局状态	None
chat(user_text, user_image)	接收文本 + PIL Image，执行完整流程	(reply, img_path, state)
chat_with_path(user_text, image_path)	接收文本 + 图片路径，执行完整流程	(reply, img_path, state)
save_uploaded_image(image)	保存上传图片，UUID 命名	filepath
2. LangGraph 状态图 (graph.py)
python

# 状态流转
intent_node → should_generate_dream (条件边)
    ├── dream_gen_node → llm_node → output_node → END
    └── llm_node → output_node → END

节点	触发条件	功能
intent_node	始终触发	关键词匹配，设置 dream_type 和 need_dream_gen
dream_gen_node	need_dream_gen == True	调用 DeepDream 生成器，写入 output_image_path
llm_node	始终触发	调用通义千问，生成梦境描述
output_node	始终触发	提取最终输出，附加错误信息
3. DeepDream 引擎 (generator.py)
方法	功能	关键参数
generate(input_path, dream_type, output_dir)	主入口，执行完整生成流程	返回输出路径
_generate_base()	多尺度 Octave 生成	OCTAVE_LAYERS, OCTAVE_SCALE
_dream_step()	单次梯度上升迭代	STEP_SIZE, GRADIENT_SCALE
_add_gradient_noise()	梯度噪声注入	GRADIENT_NOISE

核心算法流程：
text

输入图像
    ↓
加载并缩放到 MAX_SIZE
    ↓
计算 Octave 金字塔尺寸（从小到大）
    ↓
for each Octave:
    上采样到当前尺寸
    for each Iteration:
        前向传播 → 计算损失 → 反向传播 → 梯度归一化 → 添加噪声 → 更新图像
    ↓
恢复到原始尺寸
    ↓
后处理（高斯模糊/锐化/亮度调整）
    ↓
保存图像 + 生成 .meta.json

4. LLM 节点 (llm_node.py)
功能	实现
系统提示词	定义梦境解说员角色，40-60 字描述规则
语气生成	从 .meta.json 读取参数，翻译为语义描述
降级保底	检测到生成失败时，强制 LLM 如实告知
历史转换	将 LangChain 消息转为 dashscope SDK 格式
5. 工具系统 (tools/)
工具	文件	核心能力
科学计算器	calculator.py	沙箱 eval，40+ 数学函数
梦境生成	dream_gen.py	LLM 驱动的 DeepDream 调用
天气查询	weather.py	JWT + EdDSA 认证，链式 API 调用
配置说明
核心参数 (settings.py)
梦境关键词映射
python

DREAM_KEYWORDS = {
    "normal": ["正常梦", "普通梦", "常规", "normal"],
    "sweet": ["甜蜜梦", "美梦", "甜梦", "温柔", "治愈", "sweet"],
    "nightmare": ["噩梦", "恐怖", "惊悚", "吓人", "nightmare"],
    "mist": ["迷雾梦", "朦胧", "迷雾", "模糊", "mist"],
    "crazy": ["疯狂梦", "疯狂", "迷幻", "极致", "crazy"]
}

梦境超参数配置
梦境类型	STEP_SIZE	ITER_PER_OCTAVE	REALITY_FUSION	GRADIENT_SCALE	特征层
normal	0.01	80	0.18	1.0	Mixed_5b, 5c
sweet	0.006	50	0.12	0.6	Conv2d_3b, 4a
nightmare	0.018	120	0.02	1.8	Mixed_6c, 7a, 7b
mist	0.007	60	0.18	0.7	Mixed_5b, 6a, 6c, 7a
crazy	0.022	220	0.0	2.2	全层（7 层）
环境变量
变量	必填	默认值	说明
DASHSCOPE_API_KEY	✅	""	通义千问 API Key
DREAM_DEBUG_MODE	❌	False	是否打印梯度/噪声调试日志
DREAM_MAX_SIZE	❌	800	输入图像最大尺寸
API 参考
DreamAgent
"""
python

from deepdream.src.agent import DreamAgent

agent = DreamAgent(
    upload_dir="./uploads",
    output_dir="./dream_outputs"
)

# 方式一：传入 PIL Image
reply, img_path, state = agent.chat(
    user_text="生成一个美梦",
    user_image=pil_image
)

# 方式二：传入图片路径
reply, img_path, state = agent.chat_with_path(
    user_text="生成一个噩梦",
    image_path="/path/to/image.png"
)

run_graph
python

from deepdream.src.core.graph import run_graph

final_state = run_graph(
    user_input="生成一个疯狂梦",
    image_path="/path/to/image.png",
    output_dir="./dream_outputs",
    max_iterations=5
)

# 结果提取
reply = final_state["final_output"]
img_path = final_state["output_image_path"]
dream_type = final_state["dream_type"]

贡献指南
开发流程

    Fork 项目

    创建功能分支 (git checkout -b feature/amazing)

    提交变更 (git commit -m 'Add amazing feature')

    推送到分支 (git push origin feature/amazing)

    创建 Pull Request

代码规范

    Python: PEP 8，使用 black 格式化

    类型注解: 所有函数必须包含类型注解

    文档字符串: 使用 Google 风格 docstring

    日志: 使用 [MODULE] 标签分类

测试
bash

# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_generator.py -v

# 覆盖率报告
pytest --cov=deepdream tests/
"""

许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。
致谢

    Google DeepDream - 原始算法灵感

    LangChain - Agent 框架

    PyTorch - 深度学习引擎

    DashScope - 通义千问 API

    Gradio - UI 框架

联系方式

    作者: Your Name

    Email: your.email@example.com

    GitHub: github.com/yourname/deepdream-agent

🌙 让每个梦都值得被看见
