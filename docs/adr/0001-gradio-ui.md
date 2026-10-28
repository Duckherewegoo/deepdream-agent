---
id: 0001-gradio-ui
aliases: []
tags: []
---

# ADR-0001: 采用 Gradio 作为 UI 框架

- Status: Accepted
- Date: 约 2026-04 至 2026-09（演进期，早于 `task-planner`）
- Related:
  - `pyproject.toml`：`gradio>=4.0.0`
  - 独立性声明：本项目与 `task-planner` 完全独立，两者的 UI 选型（Gradio vs Dash）无因果或演进关系

## Context

**项目阶段**：项目始于 2026 年 4 月的 Jupyter Notebook（`DeepDream.ipynb`）算法实验。
随后经历了脚本化（`deepdream_agent.py`，4 月 15 日）、修补（`deepdream_fixed.py`，6 月 25 日），
并在 9 月陷入了严重的代码腐化与结构失控（本地曾积压多个 `broken_*` 目录），最终在 9 月底决定推倒重构。

**需求特征**：
1. **轻交互**：核心链路是“上传图片 -> 选梦境类型 -> 等待生成 -> 查看图片”，加上一个纯文本的对话框。
2. **算法为王**：真正的重头戏在后端（InceptionV3 的梯度上升迭代，单次 120 秒）。UI 只需要做好输入收集和结果展示。
3. **单人全栈开发**：此时的技术栈积累主要在 Python 数据科学侧。

**当时的技能认知**：在 2026 年上半年，开发者主要使用 Gradio 和 FastAPI 构建应用，尚未接触 Dash。

## Decision

采用 **Gradio** 作为 UI 框架。

通过 Gradio 直接通过 Python 函数构建 Web 界面（`gr.Interface` / `gr.Blocks`），不写独立的前端代码，不引入复杂的前端工程化。

## Alternatives Considered

评估状态如实标注——**基于立项时的认知与需求**：

- **A. FastAPI + 原生前端（HTML/JS）**：当时有 FastAPI 基础，但为了一个“上传图片+聊天框”的页面手写原生前端，对于单人项目来说性价比极低。
- **B. Streamlit**：**未评估**。当时大概率没进入视野。事后看，它与 Gradio 定位高度重合，但在大模型生态的组件支持上，当时的 Gradio 更顺手。
- **C. Dash**：**未评估**（当时未学习）。事后看，Dash 更适合“高度自定义交互流程图”，而本项目的 UI 诉求极其线性，用 Dash 属于杀鸡用牛刀，会增加大量的回调代码负担。
- **D. Gradio**：采纳。

## Consequences

**正面**

- **零前端代码**：对于上传、展示图片、多轮对话，Gradio 自带封装组件，开发极快。
- **生态契合**：与 `dashscope` SDK 和大模型对话界面的原生契合度高。
- **让开发者专注算法**：不用操心 CSS 和 DOM，能把精力放在 PyTorch 的 `generator.py` 和 LangGraph 的节点编排上。

**负面（接受的代价）**

- **UI 自定义天花板低**：虽然有 `gr.Blocks`，但若要实现精细的页面级状态联动或复杂布局，依然受限。
- **长耗时任务与 UI 线程的阻塞风险**：DeepDream 的 GPU 迭代长达 120 秒以上。虽然本项目没有引入异步，但必须处理“生成时 UI 假死”的问题（通常依赖 Gradio 内部的进度条 `gr.Progress` 或线程托管机制）。
- **端口固化印象**：Gradio 默认使用 7860 端口，这成了后续所有项目的“默认记忆”。

**遗留**

- 因为 UI 足够轻量，后续重构（v2.1.0）虽然把后端从普通脚本升级成了 LangGraph 状态图，但 UI 层几乎可以原封不动地保留。这是“组件化 UI 框架”在稳定性上的红利。

## 备注：框架选型的“场景决定论”

在后来的 `task-planner` 项目中，因为对 DAG 流程图的强交互需求，项目转而使用了 Dash + Cytoscape。

这两个项目对 UI 的选型截然不同，但并不矛盾：**Gradio 适合线性的、以输入输出为主的“表单式/对话式”应用；Dash 适合状态密集的“看板式/图操作式”应用。** 框架没有绝对好坏，只有场景是否匹配。
