---
id: 0002-langgraph-router
aliases: []
tags: []
---

# ADR-0002: 采用 LangGraph 编排 Agent 工具路由

- Status: Accepted
- Date: 约 2026-10（重构期，解决早期脚本的失控）
- Related:
  - 代码：`core/graph.py`、`node/llm_node.py`、`node/tool_node.py`
  - 关键 commit：`02230b9`（移除 RunnableLambda、接入 MemorySaver、防死循环）
  - 历史背景：前期曾因手写逻辑过于混乱，导致产生多个 `broken_deepdream_project` 废弃目录

## Context

**前史教训**：在 9 月决定重构之前，项目是一个单文件脚本（由 4 月的代码一路缝补而来）。
当试图把“生成图片、调天气、算数学”等工具塞进大模型对话时，原本手写的 `if-else` 或简单的字符串匹配逻辑彻底失控，代码耦合严重，产生了多个无法维护的 `broken_*` 目录。

**业务需求**：用户的输入是多模态且随机的（正常对话、任务型生成、知识科普）。
程序必须根据大模型的输出（Function Calling），动态决定下一步是去调天气 API、调计算器，还是去调 GPU 跑图。

为了避免重蹈“逻辑交织、无法维护”的覆辙，重构时必须引入显式的状态机。

## Decision

采用 **LangGraph** 作为状态图编排引擎，构建标准的 ReAct（Reasoning + Acting）循环。

- `llm_node` 接收用户输入，判断是否需要调用工具，产出工具调用的 JSON。
- 条件边（Conditional Edges）根据解析结果，路由到具体的 `tool_node` 或 `output_node`。
- 状态传递使用普通的 dict，节点函数**直接返回增量 dict**，而不是被包装成 `RunnableLambda`。

## Alternatives Considered

- **A. 手写 while 循环 + JSON 解析**：可行，但需要自己维护上下文历史、自己处理大模型输出 JSON 格式崩坏的重试、自己写 string 匹配的工具路由。维护成本高。
- **B. 纯 LangChain AgentExecutor**：**未采纳**。AgentExecutor 内部封装太厚，黑盒化严重，难以在工具调用前后插入自定义的日志、异常捕获和参数校验（比如判断图片尺寸是否超限）。
- **C. LangGraph**：采纳。用显式的图（Graph）来定义路由，节点即普通的 Python 函数，状态流转完全透明可控。

## Consequences

**正面**

- **路由逻辑极度清晰**：看一眼 `graph.py` 的边和条件函数，就知道从用户输入到生成图片的完整链路。
- **状态可控**：通过自定义 State，可以把上传的图片、用户的 prompt、中间工具返回的结果一并携带，节点随用随取。
- **支持复杂的异常兜底**：在 `02230b9` 中接入了 `MemorySaver` checkpointer，当流程意外崩溃时，可以捞取崩溃前的快照，便于调试（在单 Agent 场景下非常实用）。

**负面（接受的代价）**

- **心智负担**：必须时刻记住“节点函数返回的是状态增量（dict）”，不能用老式 LangChain 的 `Runnable` 思想去写。
- **死循环风险**：ReAct 模式极易因为大模型反复调用同一个工具陷入死循环。后来在 commit 里加了 `loop_count` 上限（见 `02230b9`）才解决。
- **过度设计的边缘**：对于只有 3 个工具的场景，引入图编排在初期看起来有点“杀鸡用牛刀”，但只要工具数量或交互轮次一多，收益就会陡增。

**遗留**

- `RunnableLambda` 包装曾经被引入，后来因为状态传递繁琐被废弃（2026-10-24 清理），恢复为纯函数 + dict。这提醒我们：框架的高级封装未必总是更好。

## 备注
在 `task-planner` 项目中，LangGraph 用于“复杂长链路的 DAG 图依赖与中断恢复”；而在本项目，它仅用于“单 Agent 的单轮/多轮工具路由”。同一框架，不同场景，不要混用架构思路。
