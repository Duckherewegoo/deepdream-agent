"""LangGraph 图定义 - 核心图结构"""
import os
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from ..node import (
    AgentState,
    intent_node,
    dream_gen_node,
    llm_node,
    output_node,
    should_generate_dream,
)


def build_graph() -> StateGraph:
    """构建 LangGraph 状态图

    图结构：
    入口 → intent_node → 条件判断
                      ↓ 需要生成梦境
                 dream_gen_node → llm_node → output_node → END
                      ↓ 不需要
                     llm_node → output_node → END

    Returns:
        编译后的 LangGraph 图
    """
    print(f"[GRAPH] 开始构建 LangGraph 状态图...")

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("intent", intent_node)
    workflow.add_node("dream_gen", dream_gen_node)
    workflow.add_node("llm", llm_node)
    workflow.add_node("output", output_node)

    # 设置入口点
    workflow.set_entry_point("intent")

    # 添加条件边：意图识别后判断是否需要生成梦境
    workflow.add_conditional_edges(
        "intent",
        should_generate_dream,
        {
            "dream_gen": "dream_gen",
            "llm": "llm",
        }
    )

    # 梦境生成后到 LLM
    workflow.add_edge("dream_gen", "llm")

    # LLM 后到输出节点
    workflow.add_edge("llm", "output")

    # 输出节点后结束
    workflow.add_edge("output", END)

    # 编译图
    app = workflow.compile()

    print(f"[GRAPH] LangGraph 状态图构建完成")
    return app


def create_initial_state(
    user_input: str,
    image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """创建初始状态

    Args:
        user_input: 用户输入文本
        image_path: 用户上传的图片路径
        output_dir: 梦境输出目录

    Returns:
        初始状态字典
    """
    state = {
        "messages": [],
        "user_input": user_input,
        "image_path": image_path,
        "output_image_path": None,
        "dream_type": None,
        "need_dream_gen": False,
        "tool_results": [],
        "final_output": None,
        "status": "初始化中...",
        "error": None,
    }

    if output_dir:
        state["output_dir"] = output_dir

    return state


def run_graph(
    user_input: str,
    image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """运行 LangGraph

    Args:
        user_input: 用户输入文本
        image_path: 用户上传的图片路径
        output_dir: 梦境输出目录
        max_iterations: 最大迭代次数，防止无限循环

    Returns:
        最终状态
    """
    print(f"\n{'='*60}")
    print(f"[GRAPH] 开始运行 LangGraph")
    print(f"[GRAPH] 用户输入: {user_input[:50]}...")
    print(f"[GRAPH] 图片路径: {image_path}")
    print(f"[GRAPH] 输出目录: {output_dir}")
    print(f"{'='*60}")

    # 构建图
    app = build_graph()

    # 创建初始状态
    initial_state = create_initial_state(user_input, image_path, output_dir)

    # 运行图
    print(f"[GRAPH] 开始执行图...")
    final_state = app.invoke(initial_state)
    print(f"[GRAPH] 图执行完成")

    # 打印最终状态摘要
    print(f"\n{'='*60}")
    print(f"[GRAPH] 最终状态摘要:")
    print(f"[GRAPH]   状态: {final_state.get('status', '未知')}")
    print(f"[GRAPH]   梦境类型: {final_state.get('dream_type', '无')}")
    print(f"[GRAPH]   生成图片: {final_state.get('output_image_path', '无')}")
    print(f"[GRAPH]   错误: {final_state.get('error', '无')}")
    print(f"[GRAPH]   输出长度: {len(final_state.get('final_output', ''))}")
    print(f"{'='*60}\n")

    return final_state
