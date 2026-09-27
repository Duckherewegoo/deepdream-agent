"""LangGraph 图定义 - 核心图结构"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from ..node import (
    AgentState,
    llm_node,
    tool_node,
    output_node,
    should_continue,
)


def build_graph() -> StateGraph:
    """构建 LangGraph 状态图

    Returns:
        编译后的 LangGraph 图
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("llm", llm_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("output", output_node)

    # 设置入口点
    workflow.set_entry_point("llm")

    # 添加条件边：LLM 节点后判断是否需要调用工具
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": "output",
        }
    )

    # 工具执行完后回到 LLM
    workflow.add_edge("tools", "llm")

    # 输出节点后结束
    workflow.add_edge("output", END)

    # 编译图
    app = workflow.compile()

    return app


def create_initial_state(
    user_input: str,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """创建初始状态

    Args:
        user_input: 用户输入文本
        image_path: 用户上传的图片路径

    Returns:
        初始状态字典
    """
    return {
        "messages": [],
        "user_input": user_input,
        "image_path": image_path,
        "tool_results": [],
        "final_output": None,
    }


def run_graph(
    user_input: str,
    image_path: Optional[str] = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """运行 LangGraph

    Args:
        user_input: 用户输入文本
        image_path: 用户上传的图片路径
        max_iterations: 最大迭代次数，防止无限循环

    Returns:
        最终状态
    """
    # 构建图
    app = build_graph()

    # 创建初始状态
    initial_state = create_initial_state(user_input, image_path)

    # 运行图
    final_state = app.invoke(initial_state)

    return final_state
