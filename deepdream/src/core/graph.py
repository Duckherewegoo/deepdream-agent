"""LangGraph 图定义 - 核心图结构（重构优化版）"""
import logging
import threading
from typing import Any

from langgraph.graph import END, StateGraph

from ..node import (
    dream_gen_node,
    intent_node,
    llm_node,
    tool_node,
    output_node,
    should_generate_dream,
    should_continue,
)

# 从统一的 state 模块引入
from ..state import AgentState

logger = logging.getLogger(__name__)

# 预构建图 + 线程安全锁
_prebuilt_graph = None
_graph_lock = threading.Lock()


def build_graph():
    """构建 LangGraph 状态图（预构建，线程安全）"""
    global _prebuilt_graph

    # 快路径检查
    if _prebuilt_graph is not None:
        return _prebuilt_graph

    with _graph_lock:
        # 双重检查锁定
        if _prebuilt_graph is not None:
            return _prebuilt_graph

        logger.info("[GRAPH] 开始构建 LangGraph 状态图...")

        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("intent", intent_node)
        workflow.add_node("dream_gen", dream_gen_node)
        workflow.add_node("llm", llm_node)
        workflow.add_node("tool_execution", tool_node) 
        workflow.add_node("output", output_node)

        # 设置入口
        workflow.set_entry_point("intent")

        workflow.add_edge("dream_gen", "llm")
    
        # 将 llm 到 output 的死路，改成条件路由
        workflow.add_conditional_edges(
         "llm",
         should_continue,
            {
                "tools": "tool_execution",
                "end": "output"
                }
            )
        # 工具执行完，回到 llm 继续生成最终回复
        workflow.add_edge("tool_execution", "llm")
        workflow.add_edge("output", END)

        # 编译图
        _prebuilt_graph = workflow.compile()
        logger.info("[GRAPH] LangGraph 状态图构建完成（缓存已生效）")

        return _prebuilt_graph


def create_initial_state(
    user_input: str,
    image_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """创建初始状态（过滤掉 None，确保 State 干净）"""
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
        "output_dir": output_dir,
    }
    return state


def run_graph(
    user_input: str,
    image_path: str | None = None,
    output_dir: str | None = None,
    max_iterations: int = 5,
) -> dict[str, Any]:
    """运行 LangGraph"""
    logger.info(f"[GRAPH] 开始运行 | 输入长度: {len(user_input)} | 图片: {image_path}")

    app = build_graph()
    initial_state = create_initial_state(user_input, image_path, output_dir)

    # 修复：将 max_iterations 真正传给 LangGraph 的 recursion_limit
    config = {"recursion_limit": max_iterations}

    try:
        # invoke 的第二参数就是 config
        final_state = app.invoke(initial_state, config=config)

        logger.info(
            f"[GRAPH] 执行完成 | 状态: {final_state.get('status')} | "
            f"生成图片: {final_state.get('output_image_path')} | "
            f"错误: {final_state.get('error')}"
        )
        return final_state

    except Exception as e:
        logger.exception(f"[GRAPH] 运行发生致命错误: {e}")
        # 如果图崩了，至少把错误写回 state，防止上层拿不到错误信息
        initial_state["error"] = str(e)
        initial_state["status"] = "执行崩溃"
        initial_state["final_output"] = f"系统内部错误: {e!s}"
        return initial_state
