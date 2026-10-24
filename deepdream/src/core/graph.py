"""LangGraph 图定义 - 核心图结构（重构优化版）"""
import logging
import threading
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..node import (
    dream_gen_node,
    intent_node,
    llm_node,
    output_node,
    should_continue,
    should_generate_dream,
    tool_node,
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

        memory = MemorySaver()
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("intent", intent_node)
        workflow.add_node("dream_gen", dream_gen_node)
        workflow.add_node("llm", llm_node)
        workflow.add_node("tool_execution", tool_node)
        workflow.add_node("output", output_node)

        # 设置入口
        workflow.set_entry_point("intent")

        workflow.add_conditional_edges(
            "intent",
            should_generate_dream, # 这个函数返回 "dream_gen" 或 "llm"
            {
                "dream_gen": "dream_gen",
                "llm": "llm"
            }
        )

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
        _prebuilt_graph = workflow.compile(checkpointer=memory)
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
        "loop_count": 0,
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
    """运行 LangGraph（内部自动管理 Checkpointer 会话，并在执行完毕后销毁现场）"""
    logger.info(f"[GRAPH] 开始运行 | 输入长度: {len(user_input)} | 图片: {image_path}")

    app = build_graph()
    initial_state = create_initial_state(user_input, image_path, output_dir)

    session_thread_id = str(uuid.uuid4())
    config = {
        "recursion_limit": max_iterations,
        "configurable": {"thread_id": session_thread_id}
    }

    result_state = dict(initial_state) # 兜底返回

    try:
        final_state = app.invoke(initial_state, config=config)
        logger.info(f"[GRAPH] 执行完成 | 状态: {final_state.get('status')}")
        result_state = final_state

    except Exception as e:
        logger.exception(f"[GRAPH] 运行发生致命错误: {e}")

        # 1. 尝试捞取崩溃前的遗言
        try:
            snapshot = app.get_state(config)
            if snapshot and snapshot.values:
                result_state = dict(snapshot.values)
        except Exception as inner_e:
            logger.error(f"[GRAPH] 捞取崩溃现场失败: {inner_e}")

        # 2. 覆盖错误信息
        result_state["error"] = str(e)
        result_state["status"] = "执行崩溃"
        result_state["final_output"] = f"系统内部错误: {e!s}"

    finally:
        # 3. 核心：无论成功还是失败，强制清理 Checkpointer 内存（防泄漏）
        _cleanup_checkpointer_memory(app, session_thread_id)

    return result_state


def _cleanup_checkpointer_memory(app: Any, thread_id: str) -> None:
    """手动清理 MemorySaver 中指定 thread_id 的快照，防止内存泄漏。"""
    checkpointer = getattr(app, "checkpointer", None)
    if not checkpointer:
        return

    try:
        # MemorySaver 底层用 .storage 存 checkpoint，用 .writes 存中间写入
        if hasattr(checkpointer, "storage") and thread_id in checkpointer.storage:
            del checkpointer.storage[thread_id]

        if hasattr(checkpointer, "writes") and thread_id in checkpointer.writes:
            del checkpointer.writes[thread_id]

        logger.debug(f"[GRAPH] 已清理 Checkpointer 内存 | thread_id: {thread_id}")
    except Exception as e:
        logger.warning(f"[GRAPH] 清理 Checkpointer 内存失败: {e}")
