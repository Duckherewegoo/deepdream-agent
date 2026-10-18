"""工具执行节点"""
import json
import logging
import time
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from ..state import AgentState
from ..tools import TOOL_MAP

logger = logging.getLogger(__name__)


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """统一提取工具调用的防御性辅助函数"""
    if not message:
        return []
    kwargs = getattr(message, "additional_kwargs", None) or {}
    calls = kwargs.get("tool_calls", [])
    return calls if isinstance(calls, list) else []


def tool_node(state: AgentState) -> AgentState:
    """工具执行节点 - 执行 LLM 调用的工具"""
    logger.info("[NODE] 工具执行节点开始执行")

    # 防御：使用 get，避免 KeyError
    messages = state.get("messages") or []
    if not messages:
        logger.warning("[TOOL] messages 为空，跳过工具执行")
        return state

    last_message = messages[-1]
    tool_calls = _extract_tool_calls(last_message)

    if not tool_calls:
        logger.info("[TOOL] 最后一条消息没有工具调用，节点直出")
        return state

    logger.info(f"[TOOL] 检测到 {len(tool_calls)} 个工具调用请求")

    # 防御：确保 tool_results 存在
    tool_results = state.setdefault("tool_results", [])
    tool_messages: list[ToolMessage] = []

    for idx, tool_call in enumerate(tool_calls, 1):
        # 防御：工具调用结构可能畸形
        try:
            function_info = tool_call.get("function", {})
            tool_name = function_info.get("name", "unknown")
            tool_call_id = tool_call.get("id", f"call_{idx}")
            raw_args = function_info.get("arguments", "{}")
        except AttributeError:
            logger.error(f"[TOOL] 畸形的 tool_call 结构: {tool_call}")
            continue

        logger.info(f"[TOOL] ({idx}/{len(tool_calls)}) 准备执行: {tool_name}")

        # 防御：参数解析失败时，不再静默传空字典
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            if not isinstance(arguments, dict):
                raise TypeError(f"参数解析结果不是字典: {type(arguments)}")
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"工具参数解析失败: {e} | 原始参数: {raw_args!r}"
            logger.error(f"[TOOL] ❌ {error_msg}")
            result = json.dumps(
                {"success": False, "error": error_msg},
                ensure_ascii=False,
            )
            tool_results.append({
                "tool_name": tool_name,
                "success": False,
                "error": error_msg,
            })
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call_id,
                name=tool_name,
            ))
            continue

        # 查找并执行工具
        tool = TOOL_MAP.get(tool_name)
        if not tool:
            error_msg = f"未知工具：{tool_name}"
            logger.error(f"[TOOL] ❌ {error_msg}")
            result = json.dumps(
                {"success": False, "error": error_msg},
                ensure_ascii=False,
            )
            tool_results.append({
                "tool_name": tool_name,
                "success": False,
                "error": error_msg,
            })
        else:
            start_time = time.time()
            try:
                result = tool.invoke(arguments)
                elapsed = time.time() - start_time
                logger.info(f"[TOOL] {tool_name} 执行成功 | 耗时: {elapsed:.2f}s")
                tool_results.append({
                    "tool_name": tool_name,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                elapsed = time.time() - start_time
                # 关键：完整堆栈进后台日志，结果里只给用户看友好信息
                logger.exception(
                    f"[TOOL] ❌ {tool_name} 执行异常 | 耗时: {elapsed:.2f}s"
                )
                error_result = json.dumps(
                    {"success": False, "error": str(e)},
                    ensure_ascii=False,
                )
                result = error_result
                tool_results.append({
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(e),
                })

        # 统一创建 ToolMessage
        tool_messages.append(ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name,
        ))

    # 回写历史
    state["messages"].extend(tool_messages)
    logger.info(
        f"[NODE] 工具执行节点完成 | 成功写入 {len(tool_messages)} 条 ToolMessage"
    )
    return state


def should_continue(state: AgentState) -> str:
    """条件判断节点 - 决定是否继续执行工具循环"""
    messages = state.get("messages") or []
    if not messages:
        logger.info("[ROUTE] messages 为空 -> end")
        return "end"

    last_message = messages[-1]
    tool_calls = _extract_tool_calls(last_message)

    if tool_calls:
        logger.info(f"[ROUTE] 检测到待执行工具 {len(tool_calls)} 个 -> tools")
        return "tools"

    logger.info("[ROUTE] 无工具调用 -> end")
    return "end"


# 可运行对象包装
tool_runnable = RunnableLambda(tool_node)
should_continue_runnable = RunnableLambda(should_continue)
