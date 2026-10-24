"""工具执行节点"""
import json
import logging
import time
from typing import Any

from langchain_core.messages import ToolMessage

from ..state import AgentState
from ..tools import TOOL_MAP

logger = logging.getLogger(__name__)


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """统一提取工具调用的防御性辅助函数"""
    if not message:
        return []

    # 1. 优先找顶层标准属性（新版 LangChain 惯例）
    calls = getattr(message, "tool_calls", None)

    # 2. 如果顶层没有，再去找 additional_kwargs（旧版或特定第三方模型）
    if not calls:
        kwargs = getattr(message, "additional_kwargs", None) or {}
        calls = kwargs.get("tool_calls", [])

    # 3. 必须是列表
    if not isinstance(calls, list):
        return []

    # 4. 内芯校验：过滤掉不是字典的脏数据
    return [call for call in calls if isinstance(call, dict)]


def tool_node(state: AgentState) -> dict[str, Any]:
    """工具执行节点 - 执行 LLM 调用的工具"""
    logger.info("[NODE] 工具执行节点开始执行")

    # 防御：使用 get，避免 KeyError
    messages = state.get("messages") or []
    if not messages:
        logger.warning("[TOOL] messages 为空，跳过工具执行")
        return {}

    last_message = messages[-1]
    tool_calls = _extract_tool_calls(last_message)

    if not tool_calls:
        logger.info("[TOOL] 最后一条消息没有工具调用，节点直出")
        return {}

    logger.info(f"[TOOL] 检测到 {len(tool_calls)} 个工具调用请求")

    # 防御：确保 tool_results 存在（这里用 state.get 防止直接 mutate 原对象）
    tool_results = list(state.get("tool_results") or [])
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
        except (json.JSONDecodeError, ValueError, TypeError) as e:
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

    logger.info(
        f"[NODE] 工具执行节点完成 | 成功写入 {len(tool_messages)} 条 ToolMessage"
    )

    # 核心修复：返回增量字典，让 LangGraph 自动合并
    return {
        "messages": tool_messages,
        "tool_results": tool_results
    }

def should_continue(state: AgentState) -> str:
    """条件判断节点 - 决定是否继续执行工具循环"""
    if state.get("loop_count", 0) >= 5:
        logger.warning("[ROUTE] 触发循环上限，强制结束")
        return "end"

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

