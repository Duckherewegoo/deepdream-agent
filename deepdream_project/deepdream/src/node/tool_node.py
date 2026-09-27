"""工具执行节点"""
import json
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from ..state.state import AgentState
from ..tools import TOOL_MAP


def tool_node(state: AgentState) -> AgentState:
    """工具执行节点 - 执行 LLM 调用的工具

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 检查是否有工具调用
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    if not tool_calls:
        # 没有工具调用，直接返回
        return state

    tool_results = []
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_call_id = tool_call["id"]

        # 解析参数
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            arguments = {}

        # 查找工具
        tool = TOOL_MAP.get(tool_name)

        if tool:
            try:
                # 执行工具
                result = tool.invoke(arguments)
                tool_results.append({
                    "tool_name": tool_name,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                error_result = json.dumps({
                    "success": False,
                    "error": str(e)
                }, ensure_ascii=False)
                result = error_result
                tool_results.append({
                    "tool_name": tool_name,
                    "success": False,
                    "error": str(e)
                })
        else:
            result = json.dumps({
                "success": False,
                "error": f"未知工具：{tool_name}"
            }, ensure_ascii=False)
            tool_results.append({
                "tool_name": tool_name,
                "success": False,
                "error": f"未知工具：{tool_name}"
            })

        # 创建 ToolMessage
        tool_msg = ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        )
        tool_messages.append(tool_msg)

    # 将工具消息添加到历史
    state["messages"].extend(tool_messages)
    state["tool_results"].extend(tool_results)

    return state


def should_continue(state: AgentState) -> str:
    """条件判断节点 - 决定是否继续执行

    Args:
        state: 当前 Agent 状态

    Returns:
        下一个节点名称："tools" 或 "end"
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 如果最后一条消息有工具调用，继续执行工具
    if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get("tool_calls"):
        return "tools"

    # 否则结束
    return "end"


# 可运行对象包装
tool_runnable = RunnableLambda(tool_node)
should_continue_runnable = RunnableLambda(should_continue)
