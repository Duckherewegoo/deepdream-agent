"""输出节点 - 处理最终输出"""
import json
from typing import Optional
from langchain_core.runnables import RunnableLambda

from .llm_node import AgentState


def output_node(state: AgentState) -> AgentState:
    """输出节点 - 生成最终输出

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态，包含 final_output
    """
    messages = state["messages"]
    tool_results = state.get("tool_results", [])

    # 获取最后一条 AI 消息作为基础输出
    last_ai_message = None
    for msg in reversed(messages):
        if msg.type == "ai":
            last_ai_message = msg
            break

    if last_ai_message is None:
        state["final_output"] = "抱歉，没有生成任何回复。"
        return state

    # 构建最终输出
    final_output = last_ai_message.content

    # 如果有工具结果，附加工具结果信息
    if tool_results:
        dream_result = None
        for result in tool_results:
            if result["tool_name"] == "generate_dream" and result["success"]:
                try:
                    dream_data = json.loads(result["result"])
                    if dream_data.get("success"):
                        dream_result = dream_data
                        break
                except Exception:
                    pass

        # 如果有梦境生成结果，特殊处理
        if dream_result:
            dream_type_cn = dream_result.get("type_cn", "梦境")
            output_path = dream_result.get("path", "")
            final_output = f"""✨ **{dream_type_cn}生成成功！**

📁 图片已保存至：`{output_path}`

{last_ai_message.content if not last_ai_message.content.startswith('{') else ''}

💡 提示：你可以在 dream_outputs 目录下找到生成的图片文件。
如果需要其他风格的梦境，可以继续告诉我！"""

    state["final_output"] = final_output
    return state


def get_output_image_path(state: AgentState) -> Optional[str]:
    """从状态中获取生成的图片路径

    Args:
        state: Agent 状态

    Returns:
        图片路径，如果没有则返回 None
    """
    tool_results = state.get("tool_results", [])

    for result in tool_results:
        if result["tool_name"] == "generate_dream" and result["success"]:
            try:
                dream_data = json.loads(result["result"])
                if dream_data.get("success"):
                    return dream_data.get("path")
            except Exception:
                pass

    return None


# 可运行对象包装
output_runnable = RunnableLambda(output_node)
