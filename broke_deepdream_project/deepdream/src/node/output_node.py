"""输出节点 - 处理最终输出"""
import json
import os
import traceback
from typing import Optional
from langchain_core.runnables import RunnableLambda

from .llm_node import AgentState
from ..config.settings import DREAM_CN_NAMES


def output_node(state: AgentState) -> AgentState:
    """输出节点 - 生成最终输出

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态，包含 final_output
    """
    print(f"\n[NODE] 输出节点开始执行")

    messages = state["messages"]
    output_image_path = state.get("output_image_path")
    dream_type = state.get("dream_type")
    error = state.get("error")

    print(f"[OUTPUT] 生成图片路径: {output_image_path}")
    print(f"[OUTPUT] 梦境类型: {dream_type}")
    print(f"[OUTPUT] 错误信息: {error}")

    # 获取最后一条 AI 消息作为基础输出
    last_ai_message = None
    for msg in reversed(messages):
        if msg.type == "ai":
            last_ai_message = msg
            break

    if last_ai_message is None:
        print(f"[OUTPUT] 没有找到 AI 消息，使用默认回复")
        state["final_output"] = "抱歉，没有生成任何回复。"
        return state

    # 构建最终输出
    final_output = last_ai_message.content

    # 如果有错误，添加错误信息
    if error:
        final_output = f"{final_output}\n\n⚠️ **注意**：{error}"

    state["final_output"] = final_output
    print(f"[OUTPUT] 最终输出长度: {len(final_output)}")
    print(f"[OUTPUT] 最终输出预览: {final_output[:100]}...")

    print(f"[NODE] 输出节点执行完成")
    return state


def get_output_image_path(state: AgentState) -> Optional[str]:
    """从状态中获取生成的图片路径

    Args:
        state: Agent 状态

    Returns:
        图片路径，如果没有则返回 None
    """
    output_image_path = state.get("output_image_path")

    if output_image_path and os.path.exists(output_image_path):
        return output_image_path

    return None


# 可运行对象包装
output_runnable = RunnableLambda(output_node)
