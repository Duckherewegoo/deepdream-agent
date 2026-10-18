"""输出节点 - 处理最终输出"""
import logging
import os

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ..state.state import AgentState

logger = logging.getLogger(__name__)


def output_node(state: AgentState) -> AgentState:
    """输出节点 - 生成最终输出"""
    logger.info("[NODE] 输出节点开始执行")

    messages = state.get("messages", [])
    output_image_path = state.get("output_image_path")
    dream_type = state.get("dream_type")
    error = state.get("error")

    logger.debug(f"[OUTPUT] 图片路径: {output_image_path} | 梦境类型: {dream_type}")
    if error:
        logger.warning(f"[OUTPUT] 捕获到错误信息: {error}")

    # 1. 获取最后一条 AI 消息作为基础输出
    last_ai_message = None
    for msg in reversed(messages):
        # 优化：使用 isinstance 替代硬编码的 "ai" 字符串
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    if last_ai_message is None:
        logger.warning("[OUTPUT] 消息记录中没有找到 AI 消息，使用默认兜底回复")
        state["final_output"] = "抱歉，我好像走神了，没有生成任何回复。请重新尝试。"
        return state

    final_output = last_ai_message.content

    # 2. 处理错误信息（避免重复提示）
    if error and error not in final_output:
        # 优化：如果 LLM 节点的报错已经包含在 AI 的回复里了，就不再重复追加
            final_output = f"{final_output}\n\n⚠️ **注意**：{error}"
            logger.debug("[OUTPUT] 已补充外部错误信息到最终输出")

    state["final_output"] = final_output
    logger.info(f"[OUTPUT] 最终输出组装完成 | 长度: {len(final_output)}")

    return state


def get_output_image_path(state: AgentState) -> str | None:
    """从状态中获取生成的图片路径，并确保文件真实存在"""
    output_image_path = state.get("output_image_path")

    if not output_image_path:
        return None

    try:
        # 优化：增加 OSError 防御，防止因为非法路径导致节点崩溃
        if os.path.exists(output_image_path):
            return output_image_path
        else:
            logger.warning(f"[OUTPUT] 状态中记录的图片路径不存在: {output_image_path}")
            return None
    except OSError as e:
        logger.error(f"[OUTPUT] 检查图片路径时发生系统错误: {e}")
        return None


# 可运行对象包装
output_runnable = RunnableLambda(output_node)
