"""输出节点 - 处理最终输出"""
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage

from ..state.state import AgentState

logger = logging.getLogger(__name__)


def output_node(state: AgentState) -> dict[str, Any]:  # 改成返回 dict
    """输出节点 - 生成最终输出"""
    logger.info("[NODE] 输出节点开始执行")

    messages = state.get("messages", [])
    output_image_path = state.get("output_image_path")
    error = state.get("error")

    # 倒序找最后一条 AI 消息
    last_ai_message = next(
        (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
        None
    )

    # 兜底：如果没有找到任何 AI 消息
    if last_ai_message is None:
        logger.warning("[OUTPUT] 消息记录中没有找到 AI 消息，使用默认兜底回复")
        return {"final_output": "抱歉，我好像走神了，没有生成任何回复。请重新尝试。"}

    # 安全提取内容（防多模态 List 导致报错）
    content = last_ai_message.content
    final_output = content if isinstance(content, str) else str(content)

    # 追加错误提示
    if error and error not in final_output:
        final_output = f"{final_output}\n\n⚠️ **注意**：{error}"
        logger.debug("[OUTPUT] 已补充外部错误信息到最终输出")

    if output_image_path and os.path.isfile(output_image_path):
        # Gradio 的 Markdown 默认拒载本地绝对路径图片，此处只做提示，
        # 图片展示交给 UI 的 gr.Image 组件，避免双重展示 + 破图。
        final_output = f"{final_output}\n\n🖼️ *图片已生成，请查看右侧预览区*"

    logger.info(f"[OUTPUT] 最终输出组装完成 | 长度: {len(final_output)}")

    # 只返回增量
    return {"final_output": final_output}

def get_output_image_path(state: AgentState) -> str | None:
    """从状态中获取生成的图片路径，并确保文件真实存在且合法"""
    output_image_path = state.get("output_image_path")

    if not output_image_path:
        return None

    try:
        # 1. 统一转为绝对路径，防止上层调用因工作目录变化而找不到文件
        abs_path = os.path.abspath(output_image_path)

        # 2. 必须存在，且必须是文件（排除目录）
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            # 3. 验证是否可读（可选，但极度严谨）
            if os.access(abs_path, os.R_OK):
                return abs_path
            else:
                logger.warning(f"[OUTPUT] 图片文件存在但无读取权限: {abs_path}")
                return None
        else:
            logger.warning(f"[OUTPUT] 状态中记录的图片路径不存在或不是文件: {abs_path}")
            return None

    except OSError as e:
        logger.error(f"[OUTPUT] 检查图片路径时发生系统错误: {e}")
        return None
