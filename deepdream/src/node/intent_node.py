"""意图识别节点 - 检测用户输入中的关键词，判断是否需要生成梦境"""
import logging
import os

from ..config.settings import DREAM_KEYWORDS

# 注意：如果你按照之前建议把 state 移到了 core 层，这里改成 from ..core.state import AgentState
from ..state import AgentState

logger = logging.getLogger(__name__)


def intent_node(state: AgentState) -> AgentState:
    """意图识别节点 - 检测用户输入中的梦境关键词"""
    logger.info("[NODE] 意图识别节点开始执行")

    user_input = state.get("user_input", "")
    image_path = state.get("image_path")

    logger.info(f"[INTENT] 收到请求 | 文本长度: {len(user_input)} | 图片: {'有' if image_path else '无'}")

    state["status"] = "正在分析您的请求..."

    # ================= 1. 智能关键词匹配 =================
    detected_dream_type = None
    matches = []

    # 收集所有匹配到的关键词
    for dream_type, keywords in DREAM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in user_input:
                matches.append((keyword, dream_type))

    if matches:
        # 核心优化：按匹配到的关键词长度降序排序，取最精确的结果
        # 例如：同时匹配到“噩梦”和“美梦”，如果“美梦”的触发词更长，优先选美梦
        matches.sort(key=lambda x: len(x[0]), reverse=True)
        best_match_keyword, detected_dream_type = matches[0]
        logger.info(f"[INTENT] 检测到梦境关键词: '{best_match_keyword}' -> 类型: {detected_dream_type}")

    # ================= 2. 判断执行路径 =================
    need_dream_gen = False

    if detected_dream_type:
        if image_path and os.path.exists(image_path):
            need_dream_gen = True
            logger.info(f"[INTENT] 判定生成梦境 | 目标类型: {detected_dream_type}")
            state["status"] = f"检测到{detected_dream_type}意图，准备生成梦境..."
        else:
            logger.warning("[INTENT] 用户请求了梦境，但未上传图片或图片已丢失")
            state["status"] = "检测到梦境请求，但未上传有效图片"
    else:
        logger.info("[INTENT] 未检测到梦境生成意图，进入普通聊天模式")
        state["status"] = "正在处理您的请求..."

    # ================= 3. 更新并返回状态 =================
    state["dream_type"] = detected_dream_type
    state["need_dream_gen"] = need_dream_gen

    logger.info("[NODE] 意图识别节点执行完成")
    return state


def should_generate_dream(state: AgentState) -> str:
    """条件判断：路由到下一个节点"""
    if state.get("need_dream_gen", False):
        logger.info("[ROUTE] 路由决策 -> 进入 [梦境生成] 分支")
        return "dream_gen"

    logger.info("[ROUTE] 路由决策 -> 进入 [LLM 聊天] 分支")
    return "llm"

