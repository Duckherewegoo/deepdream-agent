"""LLM 节点 - 使用 dashscope 官方 SDK 调用千问"""
import logging
import os
import random
import time
from typing import Any

import dashscope
from dashscope import Generation
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..config import LLM, LLM_SEARCH, LLM_TEMP, LLM_TOP, SYSTEM_PROMPTS
from ..state import AgentState
from ..utils.meta_utils import load_dream_meta

logger = logging.getLogger(__name__)

# ================= 配置与常量 =================

# 抽离 System Prompt，保持代码干净
SYSTEM_PROMPT = SYSTEM_PROMPTS


def init_dashscope():
    """初始化 dashscope。DEBUG 模式下允许无 Key 启动，生产环境强制要求。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_key:
        dashscope.api_key = api_key
        logger.info("[INIT] dashscope API key 已设置")
        return api_key

    # 与 settings.py 的 DEBUG_MODE 语义保持一致
    debug_mode = os.getenv("DEEPDREAM_DEBUG", "false").lower() == "true"
    if debug_mode:
        logger.warning(
            "[INIT] 未配置 DASHSCOPE_API_KEY（DEEPDREAM_DEBUG=true），真实生成功能将不可用"
        )
        return ""
    raise ValueError("[INIT] 未找到 DASHSCOPE_API_KEY 环境变量！（生产环境必须配置）")


def build_dream_tone(meta: dict) -> str:
    """根据 .meta.json 生成 LLM 描述语气"""
    if not meta or "config" not in meta:
        return "画面呈现微妙变化，纹理缓慢流动"

    cfg = meta["config"]

    def get_float(data: dict, key: str, default: float) -> float:
        val = data.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # 修复：generation_time_sec 在 meta 顶层，而不是 cfg 里
    intensity = get_float(cfg, "STEP_SIZE", 0.01)
    chaos = get_float(cfg, "GRADIENT_NOISE", 0)
    fusion = get_float(cfg, "REALITY_FUSION", 0)
    scale = get_float(cfg, "GRADIENT_SCALE", 1.0)
    elapsed = get_float(meta, "generation_time_sec", 0)

    tone = []
    if intensity > 0.02:
        tone.append("画面极度扭曲，形态剧烈崩解")
    elif intensity > 0.01:
        tone.append("纹理明显增生，结构持续变形")
    else:
        tone.append("变化细微，仅在边缘缓慢流动")

    if chaos > 0.005:
        tone.append("噪声驱动形态抖动，画面不稳定")
    if fusion < 0.1:
        tone.append("现实感几乎消失，进入纯粹幻觉")
    if scale > 1.5:
        tone.append("高层语义特征主导，物体结构异化")
    if elapsed > 30:
        tone.append("长时间迭代导致细节极度繁复")

    return "；".join(tone)

def _extract_text(content: Any) -> str:
    """将 LangChain 可能包含的 list 多模态内容，安全降维为纯字符串"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 提取列表里的纯文本部分（兼容多模态消息）
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts) if texts else str(content)
    return str(content)


def _prepare_messages(messages: list) -> list:
    """清洗并转换 LangChain 历史消息为 DashScope 格式（防御加固版）"""
    # 全局唯一的 System Prompt
    msg_list = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in messages:
        content_str = _extract_text(getattr(msg, "content", ""))

        if isinstance(msg, HumanMessage):
            msg_list.append({"role": "user", "content": content_str})

        elif isinstance(msg, AIMessage):
            msg_list.append({"role": "assistant", "content": content_str})

        elif isinstance(msg, SystemMessage):
            # 外部的 System 一律丢弃，防止多 System 冲突
            continue

        elif isinstance(msg, ToolMessage):
            msg_list.append({
                "role": "user",
                "content": f"[系统工具返回结果: {content_str}]"
            })

        else:
            # 未知消息类型（如 FunctionMessage 等）兜底打包，绝不静默丢失
            msg_list.append({
                "role": "user",
                "content": f"[系统附加消息({type(msg).__name__}): {content_str}]"
            })

    return msg_list


def llm_node(state: AgentState) -> dict[str, Any]:
    logger.info("[NODE] LLM 节点开始执行")

    # 从 state 里读取数据（只读）
    user_input = state["user_input"]
    output_image_path = state.get("output_image_path", "")
    dream_type = state.get("dream_type", "")

    # 构建历史消息
    msg_list = _prepare_messages(state.get("messages", []))

    # 构建当前用户输入与系统上下文
    current_input = user_input
    if output_image_path and dream_type:
        dream_cn = {
            "normal": "正常梦", "sweet": "美梦", "nightmare": "噩梦",
            "mist": "迷雾梦", "crazy": "疯狂梦"
        }.get(dream_type, dream_type)

        meta = load_dream_meta(output_image_path)
        is_real_generated = bool(meta) and "config" in meta

        if not is_real_generated:
            current_input = (
                f"{user_input}\n\n"
                "[系统信息] 梦境生成失败（可能是算法错误或资源不足），已返回原图。"
                "请如实告知用户无法生成梦境，不要描述画面内容。"
            )
            logger.warning("[LLM] ⚠️ 进入 Fallback 模式：未检测到有效生成结果")
        else:
            tone_description = build_dream_tone(meta)
            current_input = (
                f"{user_input}\n\n"
                f"[系统信息] 已生成{dream_cn}。\n"
                f"生成特征：{tone_description}。\n"
                f"请严格根据以上特征描述画面，禁止提及生成过程、文件路径或技术参数。"
            )
            logger.info("[LLM] 进入正常描述模式")
    msg_list.append({"role": "user", "content": current_input})

    # 调用 API（带指数退避 + Jitter 重试）
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"[LLM] 正在调用千问 API (尝试 {attempt + 1}/{max_retries + 1})...")
            response = Generation.call(
                model=LLM,
                messages=msg_list,
                result_format='message',
                temperature=LLM_TEMP,
                top_p=LLM_TOP,
                enable_search=LLM_SEARCH,
            )

            if response.status_code == 200:
                ai_content = response.output.choices[0].message.content
                logger.info(f"[LLM] API 调用成功 | 回复长度: {len(ai_content)}")
                current_loop = state.get("loop_count", 0)
                # 返回增量字典，LangGraph 会自动合并
                return {
                    "messages": [AIMessage(content=ai_content)],
                    "status": "回复生成完成",
                    "loop_count": current_loop + 1
                }
            else:
                raise RuntimeError(f"API返回错误码 {response.status_code}: {response.message}")

        except Exception as e:
            logger.exception("[LLM] API 调用异常")

            # 最后一次尝试，直接返回报错状态
            if attempt == max_retries:
                error_msg = "大模型服务暂时不可用，请稍后再试。"
                logger.error(f"[LLM] 最终失败: {e!s}")
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "error": error_msg,
                    "status": "出错了"
                }

            # 指数退避 + Jitter
            # attempt 0 -> 睡 1~2 秒
            # attempt 1 -> 睡 2~3 秒
            delay = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"[LLM] 触发退避，等待 {delay:.2f} 秒后重试...")
            time.sleep(delay)

    # 理论上进不来，但为了类型检查，给个兜底
    return {"status": "未知状态"}

