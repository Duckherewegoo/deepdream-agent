"""意图识别节点 - 检测用户输入中的关键词，判断是否需要生成梦境"""
import os
import traceback
from typing import Optional
from langchain_core.runnables import RunnableLambda

from .llm_node import AgentState
from ..config.settings import DREAM_KEYWORDS


def intent_node(state: AgentState) -> AgentState:
    """意图识别节点 - 检测用户输入中的梦境关键词

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态
    """
    print(f"\n[NODE] 意图识别节点开始执行")
    print(f"[NODE] 用户输入: {state['user_input']}")
    print(f"[NODE] 是否有图片: {'是' if state.get('image_path') else '否'}")

    state["status"] = "正在分析您的请求..."

    user_input = state["user_input"]
    image_path = state.get("image_path")

    # 检测梦境类型
    detected_dream_type = None
    for dream_type, keywords in DREAM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in user_input:
                detected_dream_type = dream_type
                print(f"[INTENT] 检测到梦境关键词: {keyword} -> 类型: {dream_type}")
                break
        if detected_dream_type:
            break

    # 判断是否需要生成梦境
    need_dream_gen = False
    if detected_dream_type and image_path and os.path.exists(image_path):
        need_dream_gen = True
        print(f"[INTENT] 需要生成梦境: 是 (类型: {detected_dream_type})")
        state["status"] = f"检测到{detected_dream_type}，准备生成梦境图片..."
    elif detected_dream_type and not image_path:
        print(f"[INTENT] 检测到梦境关键词，但没有上传图片")
        state["status"] = "检测到梦境请求，但未上传图片"
    else:
        print(f"[INTENT] 不需要生成梦境")
        state["status"] = "正在处理您的请求..."

    # 更新状态
    state["dream_type"] = detected_dream_type
    state["need_dream_gen"] = need_dream_gen

    print(f"[NODE] 意图识别节点执行完成")
    return state


def should_generate_dream(state: AgentState) -> str:
    """条件判断：是否需要生成梦境

    Args:
        state: 当前 Agent 状态

    Returns:
        下一个节点名称："dream_gen" 或 "llm"
    """
    if state.get("need_dream_gen", False):
        print(f"[ROUTE] 路由到：梦境生成节点")
        return "dream_gen"
    else:
        print(f"[ROUTE] 路由到：LLM 节点")
        return "llm"


# 可运行对象包装
intent_runnable = RunnableLambda(intent_node)
should_generate_dream_runnable = RunnableLambda(should_generate_dream)
