"""LLM 节点 - 使用 dashscope 官方 SDK 调用千问"""
import os
import json
import traceback
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
import dashscope
from dashscope import Generation


# 状态定义
class AgentState(TypedDict):
    """Agent 状态"""
    messages: List[BaseMessage]
    user_input: str
    image_path: Optional[str]          # 用户上传的图片路径
    output_image_path: Optional[str]   # 生成的梦境图片路径
    dream_type: Optional[str]          # 检测到的梦境类型
    need_dream_gen: bool               # 是否需要生成梦境
    tool_results: List[Dict[str, Any]]
    final_output: Optional[str]        # 最终文本输出
    status: str                        # 当前状态（用于进度显示）
    error: Optional[str]               # 错误信息


def init_dashscope():
    """初始化 dashscope SDK"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_key:
        dashscope.api_key = api_key
    print(f"[INIT] dashscope API key 已设置: {'是' if api_key else '否'}")
    return api_key


def llm_node(state: AgentState) -> AgentState:
    """LLM 节点 - 调用千问大模型生成响应

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态
    """
    print(f"\n[NODE] LLM 节点开始执行")
    print(f"[NODE] 用户输入: {state['user_input'][:50]}...")
    print(f"[NODE] 是否需要生成梦境: {state.get('need_dream_gen', False)}")
    print(f"[NODE] 梦境类型: {state.get('dream_type', '无')}")
    print(f"[NODE] 生成图片路径: {state.get('output_image_path', '无')}")

    state["status"] = "正在生成回复..."

    messages = state["messages"]
    user_input = state["user_input"]
    image_path = state.get("image_path", "")
    output_image_path = state.get("output_image_path", "")
    dream_type = state.get("dream_type", "")

    # 构建消息列表
    msg_list = []

    # 系统提示
    system_prompt = """你是专业的梦境生成助手，基于 DeepDream 算法为用户生成各种风格的梦境图片。

**核心功能：**
1. 梦境生成：根据用户选择的梦境类型生成对应的 DeepDream 图片
2. 知识科普：回答梦境相关知识和 DeepDream 算法原理
3. 天气查询：查询城市天气
4. 科学计算：进行数学计算

**回复规则：**
- 用中文友好回复
- 如果已经生成了梦境图片，告诉用户图片已生成，并简单描述梦境特点
- 如果用户要求生成梦境但没有上传图片，提醒用户上传图片
- 回答要简洁明了，不要太啰嗦
- 不要提到你调用了什么工具，直接给出结果"""

    msg_list.append({"role": "system", "content": system_prompt})

    # 历史消息转换
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_list.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_list.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            msg_list.append({"role": "user", "content": f"[工具结果] {msg.content}"})
        elif isinstance(msg, SystemMessage):
            msg_list.append({"role": "system", "content": msg.content})

    # 构建当前用户输入
    current_input = user_input

    # 如果有梦境生成结果，添加到上下文中
    if output_image_path and dream_type:
        dream_cn = {
            "normal": "正常梦",
            "sweet": "美梦",
            "nightmare": "噩梦",
            "mist": "迷雾梦",
            "crazy": "疯狂梦"
        }.get(dream_type, dream_type)
        current_input = f"{user_input}\n\n[系统信息] 已经成功生成了{dream_cn}图片，保存在：{output_image_path}"

    # 如果只有图片路径但没有生成梦境
    elif image_path and not state.get("need_dream_gen"):
        current_input = f"{user_input}\n\n[系统信息] 用户上传了一张图片，路径：{image_path}"

    # 添加当前用户消息
    msg_list.append({"role": "user", "content": current_input})

    try:
        print(f"[LLM] 正在调用千问 API...")
        # 调用千问 API - 使用 dashscope 官方 SDK
        response = Generation.call(
            model="qwen-turbo",
            messages=msg_list,
            result_format='message',
            temperature=0.7,
            top_p=0.8,
            enable_search=False,
        )

        if response.status_code == 200:
            ai_content = response.output.choices[0].message.content
            print(f"[LLM] API 调用成功，回复长度: {len(ai_content)}")
            print(f"[LLM] 回复预览: {ai_content[:100]}...")

            ai_message = AIMessage(content=ai_content)
            state["messages"].append(ai_message)
            state["status"] = "回复生成完成"
        else:
            error_msg = f"API 调用失败：{response.message}"
            print(f"[LLM] 错误: {error_msg}")
            state["messages"].append(AIMessage(content=error_msg))
            state["error"] = error_msg
            state["status"] = "出错了"

    except Exception as e:
        error_msg = f"调用大模型时出错：{str(e)}"
        print(f"[LLM] 异常: {error_msg}")
        traceback.print_exc()
        state["messages"].append(AIMessage(content=error_msg))
        state["error"] = error_msg
        state["status"] = "出错了"

    print(f"[NODE] LLM 节点执行完成")
    return state


# 可运行对象包装
llm_runnable = RunnableLambda(llm_node)
