"""LLM 节点 - 使用 dashscope 官方 SDK 调用千问"""
import os
import json
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import dashscope
from dashscope import Generation


# 状态定义
class AgentState(TypedDict):
    """Agent 状态"""
    messages: List[BaseMessage]
    user_input: str
    image_path: Optional[str]
    tool_results: List[Dict[str, Any]]
    final_output: Optional[str]


def init_dashscope():
    """初始化 dashscope SDK"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_key:
        dashscope.api_key = api_key
    return api_key


def llm_node(state: AgentState) -> AgentState:
    """LLM 节点 - 调用千问大模型生成响应

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态
    """
    messages = state["messages"]
    user_input = state["user_input"]
    image_path = state.get("image_path", "")

    # 构建消息列表
    msg_list = []

    # 系统提示
    system_prompt = """你是专业的梦境生成助手，基于 DeepDream 算法为用户生成各种风格的梦境图片。

**核心功能：**
1. 梦境生成：用户输入包含「做梦/美梦/噩梦/迷雾梦/疯狂梦」关键词且上传图片时，调用 generate_dream 工具生成对应梦境图片
2. 知识科普：回答梦境相关知识和 DeepDream 算法原理，可调用 get_dream_knowledge 工具
3. 天气查询：查询城市天气，调用 get_weather 工具
4. 科学计算：进行数学计算，调用 scientific_calculator 工具

**梦境类型说明：**
- normal / 正常梦：基础效果，平衡真实与梦幻
- sweet / 美梦：柔和梦幻风格，带高斯模糊
- nightmare / 噩梦：恐怖锐利风格，带锐化滤镜
- mist / 迷雾梦：朦胧模糊风格，层次感强
- crazy / 疯狂梦：极致迷幻风格，视觉冲击强烈

**回复规则：**
- 用中文友好回复
- 工具结果格式化展示
- 如果用户要求生成梦境但没有上传图片，提醒用户上传图片
- 如果用户询问梦境或 DeepDream 相关知识，可以调用 get_dream_knowledge 工具获取详细科普内容
- 严格按照工具调用格式使用工具"""

    msg_list.append({"role": "system", "content": system_prompt})

    # 历史消息转换
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_list.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_list.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            msg_list.append({"role": "tool", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            msg_list.append({"role": "system", "content": msg.content})

    # 添加当前用户输入（如果有图片路径，附加到输入中）
    current_input = user_input
    if image_path and image_path not in user_input:
        current_input = f"{user_input}\n\n图片路径：{image_path}"

    # 如果最后一条不是用户消息，添加当前输入
    if not msg_list or msg_list[-1]["role"] != "user":
        msg_list.append({"role": "user", "content": current_input})

    try:
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

            # 检查是否有工具调用（qwen-turbo 可能直接返回文本，需要我们自己解析）
            # 这里我们使用简单的关键词匹配来判断是否需要调用工具
            ai_message = AIMessage(content=ai_content)

            # 尝试从内容中提取工具调用（如果模型输出了 JSON 格式的工具调用）
            tool_calls = _extract_tool_calls(ai_content)
            if tool_calls:
                ai_message.additional_kwargs["tool_calls"] = tool_calls

            state["messages"].append(ai_message)
        else:
            error_msg = f"API 调用失败：{response.message}"
            state["messages"].append(AIMessage(content=error_msg))

    except Exception as e:
        error_msg = f"调用大模型时出错：{str(e)}"
        state["messages"].append(AIMessage(content=error_msg))

    return state


def _extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    """从 AI 回复中提取工具调用

    Args:
        content: AI 回复内容

    Returns:
        工具调用列表
    """
    tool_calls = []

    # 简单的 JSON 提取（如果模型输出了工具调用 JSON）
    try:
        # 尝试查找 JSON 格式的工具调用
        import re
        json_pattern = r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}'
        matches = re.findall(json_pattern, content)

        for match in matches:
            try:
                tool_call = json.loads(match)
                if "name" in tool_call and "arguments" in tool_call:
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False)
                        }
                    })
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return tool_calls


# 可运行对象包装
llm_runnable = RunnableLambda(llm_node)
