"""LLM 节点 - 使用 dashscope 官方 SDK 调用千问"""
import os
import traceback
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
import dashscope
from dashscope import Generation
from ..state.state import AgentState
from ..utils.meta_utils import load_dream_meta


def init_dashscope():
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if api_key:
        dashscope.api_key = api_key
    print(f"[INIT] dashscope API key 已设置: {'是' if api_key else '否'}")
    return api_key


def build_dream_tone(meta: dict) -> str:
    """根据.meta.json生成LLM描述语气"""
    if not meta or "config" not in meta:
        return "画面呈现微妙变化，纹理缓慢流动"

    cfg = meta["config"]

    def get_float(key, default):
        val = cfg.get(key, default)
        return float(val) if isinstance(val, str) else val

    intensity = get_float("STEP_SIZE", 0.01)
    chaos = get_float("GRADIENT_NOISE", 0)
    fusion = get_float("REALITY_FUSION", 0)
    scale = get_float("GRADIENT_SCALE", 1.0)
    elapsed = get_float("generation_time_sec", 0)

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


def llm_node(state: AgentState) -> AgentState:
    print(f"\n[NODE] LLM 节点开始执行")
    print(f"[NODE] 用户输入: {state['user_input'][:50]}...")
    print(f"[NODE] 是否需要生成梦境: {state.get('need_dream_gen', False)}")
    print(f"[NODE] 梦境类型: {state.get('dream_type', '无')}")
    print(f"[NODE] 生成图片路径: {state.get('output_image_path', '无')}")

    state["status"] = "正在生成回复..."

    messages = state["messages"]
    user_input = state["user_input"]
    output_image_path = state.get("output_image_path", "")
    dream_type = state.get("dream_type", "")

    # ========== 系统提示 ==========
    system_prompt = """你是专业的梦境解说员，专注于描述 DeepDream 生成的超现实主义图像。

**核心任务：**
当用户生成了梦境图片时，你必须根据梦境类型，用精准、故事感、富有画面感的文字描述图像内容。
禁止说“图片已生成”“已经成功生成”等废话，直接描写画面。
不许编造虚假信息，传播有害信息！

**描述规则：**
- 多用和梦境有关的词汇，比如“朦朦胧胧”，“奇幻”，“迷离”，“诡异”等等
- 对于梦境图片描述尽量具体：纹理、形状、颜色、空间关系
- 字数控制在 40–60 字之间
- 语感必须匹配梦境类型

**各类型语感指引：**
- 美梦（sweet）：柔软、流动、光感。
- 噩梦（nightmare）：尖锐、冰冷、崩塌。
- 迷雾梦（mist）：朦胧、弥散、呼吸。
- 疯狂梦（crazy）：混乱、爆炸、神经质。
- 正常梦（normal）：克制、平衡、微妙。

**其他功能：**
- 若用户询问 DeepDream 原理或梦境知识，简明科普
- 若用户未上传图片却要求生成梦境，礼貌提醒
- 若用户闲聊，和梦相关的话题要自然回应，其余一切话题都不许回答！"""

    msg_list = [{"role": "system", "content": system_prompt}]

    # 历史消息转换
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_list.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_list.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            msg_list.append(
                {"role": "user", "content": f"[工具结果] {msg.content}"})
        elif isinstance(msg, SystemMessage):
            msg_list.append({"role": "system", "content": msg.content})

    # ========== 构建当前用户输入 ==========
    current_input = user_input

    # 情况 1：生成了图片
    if output_image_path and dream_type:
        dream_cn = {
            "normal": "正常梦", "sweet": "美梦",
            "nightmare": "噩梦", "mist": "迷雾梦", "crazy": "疯狂梦"
        }.get(dream_type, dream_type)

        # ✅ 加载 meta
        meta = load_dream_meta(output_image_path)

        # ✅ 调试信息：打印 meta 状态
        print(f"[LLM] Meta 加载结果: {'成功' if meta else '失败（空字典）'}")
        if meta:
            print(f"[LLM] Meta 内容预览: {list(meta.keys())}")

        # ✅ 双重检查：meta 是否存在 + 文件是否正常生成
        is_real_generated = bool(meta)
        if is_real_generated and "config" not in meta:
            print(f"[LLM] ⚠️ Meta 文件存在但内容不完整，视为失败")
            is_real_generated = False

        print(f"[LLM] 生成有效性检查结果: {is_real_generated} (Meta存在: {bool(meta)})")

        if not is_real_generated:
            # ❌ 真正进入 Fallback
            current_input = (
                f"{user_input}\n\n"
                "[系统信息] 梦境生成失败（可能是算法错误或资源不足），已返回原图。"
                "请如实告知用户无法生成梦境，不要描述画面内容，也不要说'系统提示'。"
            )
            print(f"[LLM] ⚠️ 进入 Fallback 模式：未检测到有效生成结果")
        else:
            # ✅ 真正生成成功
            tone_description = build_dream_tone(meta)
            current_input = (
                f"{user_input}\n\n"
                f"[系统信息] 已生成{dream_cn}。\n"
                f"生成特征：{tone_description}。\n"
                f"请严格根据以上特征描述画面，禁止提及生成过程、文件路径或技术参数。"
            )
            print(f"[LLM] ✅ 进入正常描述模式")

    msg_list.append({"role": "user", "content": current_input})

    # ========== 调用 LLM ==========
    try:
        print(f"[LLM] 正在调用千问 API...")
        response = Generation.call(
            model="qwen-turbo",
            messages=msg_list,
            result_format='message',
            temperature=0.65,
            top_p=0.8,
            enable_search=False,
        )

        if response.status_code == 200:
            ai_content = response.output.choices[0].message.content
            print(f"[LLM] API 调用成功，回复长度: {len(ai_content)}")
            print(f"[LLM] 回复预览: {ai_content[:100]}...")

            state["messages"].append(AIMessage(content=ai_content))
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
