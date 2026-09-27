import os
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from PIL import Image, ImageFilter
import gradio as gr
import uuid

# ============ LangChain 1.2.0 核心导入（无废弃/未使用） ============
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

# 模型客户端（LangChain 1.2.0 唯一官方路径，无社区版回退）
from langchain_openai import ChatOpenAI

# ============ 配置 ============
# 百炼模型客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "your-api-key-here"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ============ 安全组件（简化，降低圈复杂度） ============


class ErrorHandler:
    """简化的错误处理器"""

    def handle(self, error: Exception) -> str:
        msg = str(error).lower()
        if any(key in msg for key in ["timeout", "connection", "network"]):
            return "❌ 网络不稳定，请检查后重试"
        elif any(key in msg for key in ["parameter", "invalid", "missing"]):
            return "❌ 参数错误"
        elif any(key in msg for key in ["auth", "key"]):
            return "❌ API认证失败，请检查配置"
        elif any(key in msg for key in ["memory", "quota", "limit"]):
            return "❌ 资源不足"
        return f"❌ 系统错误: {str(error)[:100]}"


# ============ 梦境工具（移除未使用参数） ============
@tool
def generate_dream(dream_type: str = "normal") -> Dict[str, Any]:
    """生成梦境效果图片"""
    try:
        time.sleep(1)
        output_dir = "./dream_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"dream_{dream_type}_{int(time.time())}.png")

        img = Image.new('RGB', (512, 512), color='white')
        if dream_type == "sweet":
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        elif dream_type == "nightmare":
            img = img.point(lambda x: int(x * 1.5)
                            if x < 128 else int(x * 0.8))

        img.save(output_path)
        return {
            "success": True,
            "dream_type": dream_type,
            "output_path": output_path,
            "message": f"✨ {dream_type}梦境已生成！"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"❌ 梦境生成失败: {str(e)}"
        }


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    weather_data = {
        "北京": "🌤️ 北京：晴，15-25°C，适合做梦的好天气！",
        "上海": "🌧️ 上海：小雨，18-22°C，梦里听雨声~",
        "广州": "☀️ 广州：晴朗，25-32°C，热带梦境在等你！",
        "深圳": "⛅ 深圳：多云，24-30°C，科技感梦境启动！",
    }
    return weather_data.get(city, f"🌍 {city}：天气数据获取中，先做个美梦吧~")


# ============ 工具列表 ============
TOOLS = [generate_dream, get_weather]


# ============ 智能体链（LangChain 1.2.0 标准实现） ============
def create_simple_agent():
    """创建简化的智能体链"""
    system_prompt = """你是梦境生成助手，专注于：
    1. 生成梦境效果 (generate_dream)
    2. 查询天气 (get_weather)
    使用中文回复，友好幽默。"""

    # 初始化模型（LangChain 1.2.0 正确参数）
    try:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY", "sk-test"),
            base_url=os.getenv(
                "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
    except Exception:
        # 修复：MockLLM 实现 bind_tools 方法，兼容 1.2.0
        class MockLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, prompt):
                return AIMessage(content="这是一个模拟响应，请检查你的API配置")
        llm = MockLLM()

    llm_with_tools = llm.bind_tools(TOOLS)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 工具执行函数
    def execute_tool(tool_call: dict) -> str:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        for t in TOOLS:
            if t.name == tool_name:
                try:
                    result = t.invoke(tool_args)
                    return result["message"] if isinstance(result, dict) else str(result)
                except Exception:
                    return f"工具执行错误"
        return f"未找到工具: {tool_name}"

    # 主处理链
    def process_chain(input_data: Dict) -> str:
        try:
            messages = prompt.invoke({"input": input_data["input"]})
            response = llm_with_tools.invoke(messages)

            if hasattr(response, 'tool_calls') and response.tool_calls:
                results = [execute_tool({"name": tc["name"], "args": tc.get("args", {})})
                           for tc in response.tool_calls]
                return results[0] if len(results) == 1 else f"执行结果:\n" + "\n- ".join(results)
            return response.content if hasattr(response, 'content') else str(response)

        except Exception as e:
            return ErrorHandler().handle(e)

    return RunnableLambda(process_chain)


# ============ Gradio界面 ============
def save_uploaded_image(image) -> str:
    """保存上传的图片"""
    if image is None:
        return ""

    upload_dir = "./uploads"
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"upload_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(upload_dir, filename)

    if isinstance(image, str):
        import shutil
        shutil.copy2(image, filepath)
    else:
        image.save(filepath, "PNG")
    return filepath


def process_input(user_input: str, uploaded_image=None):
    """处理用户输入"""
    try:
        image_path = save_uploaded_image(
            uploaded_image) if uploaded_image else None
        full_input = f"{user_input}，图片路径：{image_path}" if image_path else user_input

        agent = create_simple_agent()
        return agent.invoke({"input": full_input})

    except Exception as e:
        return f"❌ 处理错误: {str(e)}"


def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="梦境生成助手") as demo:
        gr.Markdown("# 🌌 梦境生成助手")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📷 上传图片（可选）",
                    type="pil",
                    height=200
                )
                dream_type = gr.Dropdown(
                    label="🎭 梦境类型",
                    choices=["normal", "sweet",
                             "nightmare", "fantasy", "scifi"],
                    value="normal"
                )

            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    label="💭 输入指令",
                    placeholder="例如：'生成甜蜜梦境' 或 '北京天气怎么样？'",
                    lines=3
                )
                submit_btn = gr.Button("🚀 提交", variant="primary")
                clear_btn = gr.Button("🧹 清空")
                output = gr.Textbox(
                    label="🤖 AI响应",
                    lines=6,
                    interactive=False
                )

        # 示例
        gr.Examples(
            examples=[
                ["生成甜蜜梦境", "sweet"],
                ["生成奇幻梦境", "fantasy"],
                ["北京天气怎么样？", "normal"],
            ],
            inputs=[user_input, dream_type],
            label="💡 示例"
        )

        def process(user_text, image, d_type):
            if image and ("生成" in user_text or "dream" in user_text.lower()):
                full_input = f"为这张图片生成{d_type}梦境"
                return process_input(full_input, image)
            return process_input(user_text, None)

        submit_btn.click(fn=process, inputs=[
                         user_input, input_image, dream_type], outputs=output)
        clear_btn.click(fn=lambda: (None, "", "normal", ""),
                        outputs=[input_image, user_input, dream_type, output])

    return demo


# ============ 主程序 ============
if __name__ == "__main__":
    for dir_path in ["./dream_outputs", "./uploads"]:
        os.makedirs(dir_path, exist_ok=True)

    print("🚀 启动梦境生成助手...")
    print("📡 服务地址: http://localhost:7860")

    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
