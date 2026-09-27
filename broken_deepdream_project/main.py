"""DeepDream 梦境生成助手 - 主程序入口
基于 LangGraph + PyTorch 的工程化重构版本
"""
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from PIL import Image

from deepdream.src.agent import DreamAgent


# 初始化 Agent
agent = DreamAgent()


def process_chat(user_text: str, user_image: Image.Image) -> tuple:
    """处理用户对话

    Args:
        user_text: 用户输入文本
        user_image: 用户上传的图片

    Returns:
        (AI回复, 生成的图片) 元组
    """
    try:
        # 调用 Agent
        response_text, output_image_path = agent.chat(user_text, user_image)

        # 加载生成的图片（如果有）
        output_image = None
        if output_image_path and os.path.exists(output_image_path):
            output_image = Image.open(output_image_path)

        return response_text, output_image

    except Exception as e:
        error_msg = f"❌ 执行错误：{str(e)}"
        return error_msg, None


def create_demo():
    """创建 Gradio 界面"""
    with gr.Blocks(title="🌙 LangGraph · DeepDream 梦境生成助手") as demo:
        gr.Markdown("""
        # 🌙 LangGraph 架构 · DeepDream 梦境生成助手

        **基于 LangGraph + PyTorch 的工程化重构版本**

        ✨ **支持功能**：
        - 5种梦境生成（正常梦/美梦/噩梦/迷雾梦/疯狂梦）
        - 梦境知识科普 & DeepDream 算法原理
        - 天气查询
        - 科学计算器

        🚀 **技术栈**：
        - LangGraph 1.2.6 + LangChain 1.3.11
        - PyTorch + InceptionV3
        - 千问大模型（dashscope 官方 SDK）
        - Gradio 界面
        """)

        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                user_img = gr.Image(
                    type="pil",
                    label="📷 上传图片（生成梦境必选）",
                    height=300,
                )
                user_input = gr.Textbox(
                    label="💬 输入指令",
                    placeholder="示例：\n- 生成美梦\n- 查询北京天气\n- 计算 sin(pi/2) + sqrt(16)\n- 讲讲 DeepDream 算法原理",
                    lines=5,
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 生成", variant="primary", size="lg")
                    clear_btn = gr.Button("🧹 清空", size="lg")

            # 右侧：输出区
            with gr.Column(scale=1):
                output_text = gr.Markdown(label="🤖 AI 回复")
                output_img = gr.Image(
                    type="pil",
                    label="🎨 生成的梦境图片",
                    height=300,
                )

        # 快捷按钮
        gr.Markdown("### ⚡ 快捷指令")
        with gr.Row():
            gr.Button("🌙 正常梦").click(
                lambda: ("帮我生成一个正常梦", None),
                outputs=[user_input, user_img]
            )
            gr.Button("💖 美梦").click(
                lambda: ("帮我生成一个美梦", None),
                outputs=[user_input, user_img]
            )
            gr.Button("👻 噩梦").click(
                lambda: ("帮我生成一个噩梦", None),
                outputs=[user_input, user_img]
            )
            gr.Button("🌫️ 迷雾梦").click(
                lambda: ("帮我生成一个迷雾梦", None),
                outputs=[user_input, user_img]
            )
            gr.Button("🤪 疯狂梦").click(
                lambda: ("帮我生成一个疯狂梦", None),
                outputs=[user_input, user_img]
            )

        with gr.Row():
            gr.Button("📚 DeepDream 科普").click(
                lambda: "给我讲讲 DeepDream 算法原理",
                outputs=user_input
            )
            gr.Button("🌤️ 天气查询").click(
                lambda: "查询北京的天气",
                outputs=user_input
            )
            gr.Button("🔢 科学计算").click(
                lambda: "计算 sin(pi/2) + sqrt(16) + log(e)",
                outputs=user_input
            )

        # 绑定事件
        submit_btn.click(
            process_chat,
            inputs=[user_input, user_img],
            outputs=[output_text, output_img]
        )

        clear_btn.click(
            lambda: (None, "", "", None),
            outputs=[user_img, user_input, output_text, output_img]
        )

        # 支持回车提交
        user_input.submit(
            process_chat,
            inputs=[user_input, user_img],
            outputs=[output_text, output_img]
        )

    return demo


def main():
    """主函数"""
    # 创建必要目录
    for d in ["./uploads", "./dream_outputs"]:
        os.makedirs(d, exist_ok=True)

    # 创建并启动 Gradio 界面
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
