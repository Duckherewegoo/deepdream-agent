"""DeepDream 梦境生成助手 - 主程序入口
基于 LangGraph + PyTorch 的工程化重构版本
"""
import os
import sys
import traceback

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
from PIL import Image

from deepdream.src.agent import DreamAgent


# 初始化 Agent（使用绝对路径）
print(f"[MAIN] 项目根目录: {PROJECT_ROOT}")
print(f"[MAIN] 正在初始化 DreamAgent...")

agent = DreamAgent(
    upload_dir=os.path.join(PROJECT_ROOT, "uploads"),
    output_dir=os.path.join(PROJECT_ROOT, "dream_outputs"),
)

print(f"[MAIN] DreamAgent 初始化完成")


def process_chat(user_text: str, user_image: Image.Image):
    """处理用户对话

    Args:
        user_text: 用户输入文本
        user_image: 用户上传的图片

    Returns:
        (AI回复文本, 生成的图片PIL对象, 状态文字) 元组
    """
    print(f"\n{'='*60}")
    print(f"[MAIN] 收到用户请求")
    print(f"[MAIN] 用户文本: {user_text[:100] if user_text else '(空)'}")
    print(f"[MAIN] 是否有图片: {'是' if user_image else '否'}")
    print(f"{'='*60}\n")

    status_text = "正在处理..."

    try:
        # 调用 Agent
        response_text, output_image_path, final_state = agent.chat(user_text, user_image)

        # 获取状态文字
        status_text = final_state.get("status", "处理完成")

        # 加载生成的图片（如果有）
        output_image = None
        if output_image_path and os.path.exists(output_image_path):
            print(f"[MAIN] 加载生成的图片: {output_image_path}")
            output_image = Image.open(output_image_path)
            print(f"[MAIN] 图片尺寸: {output_image.size}")
        else:
            print(f"[MAIN] 没有生成图片或图片不存在")

        print(f"\n[MAIN] 处理完成")
        print(f"[MAIN] 回复长度: {len(response_text)}")
        print(f"[MAIN] 状态: {status_text}")

        return response_text, output_image, status_text

    except Exception as e:
        error_msg = f"❌ 执行错误：{str(e)}"
        print(f"[MAIN] 异常: {error_msg}")
        traceback.print_exc()

        # 返回详细的错误信息
        full_error = f"""❌ **执行出错了！**

**错误信息：** {str(e)}

**错误详情：**
```
{traceback.format_exc()}
```

**请检查：**
1. DASHSCOPE_API_KEY 环境变量是否设置
2. 网络连接是否正常
3. 图片格式是否支持
"""
        return full_error, None, "出错了"


def create_demo():
    """创建 Gradio 界面"""
    print(f"[MAIN] 正在创建 Gradio 界面...")

    # 配置静态文件目录，确保生成的图片可以被访问
    dream_outputs_dir = os.path.join(PROJECT_ROOT, "dream_outputs")
    uploads_dir = os.path.join(PROJECT_ROOT, "uploads")

    with gr.Blocks(
        title="🌙 LangGraph · DeepDream 梦境生成助手",
        theme=gr.themes.Soft(),
    ) as demo:
        # 标题区
        gr.Markdown("""
        # 🌙 LangGraph 架构 · DeepDream 梦境生成助手
        **基于 LangGraph + PyTorch 的工程化重构版本**

        ✨ **支持功能**：
        - 5种梦境生成（正常梦/美梦/噩梦/迷雾梦/疯狂梦）
        - 梦境知识科普 & DeepDream 算法原理
        - 天气查询
        - 科学计算器
        """)

        # 状态显示区
        status_box = gr.Textbox(
            label="📊 当前状态",
            value="等待输入...",
            interactive=False,
            show_label=True,
        )

        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入区")

                user_img = gr.Image(
                    type="pil",
                    label="📷 上传图片（生成梦境必选）",
                    height=300,
                    interactive=True,
                )

                user_input = gr.Textbox(
                    label="💬 输入指令",
                    placeholder="示例：\n- 生成美梦\n- 查询北京天气\n- 计算 sin(pi/2) + sqrt(16)\n- 讲讲 DeepDream 算法原理",
                    lines=5,
                    interactive=True,
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 生成", variant="primary", size="lg")
                    clear_btn = gr.Button("🧹 清空", size="lg")

            # 右侧：输出区
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输出区")

                gr.Markdown("**🤖 AI 回复**")
                output_text = gr.Markdown(value="等待输入...")

                gr.Markdown("**🎨 生成的梦境图片**")
                output_img = gr.Image(
                    type="pil",
                    height=300,
                    interactive=False,
                    show_label=False,
                )

        # 快捷按钮 - 梦境类型
        gr.Markdown("### ⚡ 快捷梦境")
        with gr.Row():
            btn_normal = gr.Button("🌙 正常梦")
            btn_sweet = gr.Button("💖 美梦")
            btn_nightmare = gr.Button("👻 噩梦")
            btn_mist = gr.Button("🌫️ 迷雾梦")
            btn_crazy = gr.Button("🤪 疯狂梦")

        # 快捷按钮 - 其他功能
        gr.Markdown("### 🔧 快捷工具")
        with gr.Row():
            btn_knowledge = gr.Button("📚 DeepDream 科普")
            btn_weather = gr.Button("🌤️ 天气查询")
            btn_calc = gr.Button("🔢 科学计算")

        # 绑定梦境快捷按钮
        btn_normal.click(lambda: "帮我生成一个正常梦", outputs=user_input)
        btn_sweet.click(lambda: "帮我生成一个美梦", outputs=user_input)
        btn_nightmare.click(lambda: "帮我生成一个噩梦", outputs=user_input)
        btn_mist.click(lambda: "帮我生成一个迷雾梦", outputs=user_input)
        btn_crazy.click(lambda: "帮我生成一个疯狂梦", outputs=user_input)

        # 绑定其他快捷按钮
        btn_knowledge.click(lambda: "给我讲讲 DeepDream 算法原理", outputs=user_input)
        btn_weather.click(lambda: "查询北京的天气", outputs=user_input)
        btn_calc.click(lambda: "计算 sin(pi/2) + sqrt(16) + log(e)", outputs=user_input)

        # 清空按钮
        def clear_all():
            print(f"[MAIN] 清空所有内容")
            return None, "", "等待输入...", None, "等待输入..."

        clear_btn.click(
            clear_all,
            outputs=[user_img, user_input, output_text, output_img, status_box],
        )

        # 提交按钮
        submit_btn.click(
            process_chat,
            inputs=[user_input, user_img],
            outputs=[output_text, output_img, status_box],
        )

        # 支持回车提交
        user_input.submit(
            process_chat,
            inputs=[user_input, user_img],
            outputs=[output_text, output_img, status_box],
        )

    print(f"[MAIN] Gradio 界面创建完成")
    return demo


def main():
    """主函数"""
    print(f"\n{'='*60}")
    print(f"[MAIN] DeepDream 梦境生成助手启动")
    print(f"[MAIN] 项目根目录: {PROJECT_ROOT}")
    print(f"{'='*60}\n")

    # 创建必要目录
    for d in ["uploads", "dream_outputs"]:
        dir_path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[MAIN] 确保目录存在: {dir_path}")

    # 创建并启动 Gradio 界面
    demo = create_demo()

    print(f"\n[MAIN] 启动 Gradio 服务器...")
    print(f"[MAIN] 访问地址: http://127.0.0.1:7860")
    print()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
