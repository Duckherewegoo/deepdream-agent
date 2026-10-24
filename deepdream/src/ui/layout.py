"""
layout.py — 页面布局与事件绑定
组装 Gradio 组件，并绑定业务逻辑服务层。
"""
import logging
from typing import TYPE_CHECKING

import gradio as gr
from deepdream.src.infra.logger_setup import get_ui_logs
from deepdream.src.ui.components import render_meta_card

if TYPE_CHECKING:
    from deepdream.src.services.chat_service import ChatService

logger = logging.getLogger(__name__)

def create_ui(chat_service: "ChatService") -> gr.Blocks:
    """
    构建并返回 Gradio 界面。
    """
    logger.info("正在创建 Gradio 界面...")

    # Gradio 6.0+ 的 theme 在 launch 时设置，这里只建 Blocks
    demo = gr.Blocks(title="🌙 LangGraph · DeepDream 梦境生成助手")

    with demo:
        gr.Markdown("# 🌙 LangGraph 架构 · DeepDream 梦境生成助手")
        gr.Markdown("**基于 LangGraph + PyTorch 的工程化重构版本**")

        with gr.Row():
            status_box = gr.Textbox(label="📊 当前状态", interactive=False)
            log_box = gr.Textbox(
                label="📜 算法实时日志", interactive=False,
                lines=6, autoscroll=True, container=True
            )

        with gr.Row():
            # ================= 左列：输入区 =================
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入区")
                user_img = gr.Image(type="pil", label="📷 上传图片", height=300)
                user_input = gr.Textbox(label="💬 输入指令", lines=4)
                with gr.Row():
                    submit_btn = gr.Button("🚀 生成", variant="primary")
                    clear_btn = gr.Button("🧹 清空")

            # ================= 右列：输出区 =================
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输出区")
                output_text = gr.Markdown()
                output_img = gr.Image(type="pil", label="🎨 生成结果", height=300)
                meta_card = gr.Markdown(label="📋 梦境参数卡片")

        # ================= 快捷按钮区 =================
        gr.Markdown("### ⚡ 快捷操作")
        with gr.Row():
            gr.Button("🌙 正常梦").click(lambda: "生成正常梦", outputs=user_input)
            gr.Button("💖 美梦").click(lambda: "生成美梦", outputs=user_input)
            gr.Button("👻 噩梦").click(lambda: "生成噩梦", outputs=user_input)
            gr.Button("🌫️ 迷雾梦").click(lambda: "生成迷雾梦", outputs=user_input)
            gr.Button("🤪 疯狂梦").click(lambda: "生成疯狂梦", outputs=user_input)

        # ================= 复现区 =================
        gr.Markdown("### 🎲 同参数复现")
        with gr.Row():
            reproduce_img = gr.Image(type="pil", label="新图片（可选）", height=150)
            reproduce_btn = gr.Button("🎲 复现", variant="secondary")

        # ================= 事件回调封装 =================
        # 这些内部函数负责调用 Service，并把结果解包给 Gradio 的组件

        def handle_chat(text, img):
            res = chat_service.process_chat(text, img)
            meta_html = render_meta_card(res["meta"])
            return res["text"], res["image"], res["status"], meta_html, get_ui_logs()

        def handle_reproduce(orig_img, new_img):
            res = chat_service.reproduce_dream(orig_img, new_img)
            if not res["success"]:
                # 如果失败，保持原图不变，给出错误提示
                return orig_img, f"❌ {res['error_msg']}", get_ui_logs()

            meta_html = render_meta_card(res["meta"])
            return res["image"], meta_html, get_ui_logs()

        def clear_all():
            # 顺便清空底层 Service 的缓存
            if hasattr(chat_service, "last_output_path"):
                chat_service.last_output_path = None
            # 返回多一个空字符串给 output_text
            return None, "", "等待输入...", None, "", "等待输入...", "", ""



        # ================= 绑定事件 =================
        # 定时刷新日志
        timer = gr.Timer(0.5, active=True)
        timer.tick(fn=get_ui_logs, outputs=log_box)
        clear_btn.click(
            clear_all,
            outputs=[user_img, user_input, status_box, output_img, output_text, meta_card, log_box, reproduce_img]
        )

        submit_btn.click(
            handle_chat,
            inputs=[user_input, user_img],
            outputs=[output_text, output_img, status_box, meta_card, log_box]
        )

        reproduce_btn.click(
            handle_reproduce,
            inputs=[output_img, reproduce_img],
            outputs=[output_img, meta_card, log_box]
        )

    logger.info("Gradio 界面创建完成")
    return demo
