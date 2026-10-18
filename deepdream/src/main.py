"""
main.py — DeepDream 梦境生成助手（程序入口）
基于 LangGraph + PyTorch 的工程化重构版本

⚠️ 职责说明：
本文件只负责“组装依赖”与“启动服务”，不包含任何具体的 UI 布局或业务逻辑。
"""
import logging
import os
import sys

# ======================= 路径设置 =======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path 调整必须放在 import 业务模块之前
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


def main():
    """
    主入口函数。
    将重型业务模块的 import 放在函数内（延迟导入），既满足了 Ruff 的 E402 规范，
       又避免了模块加载时的循环依赖和启动耗时。
    """
    # ======================= 延迟导入业务组件 =======================
    import gradio as gr

    from deepdream.src.agent import DreamAgent
    from deepdream.src.infra.logger_setup import setup_logger
    from deepdream.src.services.chat_service import ChatService
    from deepdream.src.ui.layout import create_ui

    # ======================= 初始化基础设施 =======================
    setup_logger()

    logger.info("=" * 60)
    logger.info("DeepDream 梦境生成助手启动中...")

    # ======================= 准备运行环境 =======================
    upload_dir = os.path.join(PROJECT_ROOT, "uploads")
    output_dir = os.path.join(PROJECT_ROOT, "dream_outputs")

    for d in [upload_dir, output_dir]:
        os.makedirs(d, exist_ok=True)
        logger.info(f"确保目录存在: {d}")

    # ======================= 依赖组装 =======================
    logger.info("正在初始化 DreamAgent...")
    agent = DreamAgent(
        upload_dir=upload_dir,
        output_dir=output_dir,
    )

    logger.info("正在装配 ChatService...")
    chat_service = ChatService(agent=agent, upload_dir=upload_dir)

    # ======================= 构建并启动 UI =======================
    demo = create_ui(chat_service)

    logger.info("访问地址: http://127.0.0.1:7860")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=True,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
