"""DeepDream 梦境生成助手 - 主程序入口
基于 LangGraph + PyTorch 的工程化重构版本
"""
from deepdream.src.agent import DreamAgent
from deepdream.src.utils.meta_utils import load_dream_meta
import os
import sys
import traceback
from collections import deque
from PIL import Image

import gradio as gr
# ======================= 全局日志系统（修复版） =======================


class DreamLogger:
    """轻量级日志收集器，兼容 uvicorn 的 isatty 检查"""

    def __init__(self, max_lines=300):
        self.logs = deque(maxlen=max_lines)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, text):
        # 只捕获我们关心的 DeepDream 日志
        stripped = text.strip()
        if stripped.startswith("[DREAM]") or stripped.startswith("[MAIN]"):
            self.logs.append(stripped)
        # 同时写到控制台（保证终端可见）
        self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

    def isatty(self):
        """关键修复：uvicorn 需要这个方法检查终端颜色支持"""
        return self.original_stdout.isatty()

    def get_logs(self) -> str:
        """供 Gradio 调用的接口"""
        return "\n".join(self.logs)


# 实例化全局日志（在 import gradio 之前）
dream_logger = DreamLogger()
sys.stdout = dream_logger
# ⚠️ 不重定向 stderr，保证异常堆栈正常显示

# ======================= 项目路径 & Agent =======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


print(f"[MAIN] 项目根目录: {PROJECT_ROOT}")
print("[MAIN] 正在初始化 DreamAgent...")

agent = DreamAgent(
    upload_dir=os.path.join(PROJECT_ROOT, "uploads"),
    output_dir=os.path.join(PROJECT_ROOT, "dream_outputs"),
)

print("[MAIN] DreamAgent 初始化完成")

# ======================= UI 辅助函数 =======================


def format_meta_for_ui(meta: dict) -> str:
    if not meta:
        return "📭 暂无梦境参数（图片未生成或Meta文件缺失）"
    cfg = meta.get("config", {})
    step_size = cfg.get("STEP_SIZE", 0)
    if step_size > 0.02:
        intensity = "极高"
    elif step_size > 0.01:
        intensity = "中等"
    else:
        intensity = "微弱"
    return f"""
<div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; background:#f8f9fa; font-size:14px; margin-top:10px;">
    <p style="margin:0 0 8px 0;"><b>🎭 梦境类型：</b>{meta.get('dream_type', '未知')}</p>
    <p style="margin:0 0 8px 0;"><b>⏱️ 生成耗时：</b>{meta.get('generation_time_sec', 0)}s</p>
    <p style="margin:0 0 8px 0;"><b>📐 迭代次数：</b>{cfg.get('ITER_PER_OCTAVE', 'N/A')}次/Octave</p>
    <p style="margin:0 0 8px 0;"><b>🌀 步长强度：</b>{step_size}（{intensity}）</p>
    <p style="margin:0 0 8px 0;"><b>🧠 特征层：</b><code>{', '.join(meta.get('layers_used', []))}</code></p>
</div>
"""


def reproduce_dream(original_img, new_upload_img):
    if not original_img:
        return None, "❌ 没有可复用的原图参数", ""
    meta = load_dream_meta(original_img)
    if not meta:
        return None, "❌ 未找到原图参数，无法复现", ""

    input_path = new_upload_img if new_upload_img else original_img
    if new_upload_img:
        temp_path = os.path.join(PROJECT_ROOT, "uploads", "temp_reproduce.png")
        new_upload_img.save(temp_path)
        input_path = temp_path

    from deepdream.src.deepdream.generator import DeepDreamGenerator
    generator = DeepDreamGenerator()
    output_path = generator.generate(
        input_path=input_path,
        dream_type=meta["dream_type"],
        custom_cfg=meta["config"],
        custom_layers=meta["layers_used"]
    )
    new_img = Image.open(output_path)
    new_meta = load_dream_meta(output_path)
    return new_img, format_meta_for_ui(new_meta), dream_logger.get_logs()


def process_chat(user_text: str, user_image: Image.Image):
    status_text = "正在处理..."
    try:
        response_text, output_image_path, final_state = agent.chat(
            user_text, user_image)
        status_text = final_state.get("status", "处理完成")
        output_image = None
        if output_image_path and os.path.exists(output_image_path):
            output_image = Image.open(output_image_path)
        meta = load_dream_meta(output_image_path) if output_image_path else {}
        meta_content = format_meta_for_ui(meta)
        return response_text, output_image, status_text, meta_content, dream_logger.get_logs()
    except Exception as e:
        error_msg = f"❌ 执行错误：{str(e)}"
        traceback.print_exc()
        full_error = f"❌ **执行出错了！**\n\n**错误信息：** {str(e)}\n\n**详情：**\n```\n{traceback.format_exc()}\n```"
        return full_error, None, "出错了", "❌ 生成失败", dream_logger.get_logs()


def create_demo():
    print("[MAIN] 正在创建 Gradio 界面...")

    # ✅ 只在这里 import gradio，用完即走

    # ✅ Gradio 6.0+：theme 作为参数传给 launch，不在 Blocks 里设置
    demo = gr.Blocks(title="🌙 LangGraph · DeepDream 梦境生成助手")

    with demo:
        gr.Markdown("# 🌙 LangGraph 架构 · DeepDream 梦境生成助手")
        gr.Markdown("**基于 LangGraph + PyTorch 的工程化重构版本**")

        # 状态栏 + 日志实时显示
        with gr.Row():
            status_box = gr.Textbox(label="📊 当前状态", interactive=False)
            log_box = gr.Textbox(
                label="📜 算法实时日志",
                interactive=False,
                lines=6,
                autoscroll=True,
                container=True
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入区")
                user_img = gr.Image(type="pil", label="📷 上传图片", height=300)
                user_input = gr.Textbox(label="💬 输入指令", lines=4)
                with gr.Row():
                    submit_btn = gr.Button("🚀 生成", variant="primary")
                    clear_btn = gr.Button("🧹 清空")

            with gr.Column(scale=1):
                gr.Markdown("### 📤 输出区")
                output_text = gr.Markdown()
                output_img = gr.Image(type="pil", label="🎨 生成结果", height=300)
                meta_card = gr.Markdown(label="📋 梦境参数卡片")

        # 快捷按钮区
        gr.Markdown("### ⚡ 快捷操作")
        with gr.Row():
            gr.Button("🌙 正常梦").click(lambda: "生成正常梦", outputs=user_input)
            gr.Button("💖 美梦").click(lambda: "生成美梦", outputs=user_input)
            gr.Button("👻 噩梦").click(lambda: "生成噩梦", outputs=user_input)
            gr.Button("🌫️ 迷雾梦").click(lambda: "生成迷雾梦", outputs=user_input)
            gr.Button("🤪 疯狂梦").click(lambda: "生成疯狂梦", outputs=user_input)

        # 复现区
        gr.Markdown("### 🎲 同参数复现")
        with gr.Row():
            reproduce_img = gr.Image(type="pil", label="新图片（可选）", height=150)
            reproduce_btn = gr.Button("🎲 复现", variant="secondary")

        # ======================= 事件绑定 =======================
        # ✅ 核心：定时器，每0.5秒抓取一次日志
        timer = gr.Timer(0.5, active=True)
        timer.tick(fn=dream_logger.get_logs, outputs=log_box)

        def clear_all():
            return None, "", "等待输入...", None, "等待输入...", "", ""

        clear_btn.click(
            clear_all,
            outputs=[user_img, user_input, status_box,
                     output_img, meta_card, log_box, reproduce_img]
        )

        submit_btn.click(
            process_chat,
            inputs=[user_input, user_img],
            outputs=[output_text, output_img, status_box, meta_card, log_box]
        ).then(
            fn=dream_logger.get_logs,
            outputs=log_box
        )

        reproduce_btn.click(
            reproduce_dream,
            inputs=[output_img, reproduce_img],
            outputs=[output_img, meta_card, log_box]
        )

    print("[MAIN] Gradio 界面创建完成")
    return demo


def main():
    print(f"\n{'='*60}")
    print("[MAIN] DeepDream 梦境生成助手启动")

    # 创建必要目录
    for d in ["uploads", "dream_outputs"]:
        dir_path = os.path.join(PROJECT_ROOT, d)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[MAIN] 确保目录存在: {dir_path}")

    demo = create_demo()
    print("[MAIN] 访问地址: http://127.0.0.1:7860")

    # ✅ Gradio 6.0+：theme 作为 launch 参数
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=True,  # 关闭 Gradio 的冗余启动日志
        theme=gr.themes.Soft()  # ✅ 关键：theme 移到这里
    )


if __name__ == "__main__":
    main()
