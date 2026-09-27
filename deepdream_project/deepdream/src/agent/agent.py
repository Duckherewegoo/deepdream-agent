"""Agent 逻辑 - 梦境生成助手（优化版）"""
import os
import uuid
from typing import Optional, Tuple
from PIL import Image

from ..core.graph import run_graph
from ..node import init_dashscope, get_output_image_path
from ..config.settings import DREAM_KEYWORDS
from ..tools import set_output_dir as set_dream_output_dir


class DreamAgent:
    """梦境生成 Agent - 基于 LangGraph 的智能助手"""

    def __init__(self, upload_dir: str = "./uploads", output_dir: str = "./dream_outputs"):
        """初始化 Agent"""
        # 转换为绝对路径
        self.upload_dir = os.path.abspath(upload_dir)
        self.output_dir = os.path.abspath(output_dir)

        # 创建必要目录
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置梦境生成器的输出目录
        set_dream_output_dir(self.output_dir)

        # 初始化 dashscope
        init_dashscope()

    def detect_dream_type(self, text: str) -> Optional[str]:
        """从用户输入中检测梦境类型（优化关键词匹配）"""
        if not text:
            return None

        text_lower = text.lower().strip()

        # 优先精确匹配
        for dream_type, keywords in DREAM_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return dream_type

        return None

    def save_uploaded_image(self, image: Image.Image) -> str:
        """保存用户上传的图片（优化命名）"""
        os.makedirs(self.upload_dir, exist_ok=True)
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.upload_dir, filename)
        # 优化保存质量
        image.save(filepath, quality=95)
        return filepath

    def chat(self, user_text: str, user_image: Optional[Image.Image] = None) -> Tuple[str, Optional[str], dict]:
        """与 Agent 对话（优化连续使用）"""
        print(f"\n[AGENT] 收到对话请求")
        print(f"[AGENT] 用户文本: {user_text[:50]}...")
        print(f"[AGENT] 是否有图片: {'是' if user_image else '否'}")

        # 保存图片（如果有）
        image_path = None
        if user_image is not None:
            image_path = self.save_uploaded_image(user_image)
            print(f"[AGENT] 图片已保存到: {image_path}")

        # 运行 LangGraph
        final_state = run_graph(user_text, image_path,
                                output_dir=self.output_dir)

        # 获取输出
        final_output = final_state.get("final_output", "抱歉，处理失败。")

        # 获取生成的图片路径
        output_image_path = get_output_image_path(final_state)

        # 验证图片路径是否存在
        if output_image_path and not os.path.exists(output_image_path):
            output_image_path = None

        print(f"[AGENT] 对话完成")
        print(f"[AGENT] 回复长度: {len(final_output)}")
        print(f"[AGENT] 生成图片: {output_image_path}")

        return final_output, output_image_path, final_state

    def chat_with_path(self, user_text: str, image_path: Optional[str] = None) -> Tuple[str, Optional[str], dict]:
        """直接使用图片路径对话"""
        print(f"\n[AGENT] 收到路径对话请求")

        # 验证图片路径
        if image_path and not os.path.exists(image_path):
            print(f"[AGENT] 图片路径不存在: {image_path}")
            image_path = None

        # 运行 LangGraph
        final_state = run_graph(user_text, image_path,
                                output_dir=self.output_dir)

        # 获取输出
        final_output = final_state.get("final_output", "抱歉，处理失败。")

        # 获取生成的图片路径
        output_image_path = get_output_image_path(final_state)

        # 验证图片路径是否存在
        if output_image_path and not os.path.exists(output_image_path):
            output_image_path = None

        return final_output, output_image_path, final_state
