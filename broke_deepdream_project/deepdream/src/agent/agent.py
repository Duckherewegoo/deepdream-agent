"""Agent 逻辑 - 梦境生成助手"""
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
        """初始化 Agent

        Args:
            upload_dir: 上传图片目录
            output_dir: 梦境输出目录
        """
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
        """从用户输入中检测梦境类型

        Args:
            text: 用户输入文本

        Returns:
            梦境类型，如果没有检测到则返回 None
        """
        text_lower = text.lower()

        for dream_type, keywords in DREAM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    return dream_type

        return None

    def save_uploaded_image(self, image: Image.Image) -> str:
        """保存用户上传的图片

        Args:
            image: PIL 图像对象

        Returns:
            保存后的图片路径
        """
        os.makedirs(self.upload_dir, exist_ok=True)
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.upload_dir, filename)
        image.save(filepath)
        return filepath

    def chat(self, user_text: str, user_image: Optional[Image.Image] = None) -> Tuple[str, Optional[str], dict]:
        """与 Agent 对话

        Args:
            user_text: 用户输入文本
            user_image: 用户上传的图片（可选）

        Returns:
            (回复文本, 生成的图片路径, 完整状态) 元组
        """
        print(f"\n[AGENT] 收到对话请求")
        print(f"[AGENT] 用户文本: {user_text[:50]}...")
        print(f"[AGENT] 是否有图片: {'是' if user_image else '否'}")

        # 保存图片（如果有）
        image_path = None
        if user_image is not None:
            image_path = self.save_uploaded_image(user_image)
            print(f"[AGENT] 图片已保存到: {image_path}")

        # 运行 LangGraph
        final_state = run_graph(user_text, image_path, output_dir=self.output_dir)

        # 获取输出
        final_output = final_state.get("final_output", "抱歉，处理失败。")

        # 获取生成的图片路径
        output_image_path = get_output_image_path(final_state)

        print(f"[AGENT] 对话完成")
        print(f"[AGENT] 回复长度: {len(final_output)}")
        print(f"[AGENT] 生成图片: {output_image_path}")

        return final_output, output_image_path, final_state

    def chat_with_path(self, user_text: str, image_path: Optional[str] = None) -> Tuple[str, Optional[str], dict]:
        """直接使用图片路径对话

        Args:
            user_text: 用户输入文本
            image_path: 图片路径（可选）

        Returns:
            (回复文本, 生成的图片路径, 完整状态) 元组
        """
        print(f"\n[AGENT] 收到路径对话请求")

        # 运行 LangGraph
        final_state = run_graph(user_text, image_path, output_dir=self.output_dir)

        # 获取输出
        final_output = final_state.get("final_output", "抱歉，处理失败。")

        # 获取生成的图片路径
        output_image_path = get_output_image_path(final_state)

        return final_output, output_image_path, final_state
