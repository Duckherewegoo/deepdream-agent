"""Agent 逻辑 - 梦境生成助手（重构优化版）"""
import logging
import os
import uuid

from PIL import Image

from ..config.settings import DREAM_KEYWORDS
from ..core.graph import run_graph
from ..node import get_output_image_path, init_dashscope
from ..tools import set_output_dir as set_dream_output_dir

logger = logging.getLogger(__name__)


class DreamAgent:
    """梦境生成 Agent - 基于 LangGraph 的智能助手"""

    def __init__(self, upload_dir: str = "./uploads", output_dir: str = "./dream_outputs"):
        """初始化 Agent"""
        self.upload_dir = os.path.abspath(upload_dir)
        self.output_dir = os.path.abspath(output_dir)

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 设置梦境生成器的输出目录（底层全局状态）
        set_dream_output_dir(self.output_dir)

        # 初始化第三方服务
        init_dashscope()
        logger.info(f"[AGENT] 初始化完成 | Upload: {self.upload_dir} | Output: {self.output_dir}")

    def detect_dream_type(self, text: str) -> str | None:
        """
        从用户输入中检测梦境类型。
        优化：如果包含多个关键字，优先匹配最长的关键字（通常更精确）。
        """
        if not text or not text.strip():
            return None

        text_lower = text.lower()
        matches = []

        for dream_type, keywords in DREAM_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches.append((keyword, dream_type))

        if not matches:
            return None

        # 按匹配到的关键词长度降序排序，取最长的那个作为最终结果
        matches.sort(key=lambda x: len(x[0]), reverse=True)
        return matches[0][1]

    def save_uploaded_image(self, image: Image.Image) -> str:
        """保存用户上传的图片（修复 PNG quality 参数报错问题）"""
        os.makedirs(self.upload_dir, exist_ok=True)
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.upload_dir, filename)

        # 统一转换为 RGB 模式（防止 RGBA/P 模式在某些格式下保存失败）
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # 修复：PNG 是无损格式，不能传 quality，传 optimize=True
        image.save(filepath, format="PNG", optimize=True)
        return filepath

    def _execute_graph(self, user_text: str, image_path: str | None = None) -> tuple[str, str | None, dict]:
        """内部统一执行方法（DRY 原则）"""
        # 运行 LangGraph
        final_state = run_graph(user_text, image_path, output_dir=self.output_dir)

        # 获取输出文本
        final_output = final_state.get("final_output", "抱歉，处理失败。")

        # 获取生成的图片路径并验证
        output_image_path = get_output_image_path(final_state)
        if output_image_path and not os.path.exists(output_image_path):
            logger.warning(f"[AGENT] 生成的图片路径不存在: {output_image_path}")
            output_image_path = None

        return final_output, output_image_path, final_state

    def chat(self, user_text: str, user_image: Image.Image | None = None) -> tuple[str, str | None, dict]:
        """与 Agent 对话（支持传入 PIL 图片）"""
        logger.info(f"[AGENT] 收到对话请求 | 有图片: {'是' if user_image else '否'}")

        image_path = None
        if user_image is not None:
            image_path = self.save_uploaded_image(user_image)
            logger.debug(f"[AGENT] 图片已保存: {image_path}")

        return self._execute_graph(user_text, image_path)

    def chat_with_path(self, user_text: str, image_path: str | None = None) -> tuple[str, str | None, dict]:
        """直接使用图片路径对话"""
        logger.info(f"[AGENT] 收到路径对话请求 | 路径: {image_path}")

        # 验证图片路径
        if image_path and not os.path.exists(image_path):
            logger.warning(f"[AGENT] 传入的图片路径不存在: {image_path}")
            image_path = None

        return self._execute_graph(user_text, image_path)
