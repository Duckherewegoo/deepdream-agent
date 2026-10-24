"""Agent 逻辑 - 梦境生成助手（重构优化版）"""
import logging
import os
import uuid

from PIL import Image

from ..config.settings import DREAM_KEYWORDS
from ..core.graph import run_graph
from ..node import get_output_image_path, init_dashscope

logger = logging.getLogger(__name__)


class DreamAgent:
    """梦境生成 Agent - 基于 LangGraph 的智能助手"""

    def __init__(self, upload_dir: str = "./uploads", output_dir: str = "./dream_outputs"):
        self.upload_dir = os.path.abspath(upload_dir)
        self.output_dir = os.path.abspath(output_dir)
        try:
            os.makedirs(self.upload_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"[AGENT] 目录创建失败: {e}")
            raise RuntimeError(f"无法创建必要的目录: {e}") from e

        # 让 LLM 工具路径与主路径写同一目录
        from ..tools.dream_gen import set_output_dir as _set_tool_output_dir
        _set_tool_output_dir(self.output_dir)

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
                idx = text_lower.find(keyword.lower())
                if idx != -1:
                    # 往前看 2 个字符，判断有没有否定词
                    prefix = text_lower[max(0, idx - 2):idx]
                    if any(neg in prefix for neg in ["不", "别", "没", "非"]):
                        continue  # 如果是否定词，跳过这个匹配
                    matches.append((keyword, dream_type))

        if not matches:
            return None

        # 长度降序（越精确越好），同长度时按在句子中出现的先后（越靠前越好）
        matches.sort(key=lambda x: (-len(x[0]), text_lower.find(x[0])))
        return matches[0][1]

    def save_uploaded_image(self, image: Image.Image) -> str:
        """保存用户上传的图片"""
        os.makedirs(self.upload_dir, exist_ok=True)
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(self.upload_dir, filename)

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        try:
            image.save(filepath, format="PNG", optimize=True)
            return filepath
        except OSError as e:
            logger.error(f"[AGENT] 图片保存失败: {e}")
            raise RuntimeError(f"无法保存上传的图片: {e}") from e

    def _execute_graph(self, user_text: str, image_path: str | None = None) -> tuple[str, str | None, dict]:
        """内部统一执行方法（DRY 原则）"""
        try:
            # 运行 LangGraph
            final_state = run_graph(user_text, image_path, output_dir=self.output_dir)
        except Exception as e:
            logger.exception("[AGENT] 图执行发生穿透异常")
            return f"系统内部错误: {e}", None, {}

        # 获取输出文本
        final_output = final_state.get("final_output", "抱歉，处理失败。")


        output_image_path = get_output_image_path(final_state)

        return final_output, output_image_path, final_state

    def chat(self, user_text: str, user_image: Image.Image | None = None) -> tuple[str, str | None, dict]:
        """与 Agent 对话"""
        logger.info(f"[AGENT] 收到对话请求 | 有图片: {'是' if user_image else '否'}")

        image_path = None
        if user_image is not None:
            try:
                image_path = self.save_uploaded_image(user_image)
                logger.debug(f"[AGENT] 图片已保存: {image_path}")
            except Exception as e:
                # 已确认策略：图片保存失败即降级为无图模式，不阻断对话。
                logger.error(f"[AGENT] 图片保存失败，降级为无图模式: {e}")
                image_path = None

        return self._execute_graph(user_text, image_path)

    def chat_with_path(self, user_text: str, image_path: str | None = None) -> tuple[str, str | None, dict]:
        """直接使用图片路径对话"""
        logger.info(f"[AGENT] 收到路径对话请求 | 路径: {image_path}")

        # 加固校验：转绝对路径 + 必须存在 + 必须是文件
        if image_path:
            try:
                image_path = os.path.abspath(image_path)
                if not os.path.isfile(image_path):
                    logger.warning(f"[AGENT] 传入的图片路径不存在或不是文件: {image_path}")
                    image_path = None
            except OSError as e:
                logger.error(f"[AGENT] 路径校验发生异常: {e}")
                image_path = None

        return self._execute_graph(user_text, image_path)
