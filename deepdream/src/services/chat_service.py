"""
chat_service.py — 业务逻辑层
负责协调 UI 与底层的 Agent/Generator，处理文件落盘与异常捕获。
"""
import logging
import os
import traceback
from typing import Any

from deepdream.src.agent import DreamAgent
from deepdream.src.deepdream.generator import DeepDreamGenerator
from deepdream.src.utils.meta_utils import load_dream_meta
from PIL import Image

logger = logging.getLogger(__name__)


class ChatService:
    """
    聊天与图像生成的核心服务。
    隔离 UI 层与底层模型逻辑。
    """
    def __init__(self, agent: DreamAgent, upload_dir: str):
        self.agent = agent
        self.upload_dir = upload_dir
        # 延迟加载/实例化生成器
        self.generator = DeepDreamGenerator()

    def process_chat(self, user_text: str, user_image: Image.Image) -> dict[str, Any]:
        """
        处理聊天与图像生成请求。
        """
        try:
            logger.info(f"接收到用户指令: {user_text[:20]}...")
            # 调用 Agent 核心逻辑
            response_text, output_image_path, final_state = self.agent.chat(
                user_text, user_image
            )

            status_text = final_state.get("status", "处理完成")
            output_image = None

            if output_image_path and os.path.exists(output_image_path):
                output_image = Image.open(output_image_path)

            meta = load_dream_meta(output_image_path) if output_image_path else {}

            return {
                "success": True,
                "text": response_text,
                "image": output_image,
                "status": status_text,
                "meta": meta,
                "error_msg": ""
            }

        except Exception as e:
            logger.error(f"Chat 执行出错: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "text": f"❌ **执行出错了！**\n\n**错误信息：** {e!s}\n\n**详情：**\n```\n{traceback.format_exc()}\n```",
                "image": None,
                "status": "出错了",
                "meta": {},
                "error_msg": str(e)
            }

    def reproduce_dream(self, original_img: Image.Image, new_upload_img: Image.Image = None) -> dict[str, Any]:
        """
        根据原图参数复现梦境。
        """
        if not original_img:
            return {"success": False, "image": None, "meta": {}, "error_msg": "没有可复用的原图参数"}

        try:
            # 1. 加载原图参数 (假设 load_dream_meta 支持 PIL 对象或路径)
            # 这里需要看你的 load_dream_meta 实现，如果是按路径存的，可能 original_img 实际上是个路径
            meta = load_dream_meta(original_img)
            if not meta:
                return {"success": False, "image": None, "meta": {}, "error_msg": "未找到原图参数，无法复现"}

            # 2. 确定输入路径
            input_path = original_img
            if new_upload_img:
                # 如果是新上传的图片，保存为临时文件
                temp_path = os.path.join(self.upload_dir, "temp_reproduce.png")
                new_upload_img.save(temp_path)
                input_path = temp_path

            # 3. 调用底层生成器
            output_path = self.generator.generate(
                input_path=input_path,
                dream_type=meta["dream_type"],
                custom_cfg=meta["config"],
                custom_layers=meta["layers_used"]
            )

            # 4. 组装返回数据
            new_img = Image.open(output_path)
            new_meta = load_dream_meta(output_path)

            return {
                "success": True,
                "image": new_img,
                "meta": new_meta,
                "error_msg": ""
            }

        except Exception as e:
            logger.error(f"复现执行出错: {e}\n{traceback.format_exc()}")
            return {
                "success": False,
                "image": None,
                "meta": {},
                "error_msg": str(e)
            }
