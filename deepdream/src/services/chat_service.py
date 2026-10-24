"""
chat_service.py — 业务逻辑层
负责协调 UI 与底层的 Agent/Generator，处理文件落盘与异常捕获。
"""
import logging
import os
import traceback
import uuid
from typing import Any

from deepdream.src.agent import DreamAgent
from deepdream.src.tools.dream_gen import get_dream_generator, inference_lock
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
        # 删除了冗余的 self.generator = DeepDreamGenerator()
        # 记录上一次生成的图片路径，供复现使用
        self.last_output_path: str | None = None

    def process_chat(self, user_text: str, user_image: Image.Image) -> dict[str, Any]:
        """处理聊天与图像生成请求。"""
        try:
            logger.info(f"接收到用户指令: {user_text[:20]}...")
            response_text, output_image_path, final_state = self.agent.chat(
                user_text, user_image
            )

            status_text = final_state.get("status", "处理完成")
            output_image = None

            if output_image_path and os.path.exists(output_image_path):
                img_temp = Image.open(output_image_path)
                img_temp.load()
                output_image = img_temp.copy()
                img_temp.close()
                # 记录有效路径，供后续复现
                self.last_output_path = output_image_path

            meta = load_dream_meta(self.last_output_path) if self.last_output_path else {}

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
                "text": f"❌ **执行出错了！**\n\n**错误信息：** {e!s}",
                "image": None,
                "status": "出错了",
                "meta": {},
                "error_msg": str(e)
            }

    def reproduce_dream(self, original_img: Image.Image, new_upload_img: Image.Image = None) -> dict[str, Any]:
        """根据原图参数复现梦境。"""
        if not original_img:
            return {"success": False, "image": None, "meta": {}, "error_msg": "没有可复用的原图"}

        try:
            # 修复：使用缓存的上次输出路径来读取 meta
            if not self.last_output_path:
                return {"success": False, "image": None, "meta": {}, "error_msg": "找不到上一次的生成参数"}

            meta = load_dream_meta(self.last_output_path)
            if not meta:
                return {"success": False, "image": None, "meta": {}, "error_msg": "未找到原图参数，无法复现"}

            # 如果传了新图，保存新图；否则复用原输入图
            if new_upload_img:
                temp_path = os.path.join(self.upload_dir, f"temp_repro_{uuid.uuid4().hex[:6]}.png") # 随机防覆盖
                new_upload_img.save(temp_path)
                input_path = temp_path
            else:
                input_path = getattr(self, "last_input_path", None) or self.last_output_path # 优先用上次的输入图

            # 调用单例生成器，不再耗费额外显存
            generator = get_dream_generator()
            with inference_lock:
                output_path = generator.generate(
                    input_path=input_path,
                    dream_type=meta.get("dream_type", "normal"),
                    custom_cfg=meta.get("config", {}),
                    custom_layers=meta.get("layers_used", [])
                )

            img_temp = Image.open(output_path)
            img_temp.load()
            new_img = img_temp.copy()
            img_temp.close()
            new_meta = load_dream_meta(output_path)

            # 更新缓存路径
            self.last_output_path = output_path

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
