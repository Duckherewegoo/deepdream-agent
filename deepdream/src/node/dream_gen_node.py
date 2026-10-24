"""梦境生成节点 - 调用 DeepDream 生成梦境图片"""
import logging
import os
import time

from ..config.settings import DREAM_CN_NAMES, DREAM_CONFIGS
from ..state import AgentState

# 直接复用 tools/dream_gen.py 中带线程锁的单例，禁止本地再造一份。
from ..tools.dream_gen import get_dream_generator
from ..tools.dream_gen import inference_lock as _inference_lock

logger = logging.getLogger(__name__)


# ================= 核心节点逻辑 =================

def dream_gen_node(state: AgentState) -> dict:
    """梦境生成节点 - 调用 DeepDream 生成梦境图片"""
    logger.info("[NODE] 梦境生成节点开始执行")

    dream_type = state.get("dream_type", "normal")
    image_path = state.get("image_path", "")
    # 确保输出目录是绝对路径，避免底层保存时路径错乱
    output_dir = os.path.abspath(state.get("output_dir", "./dream_outputs"))

    # 1. 前置校验
    if not image_path or not os.path.exists(image_path):
        error_msg = "原图不存在，无法生成梦境"
        logger.error(f"[DREAM] {error_msg} | Path: {image_path}")
        return {
            "error": error_msg,
            "status": "生成失败：原图丢失"
        }

    # 2. 类型降级校验（把修改逻辑提前，保证后续不管怎么返回，类型都是对的）
    if dream_type not in DREAM_CONFIGS:
        logger.warning(f"[DREAM] 未知梦境类型 '{dream_type}'，自动降级为 normal")
        dream_type = "normal"

    dream_cn = DREAM_CN_NAMES.get(dream_type, "梦境")

    # 3. 执行生成（加并发锁）
    try:
        generator = get_dream_generator()

        logger.info(f"[DREAM] 等待 GPU 资源... 类型: {dream_type} ({dream_cn})")

        with _inference_lock:
            logger.info(f"[DREAM] 拿到 GPU 锁，开始生成 | 输出目录: {output_dir}")
            start_time = time.time()

            output_path = generator.generate(
                image_path,
                dream_type,
                output_dir=output_dir
            )

            elapsed = time.time() - start_time
            logger.info(f"[DREAM] 生成流程结束 | 耗时: {elapsed:.2f}s")

        # 4. 结果校验
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            logger.info(f"[DREAM] 生成成功: {output_path} ({file_size:.1f} KB)")
            return {
                "output_image_path": output_path,
                "dream_type": dream_type,  # 回传降级后的真实类型
                "status": f"{dream_cn}生成完成！",
                "error": None  # 清空之前的错误
            }
        else:
            error_msg = "算法执行完毕，但输出文件不存在"
            logger.error(f"[DREAM] {error_msg} | 预期路径: {output_path}")
            return {
                "dream_type": dream_type,  # 失败也回传
                "error": error_msg,
                "output_image_path": None,
                "status": "生成失败：无输出文件"
            }

    except Exception as e:
        error_msg = f"梦境生成失败：{e!s}"
        logger.exception("[DREAM] 生成过程发生严重异常")
        return {
            "dream_type": dream_type,  # 异常也回传
            "error": error_msg,
            "output_image_path": None,
            "status": "生成出错了"
        }
