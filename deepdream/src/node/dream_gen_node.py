"""梦境生成节点 - 调用 DeepDream 生成梦境图片"""
import logging
import os
import threading
import time

from langchain_core.runnables import RunnableLambda

from ..config.settings import DREAM_CN_NAMES, DREAM_CONFIGS
from ..deepdream.generator import DeepDreamGenerator
from ..state import AgentState

logger = logging.getLogger(__name__)

# ================= 生成器单例（线程安全） =================
_dream_generator = None
_generator_lock = threading.Lock()

# 新增：推理锁。防止多用户并发调用，撑爆 GPU 显存
_inference_lock = threading.Lock()


def get_dream_generator() -> DeepDreamGenerator:
    """获取或创建 DeepDream 生成器单例（双重检查锁）"""
    global _dream_generator
    if _dream_generator is not None:
        return _dream_generator

    with _generator_lock:
        if _dream_generator is None:
            logger.info("[DREAM] 正在初始化生成器（加载模型可能较慢，请稍候）...")
            _dream_generator = DeepDreamGenerator()
            logger.info(f"[DREAM] 生成器初始化完成，设备: {_dream_generator.device}")

    return _dream_generator


# ================= 核心节点逻辑 =================

def dream_gen_node(state: AgentState) -> dict:
    """梦境生成节点 - 调用 DeepDream 生成梦境图片"""
    logger.info("[NODE] 梦境生成节点开始执行")

    dream_type = state.get("dream_type", "normal")
    image_path = state.get("image_path", "")
    # 确保输出目录是绝对路径，避免底层保存时路径错乱
    output_dir = os.path.abspath(state.get("output_dir", "./dream_outputs"))

    dream_cn = DREAM_CN_NAMES.get(dream_type, "梦境")

    # 1. 前置校验
    if not image_path or not os.path.exists(image_path):
        error_msg = "原图不存在，无法生成梦境"
        logger.error(f"[DREAM] {error_msg} | Path: {image_path}")
        # LangGraph 规范：只返回更新的字段
        return {
            "error": error_msg,
            "status": "生成失败：原图丢失"
        }

    if dream_type not in DREAM_CONFIGS:
        logger.warning(f"[DREAM] 未知梦境类型 '{dream_type}'，自动降级为 normal")
        dream_type = "normal"

    # 2. 执行生成（加并发锁）
    try:
        generator = get_dream_generator()
        
        logger.info(f"[DREAM] 等待 GPU 资源... 类型: {dream_type} ({dream_cn})")
        
        # 关键修复：加锁，保证同一时间只有一个线程在使用 GPU
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

        # 3. 结果校验
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            logger.info(f"[DREAM] 生成成功: {output_path} ({file_size:.1f} KB)")
            return {
                "output_image_path": output_path,
                "status": f"{dream_cn}生成完成！",
                "error": None  # 清空之前的错误
            }
        else:
            error_msg = "算法执行完毕，但输出文件不存在"
            logger.error(f"[DREAM] {error_msg} | 预期路径: {output_path}")
            return {
                "error": error_msg,
                "output_image_path": None,
                "status": "生成失败：无输出文件"
            }

    except Exception as e:
        error_msg = f"梦境生成失败：{e!s}"
        logger.exception("[DREAM] 生成过程发生严重异常")
        return {
            "error": error_msg,
            "output_image_path": None,
            "status": "生成出错了"
        }


# 可运行对象包装
dream_gen_runnable = RunnableLambda(dream_gen_node)
