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


def get_dream_generator() -> DeepDreamGenerator:
    """获取或创建 DeepDream 生成器单例（双重检查锁，保证并发安全）"""
    global _dream_generator
    if _dream_generator is not None:
        return _dream_generator

    with _generator_lock:
        if _dream_generator is None:
            logger.info("[DREAM] 正在初始化 DeepDream 生成器（此过程可能需要加载模型，请稍候）...")
            _dream_generator = DeepDreamGenerator()
            logger.info(f"[DREAM] 生成器初始化完成，运行设备: {_dream_generator.device}")

    return _dream_generator


# ================= 核心节点逻辑 =================

def dream_gen_node(state: AgentState) -> AgentState:
    """梦境生成节点 - 调用 DeepDream 生成梦境图片"""
    logger.info("[NODE] 梦境生成节点开始执行")

    dream_type = state.get("dream_type", "normal")
    image_path = state.get("image_path", "")
    output_dir = state.get("output_dir", "./dream_outputs")

    dream_cn = DREAM_CN_NAMES.get(dream_type, "梦境")
    state["status"] = f"正在生成{dream_cn}图片，请稍候..."

    # 1. 前置校验
    if not image_path or not os.path.exists(image_path):
        error_msg = "原图不存在，无法生成梦境"
        logger.error(f"[DREAM] {error_msg} | Path: {image_path}")
        state["error"] = error_msg
        state["status"] = "生成失败：原图丢失"
        return state

    if dream_type not in DREAM_CONFIGS:
        logger.warning(f"[DREAM] 遇到未知梦境类型 '{dream_type}'，自动降级为 normal")
        dream_type = "normal"

    # 2. 执行生成
    try:
        generator = get_dream_generator()
        start_time = time.time()

        logger.info(f"[DREAM] 开始生成 | 类型: {dream_type} ({dream_cn}) | 输出目录: {output_dir}")

        output_path = generator.generate(
            image_path,
            dream_type,
            output_dir=output_dir
        )

        elapsed = time.time() - start_time
        logger.info(f"[DREAM] 生成流程结束 | 耗时: {elapsed:.2f}s")

        # 3. 结果校验与写回
        if output_path and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            logger.info(f"[DREAM] 生成成功: {output_path} ({file_size:.1f} KB)")
            state["output_image_path"] = output_path
            state["status"] = f"{dream_cn}生成完成！"
        else:
            error_msg = "算法执行完毕，但输出文件不存在"
            logger.error(f"[DREAM] ❌ {error_msg} | 预期路径: {output_path}")
            state["error"] = error_msg
            state["output_image_path"] = None
            state["status"] = "生成失败：无输出文件"

    except Exception as e:
        error_msg = f"梦境生成失败：{e!s}"
        # 修复：使用 logger.exception 自动捕获并打印完整堆栈
        logger.exception("[DREAM] ❌ 生成过程发生严重异常")
        state["error"] = error_msg
        state["output_image_path"] = None
        state["status"] = "生成出错了"

    logger.info("[NODE] 梦境生成节点执行完成")
    return state


# 可运行对象包装
dream_gen_runnable = RunnableLambda(dream_gen_node)
