"""梦境生成节点 - 调用 DeepDream 生成梦境图片"""
import os
import time
import traceback
from langchain_core.runnables import RunnableLambda

from .llm_node import AgentState
from ..deepdream.generator import DeepDreamGenerator
from ..config.settings import DREAM_CONFIGS, DREAM_CN_NAMES


# 全局生成器实例（懒加载）
_dream_generator = None


def get_dream_generator() -> DeepDreamGenerator:
    """获取或创建 DeepDream 生成器单例"""
    global _dream_generator
    if _dream_generator is None:
        print(f"[DREAM] 初始化 DeepDream 生成器...")
        _dream_generator = DeepDreamGenerator()
        print(f"[DREAM] DeepDream 生成器初始化完成，设备: {_dream_generator.device}")
    return _dream_generator


def set_dream_generator_output_dir(output_dir: str):
    """设置梦境生成器的输出目录"""
    # 这个函数暂时保留，生成器的 generate 方法会接收 output_dir 参数
    pass


def dream_gen_node(state: AgentState) -> AgentState:
    """梦境生成节点 - 调用 DeepDream 生成梦境图片

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态
    """
    print(f"\n[NODE] 梦境生成节点开始执行")

    dream_type = state.get("dream_type", "normal")
    image_path = state.get("image_path", "")
    output_dir = state.get("output_dir", "./dream_outputs")

    print(f"[DREAM] 梦境类型: {dream_type} ({DREAM_CN_NAMES.get(dream_type, dream_type)})")
    print(f"[DREAM] 输入图片: {image_path}")
    print(f"[DREAM] 输出目录: {output_dir}")

    dream_cn = DREAM_CN_NAMES.get(dream_type, "梦境")
    state["status"] = f"正在生成{dream_cn}图片，请稍候..."

    # 验证输入
    if not image_path or not os.path.exists(image_path):
        error_msg = "图片不存在，无法生成梦境"
        print(f"[DREAM] 错误: {error_msg}")
        state["error"] = error_msg
        state["status"] = "生成失败：图片不存在"
        return state

    if dream_type not in DREAM_CONFIGS:
        dream_type = "normal"

    try:
        # 获取生成器
        generator = get_dream_generator()

        # 记录开始时间
        start_time = time.time()
        print(f"[DREAM] 开始生成梦境...")

        # 生成梦境
        output_path = generator.generate(image_path, dream_type, output_dir=output_dir)

        # 计算耗时
        elapsed = time.time() - start_time
        print(f"[DREAM] 梦境生成完成，耗时: {elapsed:.2f} 秒")
        print(f"[DREAM] 输出图片: {output_path}")

        # 验证输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            print(f"[DREAM] 输出文件大小: {file_size:.1f} KB")
            state["output_image_path"] = output_path
            state["status"] = f"{dream_cn}生成完成！"
        else:
            error_msg = "生成失败，输出文件不存在"
            print(f"[DREAM] 错误: {error_msg}")
            state["error"] = error_msg
            state["status"] = "生成失败"

    except Exception as e:
        error_msg = f"梦境生成失败：{str(e)}"
        print(f"[DREAM] 异常: {error_msg}")
        traceback.print_exc()
        state["error"] = error_msg
        state["status"] = "生成出错了"

    print(f"[NODE] 梦境生成节点执行完成")
    return state


# 可运行对象包装
dream_gen_runnable = RunnableLambda(dream_gen_node)
