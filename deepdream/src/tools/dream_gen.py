"""梦境生成工具（重构优化版）"""
import json
import logging
import os

from langchain_core.tools import tool

from ..config.settings import DREAM_CN_NAMES, DREAM_CONFIGS
from ..deepdream.generator import DeepDreamGenerator

logger = logging.getLogger(__name__)

# ================= 全局状态 =================
# ⚠️ 注意：这依然是全局状态，仅适用于单进程单用户场景。
_dream_generator = None
_output_dir = "./dream_outputs"


def set_output_dir(output_dir: str):
    """设置梦境图片输出目录（绝对路径）"""
    global _output_dir
    _output_dir = os.path.abspath(output_dir)
    os.makedirs(_output_dir, exist_ok=True)
    logger.debug(f"[TOOL] 梦境输出目录已设置为: {_output_dir}")


def get_dream_generator(clear_cache: bool = False) -> DeepDreamGenerator:
    """获取或创建 DeepDream 生成器单例"""
    global _dream_generator
    if _dream_generator is None:
        logger.info("[TOOL] 初始化 DeepDreamGenerator 实例")
        _dream_generator = DeepDreamGenerator()
    elif clear_cache:
        logger.debug("[TOOL] 清理 DeepDreamGenerator 缓存")
        _dream_generator.clear_cache()
    return _dream_generator


def _build_json_response(data: dict) -> str:
    """统一的 JSON 响应构建器"""
    return json.dumps(data, ensure_ascii=False)


# ================= 工具：生成梦境 =================

@tool
def generate_dream(dream_type: str, user_image_path: str) -> str:
    """生成 DeepDream 梦境图片（优化错误处理与重试机制）"""

    # 1. 前置验证
    if not user_image_path or not os.path.exists(user_image_path):
        return _build_json_response({
            "success": False,
            "msg": "图片不存在，请先上传图片",
            "dream_type": dream_type
        })

    if dream_type not in DREAM_CONFIGS:
        logger.warning(f"[TOOL] 未知的梦境类型 '{dream_type}'，自动降级为 normal")
        dream_type = "normal"

    dream_cn = DREAM_CN_NAMES.get(dream_type, dream_type)
    max_retries = 2

    # 2. 带重试的执行逻辑（仅在第一次失败后清理缓存重试）
    for attempt in range(max_retries):
        try:
            # 如果是重试（attempt > 0），则清空缓存
            generator = get_dream_generator(clear_cache=(attempt > 0))

            output_path = generator.generate(
                user_image_path, dream_type, output_dir=_output_dir
            )

            logger.info(f"[TOOL] {dream_cn} 生成成功: {output_path}")
            return _build_json_response({
                "success": True,
                "type": dream_type,
                "type_cn": dream_cn,
                "path": output_path,
                "msg": f"✨ {dream_cn}生成成功！图片已保存至：{output_path}"
            })

        except Exception as e:
            # 修复：改用 logger.exception，去掉无用的 last_error 赋值
            logger.exception(
                f"[TOOL] 梦境生成失败 (尝试 {attempt + 1}/{max_retries}): {e}"
            )

    # 3. 所有重试均失败
    return _build_json_response({
        "success": False,
        "msg": "梦境生成失败，请稍后重试。",
        "dream_type": dream_type
    })


# ================= 工具：梦境知识库 =================

# 抽离死数据，保持函数轻量
DREAM_KNOWLEDGE_BASE = {
    "deepdream": {
        "title": "DeepDream 算法原理",
        "content": """DeepDream 是 Google 于 2015 年开发的一种图像生成算法，基于卷积神经网络（CNN）。

**核心原理：**
1. 反向利用预训练的图像分类网络（如 InceptionV3）
2. 不是调整网络权重，而是调整输入图像本身
3. 通过梯度上升最大化特定层的激活值
4. 让网络"增强"它在图像中"看到"的特征

**本版本优化点：**
- 美梦：使用低层特征（边缘/纹理）+ 小步长 + 高现实融合，体现温暖治愈
- 噩梦：使用高层特征（物体/结构）+ 大步长 + 低现实融合，体现扭曲恐怖
- 疯狂梦：全层特征 + 动态步长 + 无现实融合，体现极致迷幻
- 迷雾梦：多尺度 + 梯度噪声，体现朦胧神秘
- 正常梦：平衡参数，体现真实与梦幻的结合"""
    },
    "dreams": {
        "title": "梦境心理学小知识",
        "content": """**关于梦境的有趣事实：**
1. **每个人都会做梦**：即使你不记得，每晚也会做3-5个梦
2. **REM睡眠**：大多数生动的梦发生在快速眼动睡眠阶段
3. **梦的时长**：每个梦大约持续5-20分钟
4. **梦境遗忘**：醒来后5分钟内会忘记50%的梦，10分钟后忘记90%
5. **盲人与梦**：先天失明的人梦中没有视觉，但其他感官更强烈

**常见梦境类型：**
- 美梦：通常与愉悦、成功、被爱相关
- 噩梦：常与焦虑、恐惧、压力有关
- 清醒梦：知道自己在做梦，甚至能控制梦境内容
- 重复梦：反复出现相同场景，可能与未解决的心理问题有关
- 预言梦：感觉梦到了未来发生的事，大多是巧合或记忆偏差"""
    },
    "types": {
        "title": "梦境类型介绍",
        "content": """**🌙 正常梦 (Normal)**
基础版 DeepDream 效果，平衡真实感与梦幻感。
- 中层特征（Mixed_5b/5c），平衡纹理与结构
- 3层octave，80次迭代，适度现实融合

**💖 美梦 (Sweet)**
柔和梦幻风格，像童话世界一样美好。
- 低层特征（Mixed_4a/4b/4c），以纹理/边缘为主
- 4层octave，60次迭代，高现实融合，温暖治愈

**👻 噩梦 (Nightmare)**
恐怖锐利风格，适合追求刺激的你。
- 高层特征（Mixed_6b/6c/7a），以物体/结构为主
- 3层octave，100次迭代，低现实融合，扭曲恐怖

**🌫️ 迷雾梦 (Mist)**
朦胧模糊风格，像隔着雾气看世界。
- 多层特征混合（Mixed_5a-6a），层次感强
- 5层octave，70次迭代，梯度噪声增加朦胧感

**🤪 疯狂梦 (Crazy)**
极致迷幻风格，彻底放飞想象力。
- 全层特征（Mixed_4a-8a），多维度扭曲
- 4层octave，200次迭代，0%现实融合

**✨ 星空梦 (Starlight)**
深邃、闪烁、星光流动，带有宇宙的孤寂与壮丽感。
- 中层 + 高层混合，产生结构感 + 纹理闪烁
- 增强极值区域亮度，模拟星光

**🌊 海洋梦 (Ocean)**
流动、深邃、波浪起伏，带有蓝色调的沉浸感。
- 低层 + 中层，强调纹理流动
- 正弦波流场扭曲，模拟海浪

**🌸 花海梦 (Floral)**
绚烂、繁花、色彩斑斓，带有春天的生机感。
- 中层，强调繁复的纹理与色彩
- 高饱和度增强"""
    }
}


@tool
def get_dream_knowledge(topic: str = "all") -> str:
    """获取梦境相关知识（补充算法细节）"""
    if topic == "all":
        content = "\n\n".join([
            f"## {v['title']}\n{v['content']}"
            for v in DREAM_KNOWLEDGE_BASE.values()
        ])
        return _build_json_response({
            "success": True,
            "topics": list(DREAM_KNOWLEDGE_BASE.keys()),
            "content": content
        })

    elif topic in DREAM_KNOWLEDGE_BASE:
        data = DREAM_KNOWLEDGE_BASE[topic]
        return _build_json_response({
            "success": True,
            "topic": topic,
            "title": data["title"],
            "content": data["content"]
        })

    else:
        valid_topics = ', '.join(DREAM_KNOWLEDGE_BASE.keys())
        logger.warning(f"[TOOL] 请求了未知的知识主题: {topic}")
        return _build_json_response({
            "success": False,
            "msg": f"未知主题：{topic}，可选主题：{valid_topics}"
        })
