"""梦境生成工具（优化版）"""
import os
import json
from langchain_core.tools import tool

from ..deepdream.generator import DeepDreamGenerator
from ..config.settings import DREAM_CONFIGS, DREAM_CN_NAMES

# 全局 DeepDream 生成器实例（优化懒加载）
_dream_generator = None
_output_dir = "./dream_outputs"


def set_output_dir(output_dir: str):
    """设置梦境图片输出目录（绝对路径）"""
    global _output_dir
    _output_dir = os.path.abspath(output_dir)
    os.makedirs(_output_dir, exist_ok=True)


def get_dream_generator(clear_cache: bool = False) -> DeepDreamGenerator:
    """获取或创建 DeepDream 生成器单例（优化连续使用）"""
    global _dream_generator
    if _dream_generator is None:
        _dream_generator = DeepDreamGenerator()
    elif clear_cache:
        _dream_generator.clear_cache()
    return _dream_generator


@tool
def generate_dream(dream_type: str, user_image_path: str) -> str:
    """生成 DeepDream 梦境图片（优化错误处理）"""
    # 验证图片路径
    if not user_image_path or not os.path.exists(user_image_path):
        return json.dumps({
            "success": False,
            "msg": "图片不存在，请先上传图片",
            "dream_type": dream_type
        }, ensure_ascii=False)

    # 验证梦境类型
    if dream_type not in DREAM_CONFIGS:
        dream_type = "normal"

    try:
        # 获取生成器（每次生成前清理缓存）
        generator = get_dream_generator(clear_cache=True)
        output_path = generator.generate(
            user_image_path, dream_type, output_dir=_output_dir)

        dream_cn = DREAM_CN_NAMES.get(dream_type, dream_type)

        return json.dumps({
            "success": True,
            "type": dream_type,
            "type_cn": dream_cn,
            "path": output_path,
            "msg": f"✨ {dream_cn}生成成功！图片已保存至：{output_path}"
        }, ensure_ascii=False)

    except Exception as e:
        # 清理缓存后重试一次
        try:
            generator = get_dream_generator(clear_cache=True)
            output_path = generator.generate(
                user_image_path, dream_type, output_dir=_output_dir)
            dream_cn = DREAM_CN_NAMES.get(dream_type, dream_type)
            return json.dumps({
                "success": True,
                "type": dream_type,
                "type_cn": dream_cn,
                "path": output_path,
                "msg": f"✨ {dream_cn}生成成功！图片已保存至：{output_path}"
            }, ensure_ascii=False)
        except Exception as e2:
            return json.dumps({
                "success": False,
                "msg": f"梦境生成失败：{str(e2)}",
                "dream_type": dream_type
            }, ensure_ascii=False)


@tool
def get_dream_knowledge(topic: str = "all") -> str:
    """获取梦境相关知识（补充算法细节）"""
    knowledge = {
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
            "title": "5种梦境类型介绍",
            "content": """**🌙 正常梦 (Normal)**
基础版 DeepDream 效果，平衡真实感与梦幻感。
- 中层特征（Mixed_5b/5c），平衡纹理与结构
- 3层octave，80次迭代，适度现实融合
- 适合初次体验，温和不夸张

**💖 美梦 (Sweet)**
柔和梦幻风格，像童话世界一样美好。
- 低层特征（Mixed_4a/4b/4c），以纹理/边缘为主
- 4层octave，60次迭代，高现实融合（25%）
- 轻微高斯模糊+亮度增强，温暖治愈

**👻 噩梦 (Nightmare)**
恐怖锐利风格，适合追求刺激的你。
- 高层特征（Mixed_6b/6c/7a），以物体/结构为主
- 3层octave，100次迭代，低现实融合（5%）
- 轻微锐化+对比度增强，扭曲恐怖

**🌫️ 迷雾梦 (Mist)**
朦胧模糊风格，像隔着雾气看世界。
- 多层特征混合（Mixed_5a-6a），层次感强
- 5层octave，70次迭代，梯度噪声增加朦胧感
- 轻微高斯模糊，神秘氛围

**🤪 疯狂梦 (Crazy)**
极致迷幻风格，彻底放飞想象力。
- 全层特征（Mixed_4a-8a），多维度扭曲
- 4层octave，200次迭代，0%现实融合
- 动态步长+色彩增强，视觉冲击强烈"""
        }
    }

    if topic == "all":
        result = {
            "success": True,
            "topics": list(knowledge.keys()),
            "content": "\n\n".join([
                f"## {v['title']}\n{v['content']}" for v in knowledge.values()
            ])
        }
    elif topic in knowledge:
        result = {
            "success": True,
            "topic": topic,
            "title": knowledge[topic]["title"],
            "content": knowledge[topic]["content"]
        }
    else:
        result = {
            "success": False,
            "msg": f"未知主题：{topic}，可选主题：{', '.join(knowledge.keys())}"
        }

    return json.dumps(result, ensure_ascii=False)
