"""梦境生成工具"""
import os
import json
from langchain_core.tools import tool

from ..deepdream.generator import DeepDreamGenerator
from ..config.settings import DREAM_CONFIGS, DREAM_CN_NAMES

# 全局 DeepDream 生成器实例（懒加载）
_dream_generator = None


def get_dream_generator() -> DeepDreamGenerator:
    """获取或创建 DeepDream 生成器单例"""
    global _dream_generator
    if _dream_generator is None:
        _dream_generator = DeepDreamGenerator()
    return _dream_generator


@tool
def generate_dream(dream_type: str, user_image_path: str) -> str:
    """生成 DeepDream 梦境图片。

    支持的梦境类型：
    - normal: 正常梦（基础效果）
    - sweet: 美梦（柔和梦幻）
    - nightmare: 噩梦（恐怖锐利）
    - mist: 迷雾梦（朦胧模糊）
    - crazy: 疯狂梦（极致迷幻）

    Args:
        dream_type: 梦境类型，可选值：normal, sweet, nightmare, mist, crazy
        user_image_path: 用户上传的图片路径

    Returns:
        生成结果的 JSON 字符串，包含是否成功、梦境类型、输出图片路径等信息
    """
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
        # 获取生成器并生成梦境
        generator = get_dream_generator()
        output_path = generator.generate(user_image_path, dream_type)

        dream_cn = DREAM_CN_NAMES.get(dream_type, dream_type)

        return json.dumps({
            "success": True,
            "type": dream_type,
            "type_cn": dream_cn,
            "path": output_path,
            "msg": f"✨ {dream_cn}生成成功！图片已保存至：{output_path}"
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "success": False,
            "msg": f"梦境生成失败：{str(e)}",
            "dream_type": dream_type
        }, ensure_ascii=False)


@tool
def get_dream_knowledge(topic: str = "all") -> str:
    """获取梦境相关知识或 DeepDream 算法科普。

    支持的主题：
    - all: 全部知识
    - deepdream: DeepDream 算法原理
    - dreams: 梦境心理学知识
    - types: 5种梦境类型介绍

    Args:
        topic: 知识主题，可选值：all, deepdream, dreams, types

    Returns:
        科普知识的 JSON 字符串
    """
    knowledge = {
        "deepdream": {
            "title": "DeepDream 算法原理",
            "content": """DeepDream 是 Google 于 2015 年开发的一种图像生成算法，基于卷积神经网络（CNN）。

**核心原理：**
1. 反向利用预训练的图像分类网络（如 InceptionV3）
2. 不是调整网络权重，而是调整输入图像本身
3. 通过梯度上升最大化特定层的激活值
4. 让网络"增强"它在图像中"看到"的特征

**技术细节：**
- 使用多尺度（Octave）处理，从粗到细逐步增强细节
- Reality Fusion：与原图融合，保留一定真实感
- Safe Shift：随机滚动图像避免边界伪影
- 不同层对应不同特征：低层是边缘纹理，高层是物体结构

**5种梦境的区别：**
- 正常梦：基础效果，平衡真实与梦幻
- 美梦：低层特征，柔和梦幻，带高斯模糊
- 噩梦：中高层特征，锐利恐怖，带锐化滤镜
- 迷雾梦：多层特征，朦胧模糊，大量随机偏移
- 疯狂梦：全层特征，极致迷幻，无现实约束"""
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
- 2层特征提取，40次迭代
- 15% 现实融合度，保留原图主体
- 适合初次体验，温和不夸张

**💖 美梦 (Sweet)**
柔和梦幻风格，像童话世界一样美好。
- 低层特征（边缘、纹理）
- 高斯模糊滤镜，增加柔美感
- 20% 现实融合，温暖治愈

**👻 噩梦 (Nightmare)**
恐怖锐利风格，适合追求刺激的你。
- 中高层特征（物体、结构）
- 锐化滤镜，增强恐怖感
- 10% 现实融合，诡异扭曲

**🌫️ 迷雾梦 (Mist)**
朦胧模糊风格，像隔着雾气看世界。
- 3层特征叠加，层次感强
- 大量随机偏移，产生朦胧效果
- 适合营造神秘氛围

**🤪 疯狂梦 (Crazy)**
极致迷幻风格，彻底放飞想象力。
- 6层全特征，200次超多次迭代
- 0% 现实融合，完全脱离真实
- 迷幻艺术感拉满，视觉冲击强烈"""
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
