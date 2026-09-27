"""DeepDream 配置常量"""
import os

# DashScope API 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 梦境类型关键词映射（修复文字识别）
DREAM_KEYWORDS = {
    "normal": ["正常梦", "普通梦", "常规", "normal"],
    "sweet": ["甜蜜梦", "美梦", "甜梦", "温柔", "治愈", "sweet", "美好"],
    "nightmare": ["噩梦", "恐怖", "惊悚", "吓人", "nightmare", "轰隆隆"],
    "mist": ["迷雾梦", "朦胧", "迷雾", "模糊", "mist", "雾"],
    "crazy": ["疯狂梦", "疯狂", "迷幻", "极致", "crazy", "啊啊啊"]  # 修复"疯狂"识别
}

# 梦境类型中文名称
DREAM_CN_NAMES = {
    "normal": "正常梦",
    "sweet": "美梦",
    "nightmare": "噩梦",
    "mist": "迷雾梦",
    "crazy": "疯狂梦"
}

# 各梦境类型的特征层选择（核心：不同层对应不同视觉特征）
DREAM_LAYERS = {
    # 中层，平衡
    "normal": ["Mixed_5b", "Mixed_5c"],

    # 偏底层（v3 里没有 Mixed_4a，用 Conv2d_4a_3x3）
    "sweet": ["Conv2d_3b_1x1", "Conv2d_4a_3x3"],

    # 高层，结构感强
    "nightmare": ["Mixed_6c", "Mixed_7a", "Mixed_7b"],

    # 多层混合
    "mist": ["Mixed_5b", "Mixed_6a", "Mixed_6c", "Mixed_7a"],

    # 全层疯狂梦（✅ 全部真实存在）
    "crazy": [
        "Conv2d_3b_1x1",
        "Conv2d_4a_3x3",
        "Mixed_5b",
        "Mixed_5c",
        "Mixed_6c",
        "Mixed_7b",
        "Mixed_7c"
    ]
}
# 优化后的梦境配置参数（核心：通过算法参数体现梦境特征，而非滤镜）
DREAM_CONFIGS = {
    # 正常梦：平衡真实与梦幻
    "normal": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 3,          # 增加octave层数，提升细节
        "OCTAVE_SCALE": 0.75,
        "ITER_PER_OCTAVE": 80,       # 更多迭代，增强DeepDream效果
        "STEP_SIZE": 0.01,
        "REALITY_FUSION": 0.18,      # 适度现实融合
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 1.0,       # 梯度缩放
        "COLOR_JITTER": 0.02         # 轻微色彩抖动
    },
    # 美梦：柔和、温暖、治愈（低层特征+小步长+高现实融合）
    "sweet": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 4,
        "OCTAVE_SCALE": 0.68,            # 更小 scale，细节更柔
        "ITER_PER_OCTAVE": 50,           # 少一点，别过度生长
        "STEP_SIZE": 0.006,              # 极小步长，缓慢演化
        "REALITY_FUSION": 0.12,          # 👈 降低，让梦多一点
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 0.6,           # 👈 更弱梯度，防止锐边
        "COLOR_JITTER": 0.015,
        "GRADIENT_NOISE": 0.003,         # 👈 加一点点噪声，像呼吸感
    },
    # 噩梦：扭曲、恐怖、锐利（高层特征+大步长+低现实融合）
    "nightmare": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 4,              # 👈 从 3 → 4，增强多尺度畸变
        "OCTAVE_SCALE": 0.82,            # 👈 更大 scale，结构更容易崩
        "ITER_PER_OCTAVE": 120,          # 更强 hallucination
        "STEP_SIZE": 0.018,              # 更大步长
        "REALITY_FUSION": 0.02,          # 👈 几乎完全脱离现实
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 1.8,           # 放大梯度，强化边缘
        "COLOR_JITTER": 0.04,            # 冷/紫/绿偏移
        "GRADIENT_NOISE": 0.005,         # 👈 加噪声，像信号干扰
    },
    # 迷雾梦：朦胧、模糊、神秘（多尺度+随机梯度扰动）
    "mist": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 5,              # ✅ 这个很好，保留
        "OCTAVE_SCALE": 0.72,
        "ITER_PER_OCTAVE": 60,           # 少一点，别太清晰
        "STEP_SIZE": 0.007,
        "REALITY_FUSION": 0.18,
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 0.7,           # 弱化细节
        "COLOR_JITTER": 0.025,
        "GRADIENT_NOISE": 0.012,         # 👈 关键：用噪声制造“雾感”
        # ❌ 删掉 BLUR_AMOUNT
    },
    # 疯狂梦：极致迷幻、无约束（全层特征+超多迭代+无现实融合）
    "crazy": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 4,
        "OCTAVE_SCALE": 0.63,            # 略小，让结构更碎
        "ITER_PER_OCTAVE": 220,          # 更久
        "STEP_SIZE": 0.022,              # 更大
        "REALITY_FUSION": 0.0,
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 2.2,           # 👈 再拉高一点
        "COLOR_JITTER": 0.07,            # 👈 强烈色偏
        "GRADIENT_NOISE": 0.008,         # 👈 关键：每一步都抖一下
        "DYNAMIC_STEP": True
    }}

# ============================================================
# 实验性梦境参数（不影响现有配置）
# 用法：cfg = EXPERIMENTAL_DREAM_CONFIGS.get("psychedelic", DREAM_CONFIGS["crazy"])
# ============================================================

EXPERIMENTAL_DREAM_CONFIGS = {
    # 极度迷幻模式（LLM 可触发）
    "psychedelic": {
        "MAX_SIZE": 900,
        "OCTAVE_LAYERS": 5,
        "OCTAVE_SCALE": 0.6,
        "ITER_PER_OCTAVE": 250,
        "STEP_SIZE": 0.025,
        "REALITY_FUSION": 0.0,
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 2.5,
        "COLOR_JITTER": 0.08,
        "GRADIENT_NOISE": 0.01,
        "DYNAMIC_STEP": True,
        "PSYCHEDELIC_INTENSITY": 1.4,  # ← 仅用于 LLM 描述
    },

    # 极简微梦（适合 subtle 描述）
    "subtle": {
        "MAX_SIZE": 700,
        "OCTAVE_LAYERS": 3,
        "OCTAVE_SCALE": 0.75,
        "ITER_PER_OCTAVE": 40,
        "STEP_SIZE": 0.004,
        "REALITY_FUSION": 0.25,
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 0.6,
        "COLOR_JITTER": 0.01,
        "GRADIENT_NOISE": 0.002,
        "DYNAMIC_STEP": False,
    },

    # 生物噩梦（结构感极强）
    "biologic": {
        "MAX_SIZE": 850,
        "OCTAVE_LAYERS": 4,
        "OCTAVE_SCALE": 0.7,
        "ITER_PER_OCTAVE": 150,
        "STEP_SIZE": 0.018,
        "REALITY_FUSION": 0.03,
        "SAFE_SHIFT": True,
        "GRADIENT_SCALE": 1.8,
        "COLOR_JITTER": 0.05,
        "GRADIENT_NOISE": 0.006,
        "DYNAMIC_STEP": True,
    },
}

DEBUG_MODE = True  # 开启后显示梯度/噪声细节
