"""
DeepDream 全局配置中心 (settings.py)
====================================
本文件是整个梦境生成系统的唯一事实来源（Single Source of Truth）。
所有模型层、算法层、UI层所需的参数都在此定义。

⚠️ 核心契约与维护指南：
1. 基础映射字典（DREAM_CN_NAMES / DREAM_KEYWORDS / DREAM_LAYERS）：
   如果你打算新增一种梦境类型（例如 "cyberpunk"），必须在这三个字典中
   同时添加对应的键值对，并且在下方的 DREAM_CONFIGS 中配齐所有基础参数。

2. 核心算法参数（DREAM_CONFIGS）：
   - 强校验：文件底部有 Fail-Fast 逻辑，如果新增梦境缺少必填参数或类型写错，
     系统会在启动时报错，而不是跑到一半才崩溃。
   - 扩展参数：允许不同类型的梦境拥有独有参数（如 ocean 的 WAVE_FLOW），
     底层生成器会自动识别并使用，不会影响通用校验。
   - 幽灵参数警告：确保这里的参数在 `generator.py` 中都有对应的消费逻辑，
     否则调整参数将毫无意义。当前已清理 COLOR_JITTER / DEPTH_FADE / PETAL_RADIUS。

3. 实验性参数（EXPERIMENTAL_DREAM_CONFIGS）：
   存放尚未正式接入 UI 的梦境配方，仅作备用，不参与强校验。

4. API Key 策略：
   未配置 DASHSCOPE_API_KEY 且非 DEBUG_MODE 时，仅告警不阻断启动；
   生产环境如需强制阻断，可将下方 warning 改为 raise RuntimeError。
"""
import logging
import os

logger = logging.getLogger(__name__)

# ================= API 配置 =================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 控制是否输出底层调试信息（默认关闭，需要时在环境变量里设 DEEPDREAM_DEBUG=true）
DEBUG_MODE = os.getenv("DEEPDREAM_DEBUG", "false").lower() == "true"

# 防线前移：未配置 Key 且非调试模式时，至少在启动阶段给出明确告警
if not DASHSCOPE_API_KEY and not DEBUG_MODE:
    logger.warning(
        "[CONFIG] 警告：DASHSCOPE_API_KEY 未配置，真实生成功能将不可用！"
        "如需本地调试请设置 DEEPDREAM_DEBUG=true。"
    )


# ================= 1. 基础映射字典 =================

# 梦境类型中文名称
DREAM_CN_NAMES = {
    "normal": "正常梦",
    "sweet": "美梦",
    "nightmare": "噩梦",
    "mist": "迷雾梦",
    "crazy": "疯狂梦",
    "starlight": "星空梦",
    "ocean": "海洋梦",
    "floral": "花海梦"
}

# 梦境类型关键词映射
DREAM_KEYWORDS = {
    "normal": ["正常梦", "普通梦", "常规", "normal"],
    "sweet": ["甜蜜梦", "美梦", "甜梦", "温柔", "治愈", "美好", "sweet"],
    "nightmare": ["噩梦", "恐怖", "惊悚", "吓人", "nightmare"],
    "mist": ["迷雾梦", "朦胧", "迷雾", "模糊", "mist", "雾蒙蒙"],
    "crazy": ["疯狂梦", "疯狂", "迷幻", "极致", "crazy"],
    "starlight": ["星空梦", "星梦", "宇宙梦", "银河", "starlight", "星星"],
    "ocean": ["海洋梦", "深海梦", "波浪梦", "ocean", "海"],
    "floral": ["花海梦", "花梦", "花园梦", "floral", "花"]
}

# 各梦境类型的特征层选择
DREAM_LAYERS = {
    "normal": ["Mixed_5b", "Mixed_5c"],
    "sweet": ["Conv2d_3b_1x1", "Conv2d_4a_3x3"],
    "nightmare": ["Mixed_6c", "Mixed_7a", "Mixed_7b"],
    "mist": ["Mixed_5b", "Mixed_6a", "Mixed_6c", "Mixed_7a"],
    "crazy": [
        "Conv2d_3b_1x1", "Conv2d_4a_3x3", "Mixed_5b",
        "Mixed_5c", "Mixed_6c", "Mixed_7b", "Mixed_7c"
    ],
    # 星空：中层 + 高层混合，产生结构感 + 纹理闪烁
    "starlight": ["Mixed_5b", "Mixed_5c", "Mixed_6a", "Mixed_6c"],
    # 海洋：低层 + 中层，强调纹理流动
    "ocean": ["Conv2d_4a_3x3", "Mixed_5b", "Mixed_5c"],
    # 花海：中层，强调繁复的纹理与色彩
    "floral": ["Mixed_5b", "Mixed_5c", "Mixed_6a"]
}


# ================= 2. 核心算法参数 =================
# 说明：以下参数与 generator.py 的消费逻辑严格对齐。
# 通用参数见 _REQUIRED_BASE_PARAMS，扩展参数由各梦境按需声明。

DREAM_CONFIGS = {
    "normal": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 3, "OCTAVE_SCALE": 0.75, "ITER_PER_OCTAVE": 80,
        "STEP_SIZE": 0.01, "REALITY_FUSION": 0.18, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.0,
        "GRADIENT_NOISE": 0.0, "DYNAMIC_STEP": False,
        # 基础层权重与后处理
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": 0.0,
        "POST_SHARPEN": False, "POST_CONTRAST": 1.0
    },
    "sweet": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.68, "ITER_PER_OCTAVE": 50,
        "STEP_SIZE": 0.006, "REALITY_FUSION": 0.12, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.6,
        "GRADIENT_NOISE": 0.003, "DYNAMIC_STEP": False,
        # 压制深层、强化浅层 -> 负斜率
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": -0.15,
        # 美梦专属后处理 -> 柔光、提亮、加饱和
        "POST_BLUR": 0.8, "POST_BRIGHTNESS": 1.1, "POST_COLOR": 1.05
    },
    "nightmare": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.82, "ITER_PER_OCTAVE": 120,
        "STEP_SIZE": 0.018, "REALITY_FUSION": 0.02, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.8,
        "GRADIENT_NOISE": 0.005, "DYNAMIC_STEP": True,
        # 动态步长线性放大 + 强化深层 -> 正斜率
        "STEP_FACTOR": 0.6, "STEP_POWER": 1,
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": 0.35,
        # 噩梦专属后处理 -> 锐化、压暗、加对比
        "POST_SHARPEN": True, "POST_BRIGHTNESS": 0.9, "POST_CONTRAST": 1.1
    },
    "mist": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 5, "OCTAVE_SCALE": 0.72, "ITER_PER_OCTAVE": 60,
        "STEP_SIZE": 0.007, "REALITY_FUSION": 0.18, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.7,
        "GRADIENT_NOISE": 0.012, "DYNAMIC_STEP": False,
        # 噪声用 sigmoid 压制（防噪点爆炸）+ 压制深层
        "NOISE_MOD": "sigmoid", "NOISE_FACTOR": -0.3,
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": -0.25
    },
    "crazy": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.63, "ITER_PER_OCTAVE": 220,
        "STEP_SIZE": 0.022, "REALITY_FUSION": 0.0, "SAFE_SHIFT": True, "GRADIENT_SCALE": 2.2,
        "GRADIENT_NOISE": 0.008, "DYNAMIC_STEP": True,
        # 噪声用 tanh 放大 + 步长平方级爆炸 + 基础权重拉高
        "NOISE_MOD": "tanh", "NOISE_FACTOR": 0.4,
        "STEP_FACTOR": 0.9, "STEP_POWER": 2,
        "LAYER_WEIGHT_BASE": 1.2, "LAYER_WEIGHT_SLOPE": 0.25,
        # 疯狂梦后处理 -> 色彩爆炸
        "POST_COLOR": 1.2
    },
    # ===== 新梦境类型 =====
    "starlight": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 5, "OCTAVE_SCALE": 0.65, "ITER_PER_OCTAVE": 90,
        "STEP_SIZE": 0.008, "REALITY_FUSION": 0.05, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.2,
        "GRADIENT_NOISE": 0.015, "DYNAMIC_STEP": False,
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": 0.1,  # 稍微偏深层
        # 独有扩展参数
        "STAR_BRIGHTNESS": 1.4
    },
    "ocean": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.70, "ITER_PER_OCTAVE": 70,
        "STEP_SIZE": 0.009, "REALITY_FUSION": 0.15, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.8,
        "GRADIENT_NOISE": 0.008, "DYNAMIC_STEP": False,
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": -0.1,
        # 独有扩展参数
        "WAVE_FLOW": 0.3
    },
    "floral": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.72, "ITER_PER_OCTAVE": 80,
        "STEP_SIZE": 0.012, "REALITY_FUSION": 0.08, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.0,
        "GRADIENT_NOISE": 0.005, "DYNAMIC_STEP": False,
        "LAYER_WEIGHT_BASE": 1.0, "LAYER_WEIGHT_SLOPE": 0.0,
        # 花海需要高饱和
        "POST_COLOR": 1.15
    }
}


# ================= 3. 实验性参数 =================
# 注意：以下配方尚未接入 UI，不参与强校验，仅作占位/预研用途。
EXPERIMENTAL_DREAM_CONFIGS = {
    "psychedelic": {
        "MAX_SIZE": 900, "OCTAVE_LAYERS": 5, "OCTAVE_SCALE": 0.6, "ITER_PER_OCTAVE": 250,
        "STEP_SIZE": 0.025, "REALITY_FUSION": 0.0, "SAFE_SHIFT": True, "GRADIENT_SCALE": 2.5,
        "COLOR_JITTER": 0.08, "GRADIENT_NOISE": 0.01, "DYNAMIC_STEP": True,
        "PSYCHEDELIC_INTENSITY": 1.4,
    },
    "subtle": {
        "MAX_SIZE": 700, "OCTAVE_LAYERS": 3, "OCTAVE_SCALE": 0.75, "ITER_PER_OCTAVE": 40,
        "STEP_SIZE": 0.004, "REALITY_FUSION": 0.25, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.6,
        "COLOR_JITTER": 0.01, "GRADIENT_NOISE": 0.002, "DYNAMIC_STEP": False,
    },
    "biologic": {
        "MAX_SIZE": 850, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.7, "ITER_PER_OCTAVE": 150,
        "STEP_SIZE": 0.018, "REALITY_FUSION": 0.03, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.8,
        "COLOR_JITTER": 0.05, "GRADIENT_NOISE": 0.006, "DYNAMIC_STEP": True,
    },
}


# ================= 4. 模块级自检（Fail-Fast） =================

_EXPECTED_KEYS = set(DREAM_CN_NAMES.keys())

if set(DREAM_KEYWORDS.keys()) != _EXPECTED_KEYS:
    raise RuntimeError("[CONFIG ERROR] DREAM_KEYWORDS 字典与标准梦境类型不匹配！")

if set(DREAM_LAYERS.keys()) != _EXPECTED_KEYS:
    raise RuntimeError("[CONFIG ERROR] DREAM_LAYERS 字典与标准梦境类型不匹配！")

if set(DREAM_CONFIGS.keys()) != _EXPECTED_KEYS:
    raise RuntimeError("[CONFIG ERROR] DREAM_CONFIGS 字典与标准梦境类型不匹配！")

# 所有梦境必备的「基础通用参数」，允许各梦境有自己独有的扩展参数。
# 注意：COLOR_JITTER 已被清理，不再是通用参数。
_REQUIRED_BASE_PARAMS = {
    "MAX_SIZE", "OCTAVE_LAYERS", "OCTAVE_SCALE", "ITER_PER_OCTAVE",
    "STEP_SIZE", "REALITY_FUSION", "SAFE_SHIFT", "GRADIENT_SCALE",
    "GRADIENT_NOISE", "DYNAMIC_STEP",
    "LAYER_WEIGHT_BASE", "LAYER_WEIGHT_SLOPE",
}

# 基础参数类型契约，防止 "0.01" / "True" 这类手滑。
_TYPE_CHECKS = {
    "MAX_SIZE": int,
    "OCTAVE_LAYERS": int,
    "OCTAVE_SCALE": float,
    "ITER_PER_OCTAVE": int,
    "STEP_SIZE": float,
    "REALITY_FUSION": float,
    "SAFE_SHIFT": bool,
    "GRADIENT_SCALE": float,
    "GRADIENT_NOISE": float,
    "DYNAMIC_STEP": bool,
    "LAYER_WEIGHT_BASE": float,
    "LAYER_WEIGHT_SLOPE": float,
}

for dream_type, cfg in DREAM_CONFIGS.items():
    # 1) 键缺失检查
    missing = _REQUIRED_BASE_PARAMS - set(cfg.keys())
    if missing:
        raise RuntimeError(f"[CONFIG ERROR] {dream_type} 缺少基础参数: {missing}")

    # 2) 类型强校验：拦截手滑写错格式（如 "0.01"、"True"）
    for param, expected_type in _TYPE_CHECKS.items():
        value = cfg[param]
        # bool 是 int 的子类，需要单独防护，避免把 True/False 当成数值
        if expected_type is int and isinstance(value, bool):
            raise TypeError(
                f"[CONFIG ERROR] {dream_type}.{param} 类型错误：期望 int，实际 bool ({value!r})"
            )
        if not isinstance(value, expected_type):
            raise TypeError(
                f"[CONFIG ERROR] {dream_type}.{param} 类型错误："
                f"期望 {expected_type.__name__}，实际 {type(value).__name__} ({value!r})"
            )

logger.debug("[CONFIG] 所有梦境配置一致性校验通过（含类型强校验与契约对齐）。")
