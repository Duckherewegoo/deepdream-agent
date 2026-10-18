"""DeepDream 配置常量"""
import logging
import os

logger = logging.getLogger(__name__)

# ================= API 配置 =================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# 控制是否输出底层调试信息（默认关闭，需要时在环境变量里设 DEEPDREAM_DEBUG=true）
DEBUG_MODE = os.getenv("DEEPDREAM_DEBUG", "false").lower() == "true"


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

DREAM_CONFIGS = {
    "normal": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 3, "OCTAVE_SCALE": 0.75, "ITER_PER_OCTAVE": 80,
        "STEP_SIZE": 0.01, "REALITY_FUSION": 0.18, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.0,
        "COLOR_JITTER": 0.02, "GRADIENT_NOISE": 0.0, "DYNAMIC_STEP": False
    },
    "sweet": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.68, "ITER_PER_OCTAVE": 50,
        "STEP_SIZE": 0.006, "REALITY_FUSION": 0.12, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.6,
        "COLOR_JITTER": 0.015, "GRADIENT_NOISE": 0.003, "DYNAMIC_STEP": False
    },
    "nightmare": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.82, "ITER_PER_OCTAVE": 120,
        "STEP_SIZE": 0.018, "REALITY_FUSION": 0.02, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.8,
        "COLOR_JITTER": 0.04, "GRADIENT_NOISE": 0.005, "DYNAMIC_STEP": False
    },
    "mist": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 5, "OCTAVE_SCALE": 0.72, "ITER_PER_OCTAVE": 60,
        "STEP_SIZE": 0.007, "REALITY_FUSION": 0.18, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.7,
        "COLOR_JITTER": 0.025, "GRADIENT_NOISE": 0.012, "DYNAMIC_STEP": False
    },
    "crazy": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.63, "ITER_PER_OCTAVE": 220,
        "STEP_SIZE": 0.022, "REALITY_FUSION": 0.0, "SAFE_SHIFT": True, "GRADIENT_SCALE": 2.2,
        "COLOR_JITTER": 0.07, "GRADIENT_NOISE": 0.008, "DYNAMIC_STEP": True
    },
    # ===== 新梦境类型 =====
    "starlight": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 5, "OCTAVE_SCALE": 0.65, "ITER_PER_OCTAVE": 90,
        "STEP_SIZE": 0.008, "REALITY_FUSION": 0.05, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.2,
        "COLOR_JITTER": 0.06, "GRADIENT_NOISE": 0.015, "DYNAMIC_STEP": False,
        # 独有扩展参数（供 generator.py 识别）
        "STAR_BRIGHTNESS": 1.4
    },
    "ocean": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.70, "ITER_PER_OCTAVE": 70,
        "STEP_SIZE": 0.009, "REALITY_FUSION": 0.15, "SAFE_SHIFT": True, "GRADIENT_SCALE": 0.8,
        "COLOR_JITTER": 0.04, "GRADIENT_NOISE": 0.008, "DYNAMIC_STEP": False,
        # 独有扩展参数（供 generator.py 识别）
        "WAVE_FLOW": 0.3, "DEPTH_FADE": 0.6
    },
    "floral": {
        "MAX_SIZE": 800, "OCTAVE_LAYERS": 4, "OCTAVE_SCALE": 0.72, "ITER_PER_OCTAVE": 80,
        "STEP_SIZE": 0.012, "REALITY_FUSION": 0.08, "SAFE_SHIFT": True, "GRADIENT_SCALE": 1.0,
        "COLOR_JITTER": 0.08, "GRADIENT_NOISE": 0.005, "DYNAMIC_STEP": False,
        # 独有扩展参数
        "PETAL_RADIUS": 0.5
    }
}


# ================= 3. 实验性参数 =================
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

# 优化：只校验所有梦境必备的“基础通用参数”，允许各梦境有自己独有的扩展参数
_REQUIRED_BASE_PARAMS = {
    "MAX_SIZE", "OCTAVE_LAYERS", "OCTAVE_SCALE", "ITER_PER_OCTAVE",
    "STEP_SIZE", "REALITY_FUSION", "SAFE_SHIFT", "GRADIENT_SCALE",
    "COLOR_JITTER", "GRADIENT_NOISE", "DYNAMIC_STEP"
}

for dream_type, cfg in DREAM_CONFIGS.items():
    missing = _REQUIRED_BASE_PARAMS - set(cfg.keys())
    if missing:
        raise RuntimeError(f"[CONFIG ERROR] {dream_type} 缺少基础参数: {missing}")

logger.debug("[CONFIG] 所有梦境配置一致性校验通过（含 3 种新梦境扩展）。")
