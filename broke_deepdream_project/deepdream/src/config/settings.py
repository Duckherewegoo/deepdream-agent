"""梦境配置 - 保留原有5种梦境配置和参数"""
from typing import Dict

# 5种梦境配置（完全保留原有参数）
DREAM_CONFIGS: Dict[str, Dict] = {
    "normal": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 2,
        "OCTAVE_SCALE": 0.85,
        "ITER_PER_OCTAVE": 40,
        "STEP_SIZE": 0.008,
        "REALITY_FUSION": 0.15,
        "SAFE_SHIFT": False,
    },
    "sweet": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 2,
        "OCTAVE_SCALE": 0.9,
        "ITER_PER_OCTAVE": 35,
        "STEP_SIZE": 0.006,
        "REALITY_FUSION": 0.2,
        "SAFE_SHIFT": False,
    },
    "nightmare": {
        "MAX_SIZE": 800,
        "OCTAVE_LAYERS": 2,
        "OCTAVE_SCALE": 0.8,
        "ITER_PER_OCTAVE": 60,
        "STEP_SIZE": 0.012,
        "REALITY_FUSION": 0.1,
        "SAFE_SHIFT": True,
    },
    "mist": {
        "MAX_SIZE": 1000,
        "OCTAVE_LAYERS": 3,
        "OCTAVE_SCALE": 0.75,
        "ITER_PER_OCTAVE": 80,
        "STEP_SIZE": 0.012,
        "REALITY_FUSION": 0.1,
        "SAFE_SHIFT": True,
    },
    "crazy": {
        "MAX_SIZE": 1000,
        "OCTAVE_LAYERS": 6,
        "OCTAVE_SCALE": 0.75,
        "ITER_PER_OCTAVE": 200,
        "STEP_SIZE": 0.03,
        "REALITY_FUSION": 0.0,
        "SAFE_SHIFT": True,
    },
}

# 梦境中文名称映射
DREAM_CN_NAMES: Dict[str, str] = {
    "normal": "正常梦",
    "sweet": "美梦",
    "nightmare": "噩梦",
    "mist": "迷雾梦",
    "crazy": "疯狂梦",
}

# 梦境关键词映射（用于从用户输入中识别梦境类型）
DREAM_KEYWORDS: Dict[str, list] = {
    "normal": ["做梦", "正常梦", "普通梦"],
    "sweet": ["美梦", "甜蜜梦", "好梦"],
    "nightmare": ["噩梦", "恶梦", "恐怖梦"],
    "mist": ["迷雾梦", "朦胧梦", "模糊梦"],
    "crazy": ["疯狂梦", "狂热梦", "迷幻梦"],
}

# InceptionV3 层配置（对应原有 TensorFlow 版本的层选择）
DREAM_LAYERS: Dict[str, list] = {
    "normal": ["Mixed_5b", "Mixed_5c"],  # 对应 mixed0, mixed1
    "sweet": ["Mixed_5b"],                # 对应 mixed0
    "nightmare": ["Mixed_6b", "Mixed_6c"], # 对应 mixed3, mixed4
    "mist": ["Mixed_5d", "Mixed_6b", "Mixed_6c"],  # 对应 mixed2, mixed3, mixed4
    "crazy": [f"Mixed_{i}{c}" for i in range(5, 8) for c in ['b', 'c', 'd']][:7],  # 对应 mixed0-6
}
