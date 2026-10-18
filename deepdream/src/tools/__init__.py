"""工具模块"""
from .calculator import scientific_calculator
from .dream_gen import generate_dream, get_dream_knowledge, set_output_dir
from .weather import get_weather

# 所有工具列表
TOOLS = [generate_dream, get_weather, scientific_calculator, get_dream_knowledge]

# 工具名称到工具的映射
TOOL_MAP = {t.name: t for t in TOOLS}

__all__ = [
    "TOOLS",
    "TOOL_MAP",
    "generate_dream",
    "get_dream_knowledge",
    "get_weather",
    "scientific_calculator",
    "set_output_dir",
]
