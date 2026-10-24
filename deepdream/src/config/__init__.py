"""配置模块"""
from .llms import LLM, LLM_MAX_TOKENS, LLM_SEARCH, LLM_SEED, LLM_STOP, LLM_TEMP, LLM_THINKING, LLM_TOP
from .prompts import SYSTEM_PROMPTS
from .settings import DREAM_CN_NAMES, DREAM_CONFIGS, DREAM_KEYWORDS, DREAM_LAYERS

__all__ = [
           "DREAM_CN_NAMES",
           "DREAM_CONFIGS",
           "DREAM_KEYWORDS",
           "DREAM_LAYERS",
           "LLM",
           "LLM_MAX_TOKENS",
           "LLM_SEARCH",
           "LLM_SEED",
           "LLM_STOP",
           "LLM_TEMP",
           "LLM_THINKING",
           "LLM_TOP",
           "SYSTEM_PROMPTS"
           ]
