import ast
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ==========================================
# 预编译正则 (提升大量数据解析时的性能)
# ==========================================
# 匹配严格整数：允许正负号，且禁止前导零（防止 "001" 被错误转换）
_INT_PATTERN = re.compile(r"^[+-]?(?:0|[1-9]\d*)$")

# 匹配严格浮点数/科学计数法：如 3.14, -0.5, 1e-5, +2.5E10
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def parse_meta_value(v: Any) -> Any:
    """
    把 meta 里序列化后的 str 安全地转回原类型。
    加强版特性：
      1. 支持 None 识别
      2. 支持字典 {} 识别
      3. 引入正则预编译，精准区分数字与前导零字符串
      4. 全程使用 ast.literal_eval，杜绝 RCE 漏洞
    """
    # 1. 前置防御：非字符串直接返回
    if not isinstance(v, str):
        return v

    v_stripped = v.strip()

    # 2. 基础常量解析 (布尔值 & None)
    lower_val = v_stripped.lower()
    if lower_val == "true":
        return True
    if lower_val == "false":
        return False
    if lower_val in ("none", "null"):
        return None

    # 3. 智能数字解析
    # 先判断整数，再判断浮点数，正则严格限制，避免 "001" 被误转
    if _INT_PATTERN.match(v_stripped):
        try:
            return int(v_stripped)
        except ValueError:
            pass
    elif _FLOAT_PATTERN.match(v_stripped):
        try:
            return float(v_stripped)
        except ValueError:
            pass

    # 4. 复合数据结构解析 (列表 [] / 字典 {} / 元组 ())
    # 只要首尾符号对得上，就尝试用 literal_eval 解析
    if (
        (v_stripped.startswith("[") and v_stripped.endswith("]")) or
        (v_stripped.startswith("{") and v_stripped.endswith("}")) or
        (v_stripped.startswith("(") and v_stripped.endswith(")"))
    ):
        try:
            # 核心安全机制：绝不使用 eval()
            result = ast.literal_eval(v_stripped)
            # 确保解析出来的确实是复合类型，防止被畸形字符串钻空子
            if isinstance(result, (list, dict, tuple)):
                return result
        except (ValueError, SyntaxError):
            logger.warning(f"[META] 结构解析失败，保留原字符串: {v_stripped[:50]}")

    # 5. 其他情况，原样返回（注意：返回的是原始带空格的 v，以保持数据原貌）
    return v

def load_dream_meta(image_path: str | None) -> dict[str, Any]:
    """根据图片路径加载对应的 .meta.json"""
    if not image_path:
        return {}

    try:
        # 更稳健的路径推导：直接替换后缀
        base, _ = os.path.splitext(image_path)
        meta_path = f"{base}.meta.json"

        if not os.path.exists(meta_path):
            # 找不到 meta 文件时不报错，直接返回空字典（上层会做降级）
            return {}

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # 反序列化 config 里的参数
        if isinstance(meta, dict) and isinstance(meta.get("config"), dict):
            meta["config"] = {
                k: parse_meta_value(v) for k, v in meta["config"].items()
            }

        return meta

    except json.JSONDecodeError as e:
        logger.error(f"[META] JSON 格式损坏: {meta_path} | 错误: {e}")
        return {}
    except OSError as e:
        logger.error(f"[META] 读取文件失败: {meta_path} | 错误: {e}")
        return {}
    except Exception:
        # 兜底，保证 UI 层永远不会因为 Meta 加载而崩溃
        logger.exception("[META] 未预期的异常")
        return {}
