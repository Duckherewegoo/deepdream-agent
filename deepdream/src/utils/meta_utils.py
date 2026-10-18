import ast
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def parse_meta_value(v: Any) -> Any:
    """把 meta 里序列化后的 str 转回原类型"""
    if not isinstance(v, str):
        return v

    v_stripped = v.strip()

    # 1. 布尔值
    if v_stripped.lower() == "true":
        return True
    if v_stripped.lower() == "false":
        return False

    # 2. 数字（严格限制，只有看起来像数字的才转）
    try:
        # 判断是否带小数点（避免把 "001" 转成 1，如果这本身是个字符串）
        if "." in v_stripped or "e" in v_stripped.lower():
            return float(v_stripped)
        elif v_stripped.isdigit() or (v_stripped.startswith("-") and v_stripped[1:].isdigit()):
            return int(v_stripped)
    except ValueError:
        pass

    # 3. 列表 (核心修复：使用 ast.literal_eval 替代 eval)
    if v_stripped.startswith("[") and v_stripped.endswith("]"):
        try:
            result = ast.literal_eval(v_stripped)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            logger.warning(f"[META] 列表解析失败，保留原字符串: {v_stripped[:50]}")

    # 4. 其他情况，原样返回
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
