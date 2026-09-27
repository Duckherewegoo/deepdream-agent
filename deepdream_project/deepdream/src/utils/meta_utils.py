import json
import os
from typing import Dict, Any


def parse_meta_value(v: Any) -> Any:
    """把meta里序列化后的str转回原类型"""
    if isinstance(v, str):
        # 转bool
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        # 转数字
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            pass
        # 转list（比如"[1,2,3]"）
        if v.startswith("[") and v.endswith("]"):
            try:
                return eval(v)  # 仅用于自己生成的meta，安全
            except:
                pass
    return v


def load_dream_meta(image_path: str) -> Dict[str, Any]:
    """根据图片路径加载对应的.meta.json"""
    meta_path = os.path.splitext(image_path)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 反序列化config里的参数
        if "config" in meta:
            meta["config"] = {k: parse_meta_value(
                v) for k, v in meta["config"].items()}
        return meta
    except Exception as e:
        print(f"[META] 加载元数据失败: {e}")
        return {}
