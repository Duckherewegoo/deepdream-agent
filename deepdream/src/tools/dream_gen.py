"""梦境生成工具（策略路由版）"""
import json
import logging
import os
from pathlib import Path

from langchain_core.tools import tool

from ..config.settings import DREAM_CN_NAMES, DREAM_CONFIGS
from ..deepdream.generator import DeepDreamGenerator

logger = logging.getLogger(__name__)

# ================= 全局状态 =================
_dream_generator = None
_output_dir = "./dream_outputs"

# ✅ 动态定位知识库目录（无需硬编码路径）
_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledges"

def set_output_dir(output_dir: str):
    global _output_dir
    _output_dir = os.path.abspath(output_dir)
    os.makedirs(_output_dir, exist_ok=True)
    logger.debug(f"[TOOL] 梦境输出目录已设置为: {_output_dir}")

def get_dream_generator(clear_cache: bool = False) -> DeepDreamGenerator:
    global _dream_generator
    if _dream_generator is None:
        _dream_generator = DeepDreamGenerator()
    elif clear_cache:
        _dream_generator.clear_cache()
    return _dream_generator

def _build_json_response(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


# ================= 工具：生成梦境 =================

@tool
def generate_dream(dream_type: str, user_image_path: str) -> str:
    """生成 DeepDream 梦境图片"""
    if not user_image_path or not os.path.exists(user_image_path):
        return _build_json_response({"success": False, "msg": "图片不存在", "dream_type": dream_type})

    # ✅ 字典路由：以后加梦境类型，这里完全不用动
    if dream_type not in DREAM_CONFIGS:
        dream_type = "normal"

    dream_cn = DREAM_CN_NAMES.get(dream_type, dream_type)
    
    for attempt in range(2):
        try:
            generator = get_dream_generator(clear_cache=(attempt > 0))
            output_path = generator.generate(user_image_path, dream_type, output_dir=_output_dir)
            
            return _build_json_response({
                "success": True,
                "type": dream_type,
                "type_cn": dream_cn,
                "path": output_path,
                "msg": f"✨ {dream_cn}生成成功！"
            })
        except Exception as e:
            logger.exception(f"[TOOL] {dream_cn} 生成失败 (尝试 {attempt+1}/2): {e}")

    return _build_json_response({"success": False, "msg": "生成失败", "dream_type": dream_type})


# ================= 工具：梦境知识库 =================

@tool
def get_dream_knowledge(topic: str = "all") -> str:
    """获取梦境相关知识（数据外置版）"""
    if not _KNOWLEDGE_DIR.exists():
        return _build_json_response({"success": False, "msg": "知识库目录不存在"})

    # 扫描所有的 md 文件：{'deepdream': '/path/...', 'types': '/path/...'}
    available_topics = {f.stem: f for f in _KNOWLEDGE_DIR.glob("*.md")}

    if not available_topics:
        return _build_json_response({"success": False, "msg": "知识库为空"})

    # ✅ 策略 1：请求所有
    if topic == "all":
        contents = []
        for name, path in available_topics.items():
            content = path.read_text(encoding="utf-8")
            contents.append(f"## {name.capitalize()}\n{content}")
            
        return _build_json_response({
            "success": True,
            "topics": list(available_topics.keys()),
            "content": "\n\n".join(contents)
        })

    # ✅ 策略 2：请求特定主题
    elif topic in available_topics:
        content = available_topics[topic].read_text(encoding="utf-8")
        return _build_json_response({
            "success": True,
            "topic": topic,
            "content": content
        })

    # ✅ 策略 3：找不到
    else:
        return _build_json_response({
            "success": False,
            "msg": f"未知主题：{topic}，可选：{', '.join(available_topics.keys())}"
        })
