"""梦境生成工具（策略路由版）。

本模块以 LangChain Tool 的形式，向大模型暴露两个核心能力：
  1. generate_dream：调用底层 DeepDreamGenerator 生成指定类型的梦境图片。
  2. get_dream_knowledge：从外部 markdown 知识库中读取梦境相关科普内容。

设计要点：
  - 字典路由：通过 DREAM_CONFIGS 与 DREAM_CN_NAMES 做类型分派，
    新增梦境类型时本模块无需改动，只需在配置文件中同步声明。
  - 线程安全单例：全局 DeepDreamGenerator 实例通过 threading.Lock 保护，
    防止并发请求下模型被重复实例化导致显存溢出。
  - 跨模块推理锁：`inference_lock` 同时被本模块与 node/dream_gen_node.py、
    chat_service 复现路径引用，保证所有生成入口串行访问 GPU。
  - 错误分级重试：仅在疑似显存不足（OOM）时清理缓存并重试一次，
    其他类型的异常（如磁盘满、图片损坏）直接熔断，避免无谓等待。
  - 知识库数据外置：所有梦境科普内容存放在 ../knowledges/*.md，
    代码中不硬编码任何文本，便于内容迭代与扩展。
"""
import json
import logging
import os
import threading
from pathlib import Path

from langchain_core.tools import tool

from ..config.settings import DREAM_CN_NAMES, DREAM_CONFIGS
from ..deepdream.generator import DeepDreamGenerator

logger = logging.getLogger(__name__)

# ================= 全局状态 =================
_dream_generator: DeepDreamGenerator | None = None
_generator_lock = threading.Lock()   # 只保护「实例化」，防止重复加载模型

# 跨模块共享的 GPU 推理锁。node/dream_gen_node.py 与 chat_service 复现路径
# 都会 import 这把锁，保证所有生成入口串行访问 GPU。
inference_lock = threading.Lock()

# 默认值与 generator.generate() / main.py / DreamAgent 保持一致，
# 避免 LLM 工具路径把图写到另一个目录。
_output_dir = "./dream_outputs"

# 动态定位知识库目录（无需硬编码绝对路径）
_KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledges"


def set_output_dir(output_dir: str) -> None:
    """由 DreamAgent.__init__ 调用，使 LLM 工具路径与主路径写入同一目录。"""
    global _output_dir
    _output_dir = os.path.abspath(output_dir)
    os.makedirs(_output_dir, exist_ok=True)
    logger.debug(f"[TOOL] 梦境输出目录已设置为: {_output_dir}")


def get_dream_generator(clear_cache: bool = False) -> DeepDreamGenerator:
    """获取全局唯一的 DeepDreamGenerator 实例。

    Args:
        clear_cache: 是否在返回前清空特征提取器缓存与显存碎片。
                     通常在疑似显存不足（OOM）重试时置为 True。

    Returns:
        单例的 DeepDreamGenerator 实例。
    """
    global _dream_generator
    if _dream_generator is None:
        with _generator_lock:
            if _dream_generator is None:
                logger.info("[TOOL] 正在初始化 DeepDreamGenerator 单例...")
                _dream_generator = DeepDreamGenerator()
                logger.info("[TOOL] DeepDreamGenerator 初始化完成")
    elif clear_cache:
        logger.info("[TOOL] 清理 DeepDreamGenerator 缓存")
        _dream_generator.clear_cache()
    return _dream_generator


def _build_json_response(data: dict) -> str:
    """统一将字典序列化为中文字符不转义的 JSON 字符串，供大模型消费。"""
    return json.dumps(data, ensure_ascii=False)


def _is_oom_error(exc: BaseException) -> bool:
    """判断异常是否属于疑似显存不足（Out Of Memory）范畴。"""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg


# ================= 工具：生成梦境 =================

@tool
def generate_dream(dream_type: str, user_image_path: str) -> str:
    """生成 DeepDream 梦境图片。

    Args:
        dream_type: 梦境类型标识，如 normal / sweet / nightmare 等。
                    若传入未知类型会自动降级为 normal。
        user_image_path: 用户上传的原始图片在磁盘上的绝对路径。

    Returns:
        包含 success、type、type_cn、path、msg 字段的 JSON 字符串。
    """
    if not user_image_path or not os.path.exists(user_image_path):
        return _build_json_response(
            {"success": False, "msg": "图片不存在", "dream_type": dream_type}
        )

    if dream_type not in DREAM_CONFIGS:
        logger.warning(f"[TOOL] 未知梦境类型 {dream_type!r}，降级为 normal")
        dream_type = "normal"

    dream_cn = DREAM_CN_NAMES.get(dream_type, dream_type)

    # 最多尝试 2 次，仅对显存类异常进行重试
    for attempt in range(2):
        try:
            # 把「获取生成器 + 清缓存 + 推理」全部放进推理锁内，
            # 防止 clear_cache 与另一线程的推理并发。
            with inference_lock:
                generator = get_dream_generator(clear_cache=(attempt > 0))
                output_path = generator.generate(
                    user_image_path, dream_type, output_dir=_output_dir
                )
            return _build_json_response(
                {
                    "success": True,
                    "type": dream_type,
                    "type_cn": dream_cn,
                    "path": output_path,
                    "msg": f"{dream_cn}生成成功",
                }
            )
        except Exception as e:
            if attempt == 0 and _is_oom_error(e):
                logger.warning(f"[TOOL] 疑似显存不足，清理缓存后重试: {e}")
                continue
            logger.exception(
                f"[TOOL] {dream_cn} 生成失败（第 {attempt + 1} 次尝试）: {e}"
            )
            break

    return _build_json_response(
        {"success": False, "msg": "生成失败", "dream_type": dream_type}
    )


# ================= 工具：梦境知识库 =================

@tool
def get_dream_knowledge(topic: str = "all") -> str:
    """获取梦境相关的科普知识。

    Args:
        topic: 知识主题标识。传 "all" 返回全部主题的合并内容；
               传具体主题名（如 types / deepdream）返回对应文档；
               未知主题返回包含可选主题列表的提示信息。

    Returns:
        包含 success、topic、content 等字段的 JSON 字符串。
    """
    if not _KNOWLEDGE_DIR.exists():
        logger.warning(f"[TOOL] 知识库目录不存在: {_KNOWLEDGE_DIR}")
        return _build_json_response({"success": False, "msg": "知识库目录不存在"})

    available_topics = {f.stem: f for f in _KNOWLEDGE_DIR.glob("*.md")}

    if not available_topics:
        logger.warning("[TOOL] 知识库目录为空")
        return _build_json_response({"success": False, "msg": "知识库为空"})

    if topic == "all":
        contents: list[str] = []
        for name, path in available_topics.items():
            try:
                content = path.read_text(encoding="utf-8")
                contents.append(f"## {name.capitalize()}\n{content}")
            except OSError as e:
                logger.error(f"[TOOL] 读取知识库文件失败 {path}: {e}")
                continue

        return _build_json_response(
            {
                "success": True,
                "topics": list(available_topics.keys()),
                "content": "\n\n".join(contents),
            }
        )

    if topic in available_topics:
        try:
            content = available_topics[topic].read_text(encoding="utf-8")
        except OSError as e:
            logger.error(f"[TOOL] 读取知识库文件失败 {available_topics[topic]}: {e}")
            return _build_json_response(
                {"success": False, "msg": f"读取主题 {topic} 失败"}
            )
        return _build_json_response(
            {"success": True, "topic": topic, "content": content}
        )

    return _build_json_response(
        {
            "success": False,
            "msg": f"未知主题：{topic}，可选：{', '.join(available_topics.keys())}",
        }
    )
