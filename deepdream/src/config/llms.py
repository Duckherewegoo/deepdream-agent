"""
LLM 核心配置文件
====================================
集中管理调用大语言模型（如通义千问）时的全局默认参数。
所有模块在发起 API 请求前，都从这里读取基础配置。

⚠️ 关键参数说明与避坑指南：
1. LLM_MAX_TOKENS (最大输出长度)：
   - 注意：如果开启了 LLM_THINKING（思考模式），模型在输出最终结果前会先输出一段思维链。
   - 思维链同样消耗 Token。如果此值设置过小（如 200），会导致模型“刚思考完就被截断”，甚至报错。
   - 建议设置为 500 以上。

2. LLM_STOP (停止词)：
   - 当前设置为 "}"，意在强制模型在生成 JSON 对象结束时立刻刹车。
   - 风险：如果要求模型输出嵌套 JSON，它在生成内层 } 时就会停止，导致 JSON 结构残缺。
   - 非强依赖 JSON 的场景，建议清空此列表。

3. LLM_SEED (随机种子)：
   - 固定种子可以在相同参数下产生确定性的结果，非常有利于单元测试和调试复现。
   - 如果需要模型每次都发散思考，请将其设为 None。

（生产环境建议通过环境变量覆盖这些默认值，避免修改代码。）
"""
import os

LLM: str = os.getenv("DREAM_LLM_MODEL", "qwen-flash")
LLM_TEMP: float = float(os.getenv("DREAM_LLM_TEMP", "0.65"))
LLM_TOP: float = float(os.getenv("DREAM_LLM_TOP", "0.8"))
LLM_SEARCH: bool = os.getenv("DREAM_LLM_SEARCH", "false").lower() == "true"

LLM_MAX_TOKENS: int = int(os.getenv("DREAM_LLM_MAX_TOKENS", "500"))
LLM_SEED: int = int(os.getenv("DREAM_LLM_SEED", "24"))
LLM_THINKING: bool = os.getenv("DREAM_LLM_THINKING", "true").lower() == "true"

LLM_STOP: list[str] = ["}"]
