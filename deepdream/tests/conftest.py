import os
import pytest
from unittest.mock import MagicMock

def pytest_addoption(parser):
    """注册命令行参数"""
    parser.addoption(
        "--real-llm",
        action="store_true",
        default=False,
        help="运行真实 LLM 测试（需要配置环境变量）"
    )

@pytest.fixture(scope="session")
def use_real_llm(request):
    return request.config.getoption("--real-llm")

@pytest.fixture
def deepdream_engine(use_real_llm):
    """
    核心 Fixture：根据参数返回真实或 Mock 的引擎实例
    （你需要把下面的 import 替换为 deepdream 真实的入口类）
    """
    # 假设你的项目入口是 from deepdream.core import DeepDreamEngine
    # from deepdream.core import DeepDreamEngine
    
    if use_real_llm:
        api_key = os.getenv("OPENAI_API_KEY") # 或者你用的其他 Key
        if not api_key:
            pytest.skip("未配置 API Key，跳过真实 LLM 测试")
        
        # 真实环境下的初始化逻辑
        # engine = DeepDreamEngine(api_key=api_key, ...)
        # return engine
        pass
    else:
        # 极度克制：Mock 掉所有对外部 API 和底层组件的依赖
        # 如果你使用 LangChain，建议换成 FakeListChatModel
        mock_engine = MagicMock()
        
        # 模拟引擎核心方法的返回值
        mock_engine.run.return_value = {
            "status": "success",
            "output": "这是一个模拟的 DeepDream 处理结果",
            "steps": ["初始化", "处理图像", "完成"]
        }
        return mock_engine
