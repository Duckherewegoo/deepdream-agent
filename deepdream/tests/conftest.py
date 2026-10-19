import os
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

def pytest_addoption(parser):
    """注册 pytest 命令行参数"""
    parser.addoption(
        "--real-llm",
        action="store_true",
        default=False,
        help="使用真实的 LLM 和 DeepDream 跑全流程测试（极慢且烧钱）",
    )

@pytest.fixture(scope="session")
def use_real_llm(request):
    return request.config.getoption("--real-llm")

@pytest.fixture
def mock_generator():
    """Mock 掉底层的 PyTorch 生成器，防止测试卡死或显存爆炸"""
    with patch("deepdream.src.deepdream.generator.DeepDreamGenerator.generate") as mock_gen:
        def fake_generate(input_path, dream_type, output_dir, custom_cfg=None, custom_layers=None):
            # 伪造一张返回图
            out_path = os.path.join(output_dir, f"mock_{dream_type}.png")
            os.makedirs(output_dir, exist_ok=True)
            Image.new("RGB", (100, 100), color="blue").save(out_path)
            # 顺便伪造一个 meta.json，测试 load_dream_meta 是否能读取
            meta_path = out_path.replace(".png", ".meta.json")
            with open(meta_path, "w") as f:
                f.write('{"dream_type": "%s", "config": {"STEP_SIZE": "0.01"}, "layers_used": ["Mixed_5b"]}' % dream_type)
            return out_path
        
        mock_gen.side_effect = fake_generate
        yield mock_gen

@pytest.fixture
def setup_environment(use_real_llm, mock_generator):
    """统一设置测试环境变量"""
    os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY", "fake_key_for_test")
    # 如果是 Mock 模式，静默替换掉 LLM 和真实生成器
    # （这里为了极简，主要靠 mock_generator 拦截生成，LLM 靠环境变量拦截）
    yield
