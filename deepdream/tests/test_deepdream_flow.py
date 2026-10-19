import pytest

def test_full_flow_success(deepdream_engine):
    """
    全流程冒烟测试：输入 -> 引擎处理 -> 返回结构化结果
    """
    # 1. 模拟用户输入
    user_query = "帮我处理一张测试图片"
    
    # 2. 执行核心流程（如果是异步的，记得用 await 和 pytest.mark.asyncio）
    # result = await deepdream_engine.run(user_query)
    result = deepdream_engine.run(user_query)
    
    # 3. 断言结果
    assert result is not None
    assert isinstance(result, dict)
    assert result.get("status") == "success"
    
    # 4. 验证流程确实跑完了，而不仅仅是没报错
    assert "output" in result
    assert len(result.get("steps", [])) > 0


def test_full_flow_error_handling(deepdream_engine):
    """
    全流程异常测试：当底层发生异常时，引擎是否能优雅降级或报错
    （如果是 Mock，这里故意让 mock 抛出异常）
    """
    # 模拟引擎内部抛出异常
    deepdream_engine.run.side_effect = RuntimeError("大模型连接超时")
    
    # 如果你的引擎内部有 try-except 兜底，应该返回 failed 而不是让测试崩掉
    try:
        result = deepdream_engine.run("触发错误的输入")
        assert result.get("status") == "failed"
        assert "超时" in result.get("error", "")
    except RuntimeError as e:
        # 如果你的设计是直接抛出异常（由上层 API 捕获），这里直接断言异常即可
        assert str(e) == "大模型连接超时"
