"""
logger_setup.py — 日志基础设施
提供标准 logging 配置，并支持将日志实时推送到 Gradio UI。
"""
import logging
import sys
import threading
from collections import deque

# 全局日志缓存队列（供前端读取）
_ui_logs = deque(maxlen=300)
_logger = logging.getLogger(__name__)
_logs_lock = threading.Lock()

class GradioUIHandler(logging.Handler):
    """
    自定义 Handler：将格式化的日志存入 deque，供 Gradio 前端轮询。
    """

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            # 过滤掉不需要展示给用户的第三方库日志（比如 httpx, PIL 等）
            if record.name.startswith("deepdream"):
                with _logs_lock:
                    _ui_logs.append(msg)
        except Exception:
            # 必须捕获所有异常并调用 handleError，
            # 否则日志记录失败会导致整个程序崩溃。
            self.handleError(record)


def get_ui_logs() -> str:
    """供 Gradio 定时器调用的接口：获取并拼接当前缓存里的日志。"""
    with _logs_lock:
        return "\n".join(_ui_logs)


def clear_ui_logs():
    """清空前端日志"""
    with _logs_lock:
        _ui_logs.clear()


def setup_logger(level=logging.INFO):
    """
    初始化全局日志系统。
    1. 输出到控制台
    2. 输出到 Gradio UI 队列
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 防止重复添加 Handler（比如热重载时）
    if root_logger.handlers:
        return

    # 1. 定义格式
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    ui_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # 2. 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # 3. Gradio UI Handler
    ui_handler = GradioUIHandler()
    ui_handler.setFormatter(ui_fmt)
    root_logger.addHandler(ui_handler)

    # 4. 压制一些烦人的第三方库日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # 使用自己的 logger 记录，避免触发 Ruff 的 LOG015 警告
    _logger.info("系统日志初始化完成")
