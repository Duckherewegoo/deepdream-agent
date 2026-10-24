"""
components.py — UI 组件库
提供无状态、纯计算的 UI 渲染逻辑（将底层数据转为界面所需格式）。
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

def render_meta_card(meta: dict[str, Any] | None) -> str:
    if not meta:
        return "📭 暂无梦境参数（图片未生成或Meta文件缺失）"

    cfg = meta.get("config", {})

    # 强制类型转换，防 TypeError
    try:
        step_size = float(cfg.get("STEP_SIZE", 0))
    except (ValueError, TypeError):
        step_size = 0.0

    if step_size > 0.02:
        intensity = "极高"
    elif step_size > 0.01:
        intensity = "中等"
    else:
        intensity = "微弱"

    # 确保 layers_used 绝对是 list
    layers = meta.get("layers_used", [])
    if not isinstance(layers, list):
        layers = [str(layers)]  # 如果是单独的字符串，包成 list
    layers_str = ', '.join(layers) if layers else "无"

    return f"""
<div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; background:#f8f9fa; font-size:14px; margin-top:10px;">
    <p style="margin:0 0 8px 0;"><b>🎭 梦境类型：</b>{meta.get('dream_type', '未知')}</p>
    <p style="margin:0 0 8px 0;"><b>⏱️ 生成耗时：</b>{meta.get('generation_time_sec', 0)}s</p>
    <p style="margin:0 0 8px 0;"><b>📐 迭代次数：</b>{cfg.get('ITER_PER_OCTAVE', 'N/A')}次/Octave</p>
    <p style="margin:0 0 8px 0;"><b>🌀 步长强度：</b>{step_size}（{intensity}）</p>
    <p style="margin:0 0 8px 0;"><b>🧠 特征层：</b><code>{layers_str}</code></p>
</div>
"""
