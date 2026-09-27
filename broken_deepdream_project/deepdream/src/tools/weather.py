"""天气查询工具"""
import json
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。

    Args:
        city: 要查询的城市名称，如"北京"、"上海"等

    Returns:
        天气信息的 JSON 字符串
    """
    # 模拟天气数据（实际项目中可接入真实天气 API）
    weather_data = {
        "北京": "晴 15-25°C，适合做梦的好天气！",
        "上海": "小雨 18-22°C，梦里听雨声~",
        "广州": "晴 25-32°C，热带梦境！",
        "深圳": "多云 24-30°C，科技感梦境！",
        "杭州": "阴 16-23°C，江南水乡梦境~",
        "成都": "小雨 17-24°C，火锅味的梦境！",
        "齐齐哈尔": "晴 12-22°C，东北清爽梦境！",
    }

    info = weather_data.get(city, f"{city}：天气数据获取中，先做个美梦吧~")

    return json.dumps({
        "city": city,
        "info": info,
        "tip": f"当前{city}的天气是{info.split('，')[0]}，适合做个好梦！"
    }, ensure_ascii=False)
