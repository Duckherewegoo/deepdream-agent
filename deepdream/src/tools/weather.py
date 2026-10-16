"""天气查询工具"""
import json
from langchain_core.tools import tool
import time
import jwt
from pathlib import Path
import requests

pkey = Path('/home/dake/.ssh/ed25519-private.pem')
api_host = Path('./api_host')

# Open PEM
private_key = pkey.read_text()
hosts = api_host.read_text()

key_payload = {
    'iat': int(time.time()) - 30,
    'exp': int(time.time()) + 900,
    'sub': '442J4AKN6X'
}
key_headers = {
    'kid': 'CJWETHVCVV'
}

# Generate JWT
encoded_jwt = jwt.encode(key_payload, private_key,
                         algorithm='EdDSA', headers=key_headers)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。

    Args:
        city: 要查询的城市名称，如"北京"、"上海"等

    Returns:
        天气信息的 JSON 字符串
    """
    global encoded_jwt, hosts
    geo_url = f"https://{hosts}/geo/v2/city/lookup?location={city}"
    geo_headers = {
        "Authorization": "Bearer {encoded_jwt}",
        "Accept-Encoding": "gzip"
    }
    g_res = requests.get(geo_url, geo_headers)
    g_data = g_res.json()
    location = g_data.get("location")
    city_id = location.get('id')

    weather_url = f"https://{hosts}/v7/weather/now?location={city_id}"
    weather_headers = {
        "Authorization": "Bearer {encoded_jwt}",
        "Accept-Encoding": "gzip"  # 对应 curl 的 --compressed
    }
    w_res = requests.get(weather_url, weather_headers)
    w_data = w_res.json()
    now = w_data.get("now", {})
    info = f"{now.get('text')}，气温{now.get('temp')}摄氏度"
    return json.dumps({
        "city": city,
        "info": info,
        "tip": f"当前{city}的天气是{info.split('，')[0]}，适合做个好梦！"
    }, ensure_ascii=False)
