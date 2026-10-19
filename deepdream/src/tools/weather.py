"""天气查询工具 - 基于和风天气 API"""
import json
import logging
import os
import time
from pathlib import Path

import jwt
import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ================= 配置加载 =================
# 严令禁止硬编码默认凭证，必须走环境变量
_API_KEY_ID = os.getenv("QWEATHER_KID", "")
_SUBJECT = os.getenv("QWEATHER_SUB", "")

def _load_private_key() -> str | None:
    """加载 JWT 私钥"""
    env_key = os.getenv("QWEATHER_PRIVATE_KEY")
    if env_key:
        return env_key

    local_path = Path("./keys/ed25519-private.pem")
    if local_path.exists():
        return local_path.read_text()

    legacy_path = Path.home() / ".ssh" / "ed25519-private.pem"
    if legacy_path.exists():
        return legacy_path.read_text()

    logger.warning("[WEATHER] 未找到和风天气私钥，天气工具将不可用")
    return None

def _load_api_host() -> str:
    """加载 API Host"""
    env_host = os.getenv("QWEATHER_HOST")
    if env_host:
        return env_host.strip()

    local_path = Path("./api_host")
    if local_path.exists():
        return local_path.read_text().strip()

    return "devapi.qweather.com"

_PRIVATE_KEY = _load_private_key()
_API_HOST = _load_api_host()

def _generate_jwt() -> str:
    """生成最新的 JWT（有效期 15 分钟）"""
    if not _PRIVATE_KEY:
        raise RuntimeError("缺少私钥，无法生成 JWT")
    # 增加对 KID 和 SUB 的校验
    if not _API_KEY_ID or not _SUBJECT:
        raise RuntimeError("缺少 QWEATHER_KID 或 QWEATHER_SUB 环境变量")

    now = int(time.time())
    key_payload = {
        "iat": now - 30,
        "exp": now + 900,
        "sub": _SUBJECT,
    }
    key_headers = {"kid": _API_KEY_ID}
    return jwt.encode(key_payload, _PRIVATE_KEY, algorithm="EdDSA", headers=key_headers)

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息。"""
    try:
        encoded_jwt = _generate_jwt()
        headers = {
            "Authorization": f"Bearer {encoded_jwt}",
            "Accept-Encoding": "gzip",
        }

        geo_url = f"https://{_API_HOST}/geo/v2/city/lookup"
        g_res = requests.get(geo_url, params={"location": city}, headers=headers, timeout=10)
        g_data = g_res.json()

        location_list = g_data.get("location")
        if not location_list:
            return json.dumps({"success": False, "msg": f"未找到城市: {city}"}, ensure_ascii=False)

        city_id = location_list[0].get("id")
        if not city_id:
            return json.dumps({"success": False, "msg": f"城市 {city} 缺少 ID 信息"}, ensure_ascii=False)

        weather_url = f"https://{_API_HOST}/v7/weather/now"
        w_res = requests.get(weather_url, params={"location": city_id}, headers=headers, timeout=10)
        w_data = w_res.json()

        now = w_data.get("now", {})
        if not now:
            return json.dumps({"success": False, "msg": f"未能获取 {city} 的天气数据"}, ensure_ascii=False)

        info = f"{now.get('text', '未知')}，气温{now.get('temp', '?')}摄氏度"
        return json.dumps({
            "success": True,
            "city": city,
            "info": info,
            "tip": f"当前{city}的天气是{info.split('，')[0]}，适合做个好梦！",
        }, ensure_ascii=False)

    except requests.RequestException as e:
        logger.exception(f"[WEATHER] 网络请求失败: {e}")
        return json.dumps({"success": False, "msg": f"网络请求异常: {e}"}, ensure_ascii=False)
    except RuntimeError as e:
        logger.exception(f"[WEATHER] 配置缺失: {e}")
        return json.dumps({"success": False, "msg": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.exception(f"[WEATHER] 未知异常: {e}")
        return json.dumps({"success": False, "msg": "天气查询出现内部错误"}, ensure_ascii=False)
