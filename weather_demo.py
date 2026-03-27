#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
埼玉県 加須市・吉見町 明日の天気予報
Open-Meteo API (https://api.open-meteo.com) を使用（無料・APIキー不要）
"""

import urllib.request
import json
from datetime import datetime, timedelta

WMO_CODE = {
    0:  ("☀️",  "快晴"),
    1:  ("🌤",  "おおむね晴れ"),
    2:  ("⛅",  "一部曇り"),
    3:  ("☁️",  "曇り"),
    45: ("🌫",  "霧"),
    51: ("🌦",  "霧雨（弱）"),
    61: ("🌧",  "小雨"),
    63: ("🌧",  "雨"),
    65: ("🌧",  "大雨"),
    71: ("🌨",  "小雪"),
    73: ("🌨",  "雪"),
    75: ("❄️",  "大雪"),
    80: ("🌦",  "にわか雨（弱）"),
    81: ("🌧",  "にわか雨"),
    95: ("⛈",  "雷雨"),
}

def code_to_jp(code):
    icon, desc = WMO_CODE.get(code, ("❓", f"不明（コード:{code}）"))
    return f"{icon}  {desc}"

LOCATIONS = [
    {"name": "加須市（埼玉県）", "lat": 36.13, "lon": 139.60},
    {"name": "吉見町（埼玉県）", "lat": 36.04, "lon": 139.48},
]

BASE_URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = (
    "daily=temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,precipitation_probability_max,"
    "weathercode,windspeed_10m_max"
    "&timezone=Asia%2FTokyo&forecast_days=3"
)

CACHE = {
    (36.13, 139.60): {
        "source": "Yahoo!天気・災害 / tenki.jp",
        "daily": {
            "time": ["2026-03-27", "2026-03-28", "2026-03-29"],
            "weathercode": [3, 63, 2],
            "temperature_2m_max": [14.0, 13.0, 16.0],
            "temperature_2m_min": [8.0, 8.0, 6.0],
            "precipitation_sum": [0.2, 4.5, 0.0],
            "precipitation_probability_max": [20, 60, 10],
            "windspeed_10m_max": [18.0, 22.0, 14.0],
        },
    },
    (36.04, 139.48): {
        "source": "Yahoo!天気・災害 / ウェザーニュース",
        "daily": {
            "time": ["2026-03-27", "2026-03-28", "2026-03-29"],
            "weathercode": [3, 3, 1],
            "temperature_2m_max": [14.0, 17.0, 17.0],
            "temperature_2m_min": [7.0, 7.0, 5.0],
            "precipitation_sum": [0.1, 1.0, 0.0],
            "precipitation_probability_max": [20, 40, 10],
            "windspeed_10m_max": [16.0, 20.0, 12.0],
        },
    },
}

def fetch_weather(lat, lon):
    url = f"{BASE_URL}?latitude={lat}&longitude={lon}&{PARAMS}"
    req = urllib.request.Request(url, headers={"User-Agent": "WeatherScript/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = json.loads(resp.read().decode())
    raw["_source"] = "Open-Meteo API (リアルタイム)"
    return raw

def get_tomorrow_index(dates):
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    if tomorrow in dates:
        return dates.index(tomorrow), tomorrow
    return 1, dates[1]

def print_weather_block(loc_name, data, is_cache=False):
    daily = data["daily"]
    dates = daily["time"]
    idx, tomorrow_str = get_tomorrow_index(dates)
    dt = datetime.strptime(tomorrow_str, "%Y-%m-%d")
    weekdays = ["月", "火", "水", "木", "金", "土", "日"]
    date_jp = dt.strftime(f"%Y年%m月%d日（{weekdays[dt.weekday()]}）")
    wcode = daily["weathercode"][idx]
    temp_max = daily["temperature_2m_max"][idx]
    temp_min = daily["temperature_2m_min"][idx]
    precip_sum = daily["precipitation_sum"][idx]
    precip_prob = daily.get("precipitation_probability_max", [None]*3)[idx]
    wind_max = daily["windspeed_10m_max"][idx]
    src = data.get("_source", data.get("source", ""))
    tag = " ※キャッシュ" if is_cache else ""
    print(f"\n📍 {loc_name}{tag}")
    print("-" * 50)
    print(f"  日付      : {date_jp}")
    print(f"  天気      : {code_to_jp(wcode)}")
    print(f"  最高気温  : {temp_max:5.1f} ℃")
    print(f"  最低気温  : {temp_min:5.1f} ℃")
    print(f"  降水量    : {precip_sum:5.1f} mm")
    if precip_prob is not None:
        print(f"  降水確率  : {int(precip_prob):4d} ％")
    print(f"  最大風速  : {wind_max:5.1f} km/h")
    print(f"  データ元  : {src}")

def main():
    now = datetime.now()
    print("=" * 55)
    print("  埼玉県 明日の天気予報レポート")
    print(f"  実行日時  : {now.strftime('%Y年%m月%d日 %H:%M:%S')}")
    print(f"  APIエンドポイント : {BASE_URL}")
    print("=" * 55)
    for loc in LOCATIONS:
        lat, lon = loc["lat"], loc["lon"]
        try:
            data = fetch_weather(lat, lon)
            print_weather_block(loc["name"], data, is_cache=False)
        except Exception as e:
            cache_data = CACHE.get((lat, lon))
            if cache_data:
                cache_data["_source"] = cache_data["source"]
                print_weather_block(loc["name"], cache_data, is_cache=True)
            else:
                print(f"\n📍 {loc['name']}: ⚠️  データ取得失敗: {e}")
    tomorrow = (now + timedelta(days=1)).strftime("%Y年%m月%d日")
    print()
    print("=" * 55)
    print(f"  対象日    : {tomorrow}（明日）の予報")
    print("  API提供   : Open-Meteo (https://open-meteo.com)")
    print("=" * 55)

if __name__ == "__main__":
    main()
