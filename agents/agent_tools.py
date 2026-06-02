from dataclasses import dataclass
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Callable, Dict, List

import holidays
import pandas as pd
from meteostat import Daily, Hourly, Point

from utils import FileCache


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str]
    handler: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": self.required,
                },
            },
        }

    def call(self, **kwargs):
        return self.handler(**kwargs)


def classify_weather(temp, precip, wind_speed):
    if pd.isna(temp) or pd.isna(precip) or pd.isna(wind_speed):
        return "Unknown"
    if (temp < 0 or temp > 35) and precip > 15 and wind_speed > 25:
        return "Severe"
    if precip > 25:
        return "Heavy_Rain"
    elif precip > 10:
        return "Moderate_Rain"
    elif precip > 2:
        return "Light_Rain"
    if wind_speed > 30:
        return "Windy"
    if temp > 30:
        return "Hot"
    elif temp < 5:
        return "Cold"
    elif 15 <= temp <= 25 and precip <= 1 and wind_speed <= 15:
        return "Pleasant"
    return "Mild"


def get_weather_info_day(latitude, longitude, date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        dt = datetime(dt.year, dt.month, dt.day)
        latitude = round(float(latitude), 4)
        longitude = round(float(longitude), 4)
        location = Point(latitude, longitude, 10)
        data = Daily(location, dt, dt).fetch()
        if data.empty:
            return None

        weather = data.iloc[0].to_dict()
        temp = weather.get('tavg', None)
        precip = weather.get('prcp', None)
        wind_speed = weather.get('wspd', None)
        return classify_weather(temp, precip, wind_speed)
    except Exception as error:
        print(f"Daily weather lookup failed: {error}")
        print(f"Latitude: {latitude}, Longitude: {longitude}, Date: {date_str}")
        return None


def weather_search(latitude, longitude, date_str):
    try:
        weather_cache = FileCache('weather_cache.json')
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        dt = datetime(dt.year, dt.month, dt.day, dt.hour)
        latitude = round(float(latitude), 4)
        longitude = round(float(longitude), 4)
        location = Point(latitude, longitude, 10)
        weather_key = f"{latitude},{longitude},{dt}"
        temp, precip, wind_speed, weather_code = None, None, None, None

        weather_val = weather_cache.get(weather_key)
        if weather_val is not None:
            temp, precip, wind_speed, weather_code = weather_val

        if temp is None or precip is None or wind_speed is None or weather_code is None:
            data = Hourly(location, dt, dt)
            data = data.fetch()
            if not data.empty:
                weather = data.iloc[0].to_dict()
                temp = weather.get('temp', None)
                precip = weather.get('prcp', None)
                wind_speed = weather.get('wspd', None)
                if 'coco' in data.columns and not data['coco'].empty:
                    coco_val = data['coco'].iloc[0]
                    if pd.isna(coco_val):
                        weather_code = 999
                    else:
                        weather_code = int(coco_val)
                else:
                    weather_code = 999
            else:
                weather_code = 999

            if temp is not None or precip is not None or wind_speed is not None:
                weather_cache.set(weather_key, [temp, precip, wind_speed, weather_code])
                weather_cache.save()

        weather_type = classify_weather(temp, precip, wind_speed)
        if weather_type != "Unknown":
            return weather_type

        return get_weather_info_day(latitude, longitude, date_str) or "Unknown"
    except Exception as error:
        print(f"Weather lookup failed: {error}")
        print(f"Latitude: {latitude}, Longitude: {longitude}, Date: {date_str}")
        return None


def date_search(time_str, datasetName):
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    date = datetime(dt.year, dt.month, dt.day)
    country_code = "US" if datasetName == "nyc" or datasetName == "ca" else "JP" if datasetName == "tky" else None
    subdiv = "NY" if datasetName == "nyc" else "CA" if datasetName == "ca" else None
    year = date.year
    try:
        country_holidays = holidays.country_holidays(country_code, years=[year], subdiv=subdiv)
        if date.date() in country_holidays:
            holiday_name = country_holidays.get(date.date())
            return f"Today is a holiday: {holiday_name}"

        if date.weekday() >= 5:
            return f"Today is neither a holiday but a weekend: {date.strftime('%A')}"
        return f"Today is a working day, neither a holiday nor a weekend: {date.strftime('%A')}"
    except Exception as error:
        print(f"Holiday lookup failed: {error}")
        if date.weekday() >= 5:
            return f"Today is not a holiday but a weekend: {date.strftime('%A')}"
        return f"Today is not a holiday and a weekday: {date.strftime('%A')}"


def dis_calculate(lat1, lon1, lat2, lon2):
    lat1 = eval(lat1)
    lon1 = eval(lon1)
    lat2 = eval(lat2)
    lon2 = eval(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance


DEFAULT_AGENT_TOOLS: Dict[str, AgentTool] = {
    "WeatherSearch": AgentTool(
        name="WeatherSearch",
        description="Query weather type by latitude/longitude/time, preferring hourly data and falling back to daily data.",
        properties={
            "latitude": {"type": "number", "description": "Latitude"},
            "longitude": {"type": "number", "description": "Longitude"},
            "date_str": {"type": "string", "description": "Timestamp in format YYYY-MM-DD HH:MM:SS"},
        },
        required=["latitude", "longitude", "date_str"],
        handler=weather_search,
    ),
    "DateSearch": AgentTool(
        name="DateSearch",
        description="Query holiday/weekend/weekday information for a given timestamp and dataset.",
        properties={
            "time_str": {"type": "string", "description": "Timestamp in format YYYY-MM-DD HH:MM:SS"},
            "datasetName": {"type": "string", "description": "Dataset name, e.g., nyc, tky, ca"},
        },
        required=["time_str", "datasetName"],
        handler=date_search,
    ),
    "DisCalculate": AgentTool(
        name="DisCalculate",
        description="Compute great-circle distance between two coordinates in kilometers.",
        properties={
            "lat1": {"type": "number", "description": "Start latitude"},
            "lon1": {"type": "number", "description": "Start longitude"},
            "lat2": {"type": "number", "description": "End latitude"},
            "lon2": {"type": "number", "description": "End longitude"},
        },
        required=["lat1", "lon1", "lat2", "lon2"],
        handler=dis_calculate,
    ),
}


def get_tool_schemas() -> List[dict]:
    return [tool.schema() for tool in DEFAULT_AGENT_TOOLS.values()]


def invoke_tool(tool_name: str, **kwargs):
    if tool_name not in DEFAULT_AGENT_TOOLS:
        raise KeyError(f"Unknown agent tool: {tool_name}")
    return DEFAULT_AGENT_TOOLS[tool_name].call(**kwargs)