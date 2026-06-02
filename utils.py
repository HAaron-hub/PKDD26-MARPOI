import re
import json 
import os
from datetime import datetime
import threading
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import holidays
from typing import Any, Dict, List, Optional

LLM_NAME = 'models/qwen/Qwen2.5-7B-Instruct'
# 'models/qwen/Qwen2.5-7B-Instruct'
# 'models/gemma/LLM-Research/gemma-2-9b-it'
# 'models/mistral/mistralai/Mistral-7B-Instruct-v0.3'
# 'models/llama/LLM-Research/Meta-Llama-3.1-8B-Instruct'
# 'models/qwen/Qwen2.5-3B-Instruct'

# Create OpenAI clients.
clients = [
    # OpenAI(base_url="http://localhost:8020/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8021/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8022/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8023/v1", api_key="not-needed", timeout=60.0),## 4090*8
    OpenAI(base_url="http://localhost:8024/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8025/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8026/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8027/v1", api_key="not-needed", timeout=60.0),## 3090
    OpenAI(base_url="http://localhost:8030/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8031/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8032/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8033/v1", api_key="not-needed", timeout=60.0),## 4090*4
    # OpenAI(base_url="http://localhost:8034/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8035/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8036/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8037/v1", api_key="not-needed", timeout=60.0),## 5880*4 47
    OpenAI(base_url="http://localhost:8040/v1", api_key="not-needed", timeout=60.0),
    OpenAI(base_url="http://localhost:8041/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8042/v1", api_key="not-needed", timeout=60.0),
    # OpenAI(base_url="http://localhost:8043/v1", api_key="not-needed", timeout=60.0),## 5880*4 48
]

# Round-robin client counter.
client_counter = 0
client_lock = threading.Lock()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(2))
def openaiAPIcall(**kwargs):
    """Thread-safe OpenAI API call with retry support."""
    global client_counter
    
    # Select a client in a thread-safe way.
    with client_lock:
        client_index = client_counter % len(clients)
        client_counter += 1
    
    client = clients[client_index]
    return client.chat.completions.create(**kwargs)


def time2period(time_str):
    """Convert a time string to a period label."""
    try:
        dt = datetime.strptime(time_str.strip(), '%Y-%m-%d %H:%M:%S')
        hour = dt.hour
        periods = [
            (5, 8, "Early Morning (05:00-08:00)"),
            (8, 11, "Morning (08:00-11:00)"),
            (11, 14, "Noon (11:00-14:00)"),
            (14, 18, "Afternoon (14:00-18:00)"),
            (18, 20, "Evening (18:00-20:00)"),
            (20, 23, "Night (20:00-23:00)"),
            (23, 24, "Midnight (23:00-05:00)"),
            (0, 5, "Midnight (23:00-05:00)"),
        ]
        for start, end, period_name in periods:
            if start <= hour < end:
                return period_name
    except Exception as e:
        print(f"Time[period] formatting error: {e}")
        return time_str
    
def time2day(time_str):
    """Convert a time string to day-of-week."""
    try:
        dt = datetime.strptime(time_str.strip(), '%Y-%m-%d %H:%M:%S')
        weekday = dt.strftime('%A')
        return f"{weekday}"
    except Exception as e:
        print(f"Time[weekday] formatting error: {e}")
        return time_str

def get_date_info(time_str, datasetName):
    """Determine whether a date is a weekday, weekend, or holiday."""
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    date = datetime(dt.year, dt.month, dt.day)
    country_code = "US" if datasetName == "nyc" or datasetName == "ca" else "JP" if datasetName == "tky" else None
    subdiv = "NY" if datasetName == "nyc" else "CA" if datasetName == "ca" else None
    # Get year.
    year = date.year
    
    # Get national/regional holidays.
    try:
        country_holidays = holidays.country_holidays(country_code, years=[year], subdiv=subdiv)
        
        # Check whether it is a holiday.
        if date.date() in country_holidays:
            return "Holiday"
        
        # Check whether it is weekend (0=Mon, 6=Sun).
        if date.weekday() >= 5:  # 5=Sat, 6=Sun
            return "Weekend"
        else:
            return "Weekday"
            
    except Exception as e:
        print(f"Failed to fetch holiday info: {e}")
        # Fallback to simple weekend check.
        if date.weekday() >= 5:
            return "Weekend"
        else:
            return "Weekday"

def eval_js_math(expr):
    """
    Support simple addition/division and Math.max.
    """
    expr = expr.replace('Math.max', '')
    nums = re.findall(r'[\d.]+', expr)
    if '/' in expr and '+' in expr:
        total = sum(float(x) for x in nums[:-1])
        divisor = float(nums[-1])
        return total / divisor if divisor else 0.0
    elif len(nums) > 0:
        return max(float(x) for x in nums)
    return None

def preprocess_json(json_str):
    # Remove // line comments (keep JSON content only).
    cleaned_lines = []
    for line in json_str.splitlines():
        if '//' in line:
            quote_count = line[:line.find('//')].count('"')
            if quote_count % 2 == 0:
                line = line[:line.find('//')]
        cleaned_lines.append(line.rstrip())
    json_str = '\n'.join(cleaned_lines)

    # Auto-insert commas between adjacent key-value lines when missing.
    json_str = re.sub(r'(":[^,\{\}\[\]\n]+)(\n\s*"[\w_]+":)', r'\1,\2', json_str)
    # Patch the last missing comma case.
    json_str = re.sub(r'(":[^,\{\}\[\]\n]+)(\n\s*\})', r'\1\2', json_str)

    # Replace expressions in travel_radius fields.
    def average_replace(match):
        expr = match.group(1)
        val = eval_js_math(expr)
        return f'"average_km": {val:.8f},' if val is not None else '"average_km": null,'
    json_str = re.sub(r'"average_km":\s*\((.*?)\),', lambda m: average_replace(m), json_str)
    json_str = re.sub(r'"average_km":\s*\((.*?)\)', lambda m: average_replace(m), json_str)
    def max_replace(match):
        expr = match.group(1)
        val = eval_js_math(expr)
        return f'"max_km": {val:.8f}' if val is not None else '"max_km": null'
    json_str = re.sub(r'"max_km":\s*Math\.max\((.*?)\)', lambda m: max_replace(m), json_str)

    # Remove trailing commas in arrays/objects.
    json_str = re.sub(r',\s*\]', ']', json_str)
    json_str = re.sub(r',\s*\}', '}', json_str)

    # Remove empty elements or comments in arrays.
    json_str = re.sub(r'\[\s*("[^"]*")\s*,\s*\]', r'[\1]', json_str)

    # Remove comments in empty objects/arrays.
    json_str = re.sub(r'([,\[])\s*//[^\]\}]*', r'\1', json_str)

    return json_str

def parse_profile(profile_content):
    # Extract JSON block.
    match = re.search(r"```json\s*([\s\S]*?)\s*```", profile_content)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = profile_content.strip()

    # Preprocess.
    json_str = preprocess_json(json_str)

    # Load JSON.
    profile = json.loads(json_str)
    
    return profile

def parse_llm_response_with_scores_robust_conf(res_content: str) -> Dict[str, Any]:
    """
    Robustly parse LLM-generated JSON or pseudo-JSON content,
    extracting recommendation and reason fields.
    """
    def clean_json_like(text: str) -> str:
        # Remove comments, duplicate fields, and extra braces.
        text = re.sub(r'//.*', '', text)  # Remove inline comments.
        text = re.sub(r'/\*[\s\S]*?\*/', '', text)  # Remove block comments.
        text = re.sub(r',\s*([\]}])', r'\1', text)  # Remove trailing commas.
        # Keep only the first duplicated "recommendation" field.
        text = re.sub(r'("recommendation"\s*:\s*\[.*?\])[\s,]*("recommendation"\s*:\s*\[)', r'\1,', text, flags=re.DOTALL)
        # Fix extra braces.
        text = re.sub(r'^\s*{+\s*', '{', text)
        text = re.sub(r'\s*}+\s*$', '}', text)
        return text

    def extract_json_block(text: str) -> str:
        # Prefer markdown code block content.
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Or use the first brace-wrapped content.
        match = re.search(r"({[\s\S]*})", text)
        if match:
            return match.group(1).strip()
        return text

    def try_load_json(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            return {}

    def extract_recommendation(text: str) -> List[str]:
        # Match recommendation: [ ... ]
        match = re.search(r'"recommendation"\s*:\s*\[([\s\S]*?)\]', text)
        if match:
            arr_raw = match.group(1)
            # Match "xxx", 'xxx', or pure digits.
            items = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\d+)', arr_raw)
            # items is a tuple; take the first non-empty element.
            result = [next(filter(None, tup)) for tup in items]
            return result
        return []

    def extract_reason(text: str) -> str:
        # Match reason field, including multiline values.
        match = re.search(r'"reason"\s*:\s*"([\s\S]*?)"', text)
        if match:
            return match.group(1).strip()
        # Fallback when reason appears before final closing brace.
        match = re.search(r'"reason"\s*:\s*([\s\S]*?)(?:,?\s*[\}\]])', text)
        if match:
            return match.group(1).strip().strip('"')
        return ""

    def normalize_confidence(val: Any) -> Optional[float]:
        """
        Convert input to a float in [0, 1].
        - Supports numeric values, "0.85", "85%", and similar formats.
        - Out-of-range values are clamped to [0, 1].
        """
        if val is None:
            return None
        num = float(val)
       
        # Clamp to [0, 1].
        if num < 0.0:
            num = 0.0
        if num > 1.0:
            num = 1.0
        return num
    
    def extract_confidence(text: str) -> Optional[float]:
        # Capture value after "confidence" until comma/newline/closing brace.
        m = re.search(r'"confidence"\s*:\s*([^\n,}\]]+)', text, re.IGNORECASE)
        if not m:
            return None
        raw = m.group(1).strip()
        # Strip wrapping quotes.
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            raw = raw[1:-1].strip()
        return normalize_confidence(raw)
    
    # === Main flow ===
    block = extract_json_block(res_content)
    cleaned = clean_json_like(block)
    # Try JSON parsing first.
    data = try_load_json(cleaned)
    if isinstance(data, dict) and {"recommendation", "reason", "confidence"} & set(data.keys()):
        recommendation = data.get("recommendation", [])
        # If recommendation contains commented strings, clean again.
        if isinstance(recommendation, list):
            recommendation = [
                str(x).split("//")[0].strip().strip('"').strip("'") for x in recommendation
            ]
        reason = data.get("reason", "")
        confidence_val = normalize_confidence(data.get("confidence"))
        # If missing/unparseable, default to 0.0.
        if confidence_val is None:
            confidence_val = 0.0
        return {
            "recommendation": recommendation,
            "reason": reason,
            "confidence": confidence_val,
        }
    # Otherwise, fall back to regex extraction.
    recommendation = extract_recommendation(cleaned)
    reason = extract_reason(cleaned)
    confidence_val = extract_confidence(cleaned)
    if confidence_val is None:
        confidence_val = 0.0

    return {
        "recommendation": recommendation,
        "reason": reason,
        "confidence": confidence_val,
    }


def parse_llm_response_with_scores_robust(res_content: str) -> Dict[str, Any]:
    """
    Robustly parse LLM-generated JSON or pseudo-JSON content,
    extracting recommendation and reason fields.
    """
    def clean_json_like(text: str) -> str:
        # Remove comments, duplicate fields, and extra braces.
        text = re.sub(r'//.*', '', text)  # Remove inline comments.
        text = re.sub(r'/\*[\s\S]*?\*/', '', text)  # Remove block comments.
        text = re.sub(r',\s*([\]}])', r'\1', text)  # Remove trailing commas.
        # Keep only the first duplicated "recommendation" field.
        text = re.sub(r'("recommendation"\s*:\s*\[.*?\])[\s,]*("recommendation"\s*:\s*\[)', r'\1,', text, flags=re.DOTALL)
        # Fix extra braces.
        text = re.sub(r'^\s*{+\s*', '{', text)
        text = re.sub(r'\s*}+\s*$', '}', text)
        return text

    def extract_json_block(text: str) -> str:
        # Prefer markdown code block content.
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Or use the first brace-wrapped content.
        match = re.search(r"({[\s\S]*})", text)
        if match:
            return match.group(1).strip()
        return text

    def try_load_json(text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            return {}

    def extract_recommendation(text: str) -> List[str]:
        # Match recommendation: [ ... ]
        match = re.search(r'"recommendation"\s*:\s*\[([\s\S]*?)\]', text)
        if match:
            arr_raw = match.group(1)
            # Match "xxx", 'xxx', or pure digits.
            items = re.findall(r'"([^"]+)"|\'([^\']+)\'|(\d+)', arr_raw)
            # items is a tuple; take the first non-empty element.
            result = [next(filter(None, tup)) for tup in items]
            return result
        return []

    def extract_reason(text: str) -> str:
        # Match reason field, including multiline values.
        match = re.search(r'"reason"\s*:\s*"([\s\S]*?)"', text)
        if match:
            return match.group(1).strip()
        # Fallback when reason appears before final closing brace.
        match = re.search(r'"reason"\s*:\s*([\s\S]*?)(?:,?\s*[\}\]])', text)
        if match:
            return match.group(1).strip().strip('"')
        return ""

    # === Main flow ===
    block = extract_json_block(res_content)
    cleaned = clean_json_like(block)
    # Try JSON parsing first.
    data = try_load_json(cleaned)
    if isinstance(data, dict) and ("recommendation" in data or "reason" in data):
        recommendation = data.get("recommendation", [])
        # If recommendation contains commented strings, clean again.
        if isinstance(recommendation, list):
            recommendation = [str(x).split("//")[0].strip().strip('"').strip("'") for x in recommendation]
        reason = data.get("reason", "")
        return {"recommendation": recommendation, "reason": reason}
    # Otherwise, fall back to regex extraction.
    recommendation = extract_recommendation(cleaned)
    reason = extract_reason(cleaned)
    return {
        "recommendation": recommendation,
        "reason": reason
    }

class FileCache:
    def __init__(self, filename):
        self.filename = filename
        self.data = {}
        # Try loading local cache.
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def get(self, key):
        return self.data.get(key, None)
    
    def set(self, key, value):
        self.data[key] = value

    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False)

