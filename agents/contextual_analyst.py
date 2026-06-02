import json
from .base_agent import BaseAgent
from utils import openaiAPIcall, parse_llm_response_with_scores_robust, get_date_info, LLM_NAME

class ContextualAnalyst(BaseAgent):
    """
    Contextual analyst focused on the user's current context.
    """
    def __init__(self, mapoi_instance):
        super().__init__(mapoi_instance)
        self.agent_name = "contextual_analyst"
    
    def recommend(self, data):
        """Simplified recommendation implementation for pretraining."""
        candidates = data["candidates"]
        
        # Simplest baseline: keep the existing distance-sorted order.
        sorted_pois = [poi for poi, _, _ in candidates]
        
        return sorted_pois[:20]  # Return top 20.

    def generate_recommendation(self, data, candidate_pois, if_profile=True):
        """Full LLM-based contextual recommendation for production usage."""
        knowledge = self.get_shared_data().get('knowledge', {}) or {}
        profile = data["profile"]
        candidates = data["candidates"]

        shared_data = self.get_shared_data()
        current_poi = data["current_poi"]
        current_time = data["current_time"]
        poiInfos = shared_data['poiInfos']
        current_lat = poiInfos[current_poi]["latitude"]
        current_lon = poiInfos[current_poi]["longitude"]
        current_weather = self.call_tool(
            "WeatherSearch",
            latitude=current_lat,
            longitude=current_lon,
            date_str=current_time,
        )
        holiday_info = self.call_tool(
            "DateSearch",
            time_str=current_time,
            datasetName=shared_data['datasetName'],
        )
        date_type = get_date_info(current_time, shared_data['datasetName'])

        profile_str = json.dumps(profile, indent=2)
        profile_str = profile_str.replace('\\n', '\n')

        # Build dynamic priors (Top-K to avoid overlong prompts).
        def top_k_dict(d: dict, k=5):
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
            return {k_: float(v) for k_, v in items}

        weather_prior_full = (knowledge.get("weather_category_distribution", {}) or {}).get(current_weather, {}) or {}
        date_prior_full = (knowledge.get("date_type_category_distribution", {}) or {}).get(date_type, {}) or {}
        weather_prior_top = top_k_dict(weather_prior_full, k=5)
        date_prior_top = top_k_dict(date_prior_full, k=5)

        # Convert weather categories to readable text, e.g., "Sunny suitable for home".
        weather_prior_text = ", ".join([f"{k}" for k, v in weather_prior_top.items()])
        date_prior_text = ", ".join([f"{k}" for k, v in date_prior_top.items()])
        if if_profile:
            context_prompt = f"""
You are a Contextual Behavior Analyst specializing in external context analysis for POI recommendations.
Your task is to adjust the ranking of POIs in the candidate set based on the user's temporal preferences.
Your goal is to slightly adjust the existing order to reflect temporal suitability, while keeping the overall order as stable as possible (minimal change principle).

WEATHER INFORMATION:
Current weather: {current_weather}
Weather-based Category Priors: {current_weather} is suitable for {weather_prior_text}
DATE INFORMATION:
Current date type: {date_type}
Date-based Category Priors: {date_type} is suitable for {date_prior_text}

HOLIDAY INFORMATION: 
{holiday_info}

<candidate set> [Format: (POIID, Category)]: {candidates}

Analysis Steps:
1. Analyze the environmental context:
- Identify weather-related preferences (e.g., indoor activities preferred on rainy, snowy, stormy, or extremely hot/cold days; outdoor or scenic POIs preferred on mild or sunny days).
- Identify holiday-related preferences (e.g., shopping, leisure, entertainment during holidays; regular routines on workdays).
2. Evaluate each candidate POI's contextual suitability:
- HIGH match: POI is highly suitable for the current weather/holiday.
- MODERATE match: POI is somewhat suitable for current context.
- UNSUITABLE: POI is clearly inappropriate under extreme weather (e.g., outdoor park or mountain during heavy rain, snow, storm, or high heat).
3. Adjust the ranking:
- Move HIGH matches forward by 2 positions, MODERATE matches forward by 1.
- Move UNSUITABLE POIs backward by 3 positions or to the end of the list.
4. Do NOT delete or invent any POIs. All POIIDs must come from the candidate set.

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority.
2. "reason": A brief explanation of your reasoning.
"""
        else:
            context_prompt = f"""
You are a Contextual Behavior Analyst specializing in external context analysis for POI recommendations.
Your task is to adjust the ranking of POIs in the candidate set based on the user's temporal preferences.
Your goal is to slightly adjust the existing order to reflect temporal suitability, while keeping the overall order as stable as possible (minimal change principle).

WEATHER INFORMATION:
Current weather: {current_weather}
DATE INFORMATION:
Current date type: {date_type}

HOLIDAY INFORMATION: 
{holiday_info}

<candidate set> [Format: (POIID, Category)]: {candidates}

Analysis Steps:
1. Analyze the environmental context:
- Identify weather-related preferences (e.g., indoor activities preferred on rainy, snowy, stormy, or extremely hot/cold days; outdoor or scenic POIs preferred on mild or sunny days).
- Identify holiday-related preferences (e.g., shopping, leisure, entertainment during holidays; regular routines on workdays).
2. Evaluate each candidate POI's contextual suitability:
- HIGH match: POI is highly suitable for the current weather/holiday.
- MODERATE match: POI is somewhat suitable for current context.
- UNSUITABLE: POI is clearly inappropriate under extreme weather (e.g., outdoor park or mountain during heavy rain, snow, storm, or high heat).
3. Adjust the ranking:
- Move HIGH matches forward by 2 positions, MODERATE matches forward by 1.
- Move UNSUITABLE POIs backward by 3 positions or to the end of the list.
4. Do NOT delete or invent any POIs. All POIIDs must come from the candidate set.

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority.
2. "reason": A brief explanation of your reasoning.
"""

        messages = [{"role": "user", "content": context_prompt}]
        # print(f"Contextual analyst prompt: {context_prompt}")
        response = openaiAPIcall(
            model = LLM_NAME,
            messages=messages,
            temperature=0,
            top_p=0.95,
            presence_penalty=0,
            frequency_penalty=0,
            extra_body={"top_k": 50}
        )
        response_content = response.choices[0].message.content
        
        try:
            # print(f"Contextual analyst response content: {response_content}")
            response = parse_llm_response_with_scores_robust(response_content)
            response["prompt"] = context_prompt  # Attach prompt to response.
        except Exception as e:
            print(f"Contextual analyst parse error: {e}")
            # Fallback to simplified recommendation logic.
            default_recs = self.recommend(data)
            default_recs = default_recs[:20]
            response = {
                "prompt": context_prompt,
                "recommendation": default_recs,
                "reason": "Default distance-based fallback recommendation.",
                "confidence": 0.5
            }
        
        candidates = response["recommendation"]
        candidates = [poi for poi in candidates if poi in candidate_pois]
        
        return response, candidates