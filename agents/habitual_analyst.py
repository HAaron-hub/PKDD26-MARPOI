import json
from .base_agent import BaseAgent
from utils import openaiAPIcall, parse_llm_response_with_scores_robust_conf, parse_llm_response_with_scores_robust, LLM_NAME

class HabitualAnalyst(BaseAgent):
    """
    Habitual analyst focused on historical visit frequency and preferences.
    """
    def __init__(self, mapoi_instance):
        super().__init__(mapoi_instance)
        self.agent_name = "habitual_analyst"
    
    def recommend(self, data):
        """Simplified recommendation implementation for pretraining."""
        candidates = data["candidates"]
        longterm = data["longterm"]
        longterm2 = data["longterm2"]
        
        # Count POI visit frequency.
        poi_frequency = {}
        for poi, _, _, _, _ in longterm:
            if poi not in poi_frequency:
                poi_frequency[poi] = 0
            poi_frequency[poi] += 1
        
        # Count category visit frequency.
        category_frequency = {}
        for _, cat, _, _, _ in longterm:
            if cat not in category_frequency:
                category_frequency[cat] = 0
            category_frequency[cat] += 1
        
        # Compute frequency-based scores.
        freq_scores = {}
        for poi, _, cat in candidates:
            freq_scores[poi] = 0
            
            # 1. Direct historical frequency.
            if poi in poi_frequency:
                freq_scores[poi] += poi_frequency[poi] * 2
                
            # 2. Category preference.
            if cat in category_frequency:
                freq_scores[poi] += category_frequency[cat] * 0.5
        
        # Sort by frequency score.
        sorted_pois = sorted(freq_scores.items(), key=lambda x: x[1], reverse=True)
        return [poi for poi, _ in sorted_pois][:10]  # Return top 10.

    def generate_recommendation(self, data, candidate_pois, if_profile=True):
        """Full LLM-based habitual recommendation for production usage."""
        profile = data["profile"]
        candidates = data["candidates"]
        recent = data["recent"]
        profile_str = json.dumps(profile, indent=2)
        profile_str = profile_str.replace('\\n', '\n')
        if if_profile:
            habit_prompt = f"""
You are the Behavioral Habit Analyst, specializing in analyzing user preferences based on historical visit patterns.
Your task is to recommend a user's next point-of-interest (POI) strictly from the <candidate set> listed below.

USER PROFILE:
{profile_str}

<recent check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}

Requirements:
Generate recommendations based ONLY on habitual preferences factors:
1. Prioritize the user's frequently visited locations
2. Consider the user's favorite categories and establishments
3. Consider recent changes in visitation patterns
4. Balance between familiar favorites and potentially interesting spots

Focus EXCLUSIVELY on HABITUAL PREFERENCES factors - do NOT consider time patterns, or context factors.

IMPORTANT:
You MUST ONLY output POIIDs from the <candidate set>.
If you output any POIID not in this set, your answer is invalid. This is a hard requirement.
Double-check your output: every recommended POIID must be present in the candidate set and not duplicated.
Do NOT invent, hallucinate, or fabricate any POIID. Use only those listed in the candidate set.

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority. Each POIID MUST be from the candidate set above. Do NOT include any POIID that is not listed.
2. "reason": A brief explanation of your reasoning.
3. "confidence": A value between 0 and 1 indicating how confident you are in the recommendation results (1 = highly confident, 0 = not confident).
Before outputting, CHECK that every POIID in your "recommendation" is in the <candidate set>.
If any POIID is outside the candidate set, your answer will be treated as invalid.
"""
        else:
            # Build habitual analyst prompt (without user profile).
            habit_prompt = f"""
You are the Behavioral Habit Analyst, specializing in analyzing user preferences based on historical visit patterns.
Your task is to recommend a user's next point-of-interest (POI) strictly from the <candidate set> listed below.

<recent check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}

Requirements:
Generate recommendations based ONLY on habitual preferences factors:
1. Prioritize the user's frequently visited locations
2. Consider the user's favorite categories and establishments
3. Consider recent changes in visitation patterns
4. Balance between familiar favorites and potentially interesting spots

Focus EXCLUSIVELY on HABITUAL PREFERENCES factors - do NOT consider time patterns, or context factors.

IMPORTANT:
You MUST ONLY output POIIDs from the <candidate set>.
If you output any POIID not in this set, your answer is invalid. This is a hard requirement.
Double-check your output: every recommended POIID must be present in the candidate set and not duplicated.
Do NOT invent, hallucinate, or fabricate any POIID. Use only those listed in the candidate set.

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority. Each POIID MUST be from the candidate set above. Do NOT include any POIID that is not listed.
2. "reason": A brief explanation of your reasoning.
3. "confidence": A value between 0 and 1 indicating how confident you are in the recommendation results (1 = highly confident, 0 = not confident).
Before outputting, CHECK that every POIID in your "recommendation" is in the <candidate set>.
If any POIID is outside the candidate set, your answer will be treated as invalid.
"""
        messages = [{"role": "user", "content": habit_prompt}]
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
            # print(f"Habitual analyst response content: {response_content}")
            response = parse_llm_response_with_scores_robust_conf(response_content)
            response["prompt"] = habit_prompt  # Attach prompt to response.
        except Exception as e:
            print(f"Habitual analyst parse error: {e}")
            # Fallback to simplified recommendation logic.
            default_recs = self.recommend(data)
            default_recs = default_recs[:10]
            response = {
                "prompt": habit_prompt,
                "recommendation": default_recs,
                "reason": "Default historical-frequency fallback recommendation.",
                "confidence": 0.5
            }
        
        candidates = response["recommendation"]
        candidates = [poi for poi in candidates if poi in candidate_pois]
        
        return response, candidates


    def generate_recommendation_final(self, data, candidate_pois, if_profile=True):
        """Final-stage LLM habitual recommendation for production usage."""
        profile = data["profile"]
        candidates = data["candidates"]
        recent = data["recent"]

        profile_str = json.dumps(profile, indent=2)
        profile_str = profile_str.replace('\\n', '\n')
        if if_profile:
            habit_prompt = f"""
You are the Behavioral Habit Analyst, specializing in analyzing user preferences based on historical visit patterns.
Your task is to recommend a user's next point-of-interest (POI) strictly from the <candidate set> listed below.

USER PROFILE:
{profile_str}

<recent check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}

Requirements:
Generate recommendations based ONLY on habitual preferences factors:
1. Prioritize the user's frequently visited locations
2. Consider the user's favorite categories and establishments
3. Consider recent changes in visitation patterns
4. Balance between familiar favorites and potentially interesting spots

Focus EXCLUSIVELY on HABITUAL PREFERENCES factors - do NOT consider time patterns, or context factors.

IMPORTANT:
You MUST ONLY output POIIDs from the <candidate set>.
If you output any POIID not in this set, your answer is invalid. This is a hard requirement.
Double-check your output: every recommended POIID must be present in the candidate set and not duplicated.
Do NOT invent, hallucinate, or fabricate any POIID. Use only those listed in the candidate set.

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority. Each POIID MUST be from the candidate set above. Do NOT include any POIID that is not listed.
2. "reason": A brief explanation of your reasoning.
"""
        else:
            # Build habitual analyst prompt (without user profile).
            habit_prompt = f"""
You are the Behavioral Habit Analyst, specializing in analyzing user preferences based on historical visit patterns.
Your task is to recommend a user's next point-of-interest (POI) strictly from the <candidate set> listed below.

<recent check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}

Requirements:
Generate recommendations based ONLY on habitual preferences factors:
1. Prioritize the user's frequently visited locations
2. Consider the user's favorite categories and establishments
3. Consider recent changes in visitation patterns
4. Balance between familiar favorites and potentially interesting spots

Focus EXCLUSIVELY on HABITUAL PREFERENCES factors - do NOT consider time patterns, or context factors.

IMPORTANT:
You MUST ONLY output POIIDs from the <candidate set>.
If you output any POIID not in this set, your answer is invalid. This is a hard requirement.
Double-check your output: every recommended POIID must be present in the candidate set and not duplicated.
Do NOT invent, hallucinate, or fabricate any POIID. Use only those listed in the candidate set.

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority. Each POIID MUST be from the candidate set above. Do NOT include any POIID that is not listed.
2. "reason": A brief explanation of your reasoning.
"""
        messages = [{"role": "user", "content": habit_prompt}]
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
            # print(f"Habitual analyst response content: {response_content}")
            response = parse_llm_response_with_scores_robust(response_content)
            response["prompt"] = habit_prompt  # Attach prompt to response.
        except Exception as e:
            print(f"Habitual analyst parse error: {e}")
            # Fallback to simplified recommendation logic.
            default_recs = self.recommend(data)
            default_recs = default_recs[:10]
            response = {
                "prompt": habit_prompt,
                "recommendation": default_recs,
                "reason": "Default historical-frequency fallback recommendation."
            }
        
        candidates = response["recommendation"]
        candidates = [poi for poi in candidates if poi in candidate_pois]
        
        return response, candidates