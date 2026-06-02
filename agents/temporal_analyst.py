import json
from .base_agent import BaseAgent
from utils import openaiAPIcall, parse_llm_response_with_scores_robust, LLM_NAME

class TemporalAnalyst(BaseAgent):
    """
    Temporal analyst focused on time patterns and temporal preferences.
    """
    def __init__(self, mapoi_instance):
        super().__init__(mapoi_instance)
        self.agent_name = "temporal_analyst"
    

    @staticmethod
    def _map_day_to_date_type(next_time_day: str) -> str:
        """Map input target day to one of: Weekday or Weekend."""
        d = str(next_time_day).strip().lower()
        if d in {"Saturday", "Sunday"}:
            return "Weekend"
        if d in {
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday"
        }:
            return "Weekday"
    
    def recommend(self, data):
        """Simplified recommendation implementation for pretraining."""
        candidates = data["candidates"]
        next_time_period = data["next_time_period"]
        longterm = data["longterm"]
        shared_data = self.get_shared_data()
        
        # Build temporal preference model.
        period_pois = {}
        
        # Learn period preferences from historical records.
        for poi, cat, day, period, _ in longterm:
            if period not in period_pois:
                period_pois[period] = {}
            
            if poi not in period_pois[period]:
                period_pois[period][poi] = 0
            
            period_pois[period][poi] += 1
        
        # Compute temporal relevance scores.
        time_scores = {}
        for poi, _, cat in candidates:
            # Initial score.
            time_scores[poi] = 0
            
            # Increase score if this POI appears in target time period.
            if next_time_period in period_pois and poi in period_pois[next_time_period]:
                time_scores[poi] += period_pois[next_time_period][poi] * 2
                
            # Temporal relevance from same-category POIs.
            if next_time_period in period_pois:
                for hist_poi, hist_score in period_pois[next_time_period].items():
                    if hist_poi in shared_data['poiInfos'] and poi in shared_data['poiInfos']:
                        hist_cat = shared_data['poiInfos'][hist_poi]["category"]
                        poi_cat = shared_data['poiInfos'][poi]["category"]
                        if hist_cat == poi_cat:
                            time_scores[poi] += hist_score * 0.5
            
            # # Get common categories in this period from knowledge base.
            # if next_time_period in shared_data['knowledge'].get(shared_data['datasetName'], {}).get("time_patterns", {}):
            #     common_categories = shared_data['knowledge'][shared_data['datasetName']]["time_patterns"][next_time_period]
            #     if any(common_cat.lower() in cat.lower() for common_cat in common_categories):
            #         time_scores[poi] += 1
        
        # Sort by temporal relevance.
        sorted_pois = sorted(time_scores.items(), key=lambda x: x[1], reverse=True)
        return [poi for poi, _ in sorted_pois][:20]  # Return top 20.
    
    def generate_recommendation(self, data, candidate_pois, if_profile=True):
        """Full LLM-based temporal recommendation for production usage."""
        profile = data["profile"]
        shared = self.get_shared_data()
        knowledge = shared.get('knowledge', {}) or {}

        candidates = data["candidates"] 

        next_time_period = data["next_time_period"]
        next_time_day = data["next_time_day"]
        
        # Dynamic priors.
        date_type = self._map_day_to_date_type(next_time_day)
        period_prior = (knowledge.get("period_category_distribution", {}) or {}).get(next_time_period, {}) or {}
        date_type_prior = (knowledge.get("date_type_category_distribution", {}) or {}).get(date_type, {}) or {}

        # Format priors (Top-K to avoid oversized prompts).
        def top_k_dict(d: dict, k=5):
            # Return top-K items sorted by probability.
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
            return {k_: float(v) for k_, v in items}

        period_prior_top = top_k_dict(period_prior, k=5)
        date_type_prior_top = top_k_dict(date_type_prior, k=5)


        profile_str = json.dumps(profile, indent=2)
        profile_str = profile_str.replace('\\n', '\n')
        # Extract content after "target_date_temporal_patterns" from profile string.
        temporal_patterns = profile_str.split("target_date_temporal_patterns")[1]
        
        if if_profile:
            temporal_prompt = f"""
You are a Temporal Reflector specializing in temporal pattern analysis for POI recommendations.
Your task is to adjust the ranking of POIs in the candidate set based on the user's temporal preferences.
Your goal is to slightly adjust the existing order to reflect temporal suitability, while keeping the overall order as stable as possible (minimal change principle).

TARGET TIME INFORMATION:
- Target Day: {next_time_day}, {date_type}
- Target Time Period: {next_time_period}

User's Temporal Patterns: {temporal_patterns}

General Temporal Category Priors:
- Category distribution for time period '{next_time_period}': {period_prior_top}
- Category distribution for date type '{date_type}': {date_type_prior_top}

Candidate POI Set: [Format: (POIID, Category)]: {candidates}

Analysis Steps:
1. Extract the user's preferred POI categories and POIs during the target time period and day from the temporal patterns and general priors.
2. Evaluate each candidate POI's temporal appropriateness:
- HIGH match: User frequently visits this POIID during the same weekday and time period
- MODERATE match: User frequently visits this POIID only during either the same weekday or time period, or frequently visits its category during the same weekday and time period.
- No match: Other cases
3. Adjust the ranking by moving match POIs forward 1, or 2 positions according to match level (2 for HIGH, 1 for MODERATE), while keeping original order for neutral POIs

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority. 
2. "reason": A brief explanation of your reasoning.
"""
        else:
            temporal_prompt = f"""
You are a Temporal Reflector specializing in temporal pattern analysis for POI recommendations.
Your task is to adjust the ranking of POIs in the candidate set based on the user's temporal preferences.
Your goal is to slightly adjust the existing order to reflect temporal suitability, while keeping the overall order as stable as possible (minimal change principle).

TARGET TIME INFORMATION:
- Target Day: {next_time_day}, {date_type}
- Target Time Period: {next_time_period}

User's Temporal Patterns: {temporal_patterns}

Candidate POI Set: [Format: (POIID, Category)]: {candidates}

Analysis Steps:
1. Extract the user's preferred POI categories and POIs during the target time period and day from the temporal patterns and general priors.
2. Evaluate each candidate POI's temporal appropriateness:
- HIGH match: User frequently visits this POIID during the same weekday and time period
- MODERATE match: User frequently visits this POIID only during either the same weekday or time period, or frequently visits its category during the same weekday and time period.
- No match: Other cases
3. Adjust the ranking by moving match POIs forward 1, or 2 positions according to match level (2 for HIGH, 1 for MODERATE), while keeping original order for neutral POIs

Output format:
Output a JSON with:
1. "recommendation": A ranked list of maximum 20 POIIDs, in order of recommendation priority. 
2. "reason": A brief explanation of your reasoning.
"""            
        messages = [{"role": "user", "content": temporal_prompt}]
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
            # print(f"Temporal analyst response content: {response_content}")
            response = parse_llm_response_with_scores_robust(response_content)
            response["prompt"] = temporal_prompt  # Attach prompt to response.
        except Exception as e:
            print("Temporal analyst parse error")
            print(response_content)
            # Fallback to simplified recommendation logic.
            default_recs = self.recommend(data)
            default_recs = default_recs[:20]
            response = {
                "prompt": temporal_prompt,
                "recommendation": default_recs,
                "reason": "Default target-time fallback recommendation."
            }
        candidates = response["recommendation"]
        candidates = [poi for poi in candidates if poi in candidate_pois]
        
        return response, candidates