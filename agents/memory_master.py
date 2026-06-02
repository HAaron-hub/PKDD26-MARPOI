import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import Optional, List, Dict
from utils import (
    time2period,
    openaiAPIcall,
    time2day,
    LLM_NAME,
    parse_profile,
    get_date_info,
)
from .agent_tools import invoke_tool


class MemoryMaster:
    def __init__(
        self,
        mapoi_instance=None,
        memory_dir: str = "./memory",
        weather_cache_file: Optional[str] = None,
    ):
        self.mapoi = mapoi_instance
        self.memory_dir = memory_dir
        os.makedirs(self.memory_dir, exist_ok=True)
        self._weather_cache: Dict[str, Optional[str]] = {}
        self._weather_cache_file: Optional[str] = weather_cache_file  # If provided, cache will be persisted after build.

    # ---------------------------
    # Public entry point
    # ---------------------------
    def build_or_load_memory(
        self,
        dataset_name: str,
        input_paths: Optional[List[str]] = None,
        force_rebuild: bool = False,
        save_transitions: bool = True
    ) -> dict:
        """
        Prefer loading memory from disk.
        If missing or force_rebuild=True, build from input files.
        input_paths uses `data/{dataset}/new_{train,test}_sample.csv`.
        """
        mem_root = os.path.join(self.memory_dir, dataset_name.lower())
        os.makedirs(mem_root, exist_ok=True)
        memory_path = os.path.join(mem_root, "global_memory.json")

        if (not force_rebuild) and os.path.exists(memory_path):
            return self._load_json(memory_path)

        # Auto-discover available inputs.
        if not input_paths:
            base = os.path.join("data", dataset_name.lower())
            candidates = [
                os.path.join(base, "new_train_sample.csv"),
                os.path.join(base, "new_test_sample.csv"),
            ]
            input_paths = [p for p in candidates if os.path.exists(p)]
        else:
            input_paths = [p for p in input_paths if os.path.exists(p)]

        if not input_paths:
            # No available input files, return empty knowledge skeleton.
            knowledge = self._empty_knowledge()
            self._save_json(memory_path, knowledge)
            return knowledge

        # Read and merge data.
        dfs: List[pd.DataFrame] = []
        for p in input_paths:
            df = self._load_any_dataset_csv(p, dataset_name)
            if df is not None and len(df) > 0:
                dfs.append(df)
        if not dfs:
            knowledge = self._empty_knowledge()
            self._save_json(memory_path, knowledge)
            return knowledge

        df_all = pd.concat(dfs, ignore_index=True)

        # Fill enhanced dimensions if missing.
        df_all = self._ensure_enhanced_columns(df_all, dataset_name)

        # Statistics and memory building.
        knowledge = self._compute_global_memory(df_all)

        # Optional: transition statistics and user profiling.
        if save_transitions:
            self._save_transitions(df_all, dataset_name)

        # Persist to disk.
        self._save_json(memory_path, knowledge)

        # Optional: persist weather cache.
        if self._weather_cache_file:
            self._save_json(self._weather_cache_file, self._weather_cache)

        return knowledge

    # ---------------------------
    # User profiling capability (migrated from UserProfilingAnalyst)
    # ---------------------------
    def data_preprocessing(self, trajectory, candidateSet, groundTruth):
        if self.mapoi is None:
            raise ValueError("MemoryMaster requires mapoi_instance to run data_preprocessing.")

        user_id = self.mapoi.traj2u[trajectory]
        long_history = self.mapoi.longs[user_id] if user_id in self.mapoi.longs else []
        recent_trajectory = self.mapoi.recents[trajectory] if trajectory in self.mapoi.recents else []

        current_poi = recent_trajectory[-1][0] if recent_trajectory else None

        long_selected = long_history[-40:] if len(long_history) > 40 else long_history[:]
        longterm = []
        for i, (poi, time) in enumerate(long_selected):
            if i == 0:
                travel_distance = 0
            else:
                prev_poi = long_selected[i - 1][0]
                travel_distance = invoke_tool(
                    "DisCalculate",
                    lat1=self.mapoi.poiInfos[prev_poi]["latitude"],
                    lon1=self.mapoi.poiInfos[prev_poi]["longitude"],
                    lat2=self.mapoi.poiInfos[poi]["latitude"],
                    lon2=self.mapoi.poiInfos[poi]["longitude"],
                )
            longterm.append((poi, self.mapoi.poiInfos[poi]["category"], time2day(time), time2period(time), travel_distance))
        longterm_groups = [longterm[i:i + 10] for i in range(0, len(longterm), 10)]

        rec_selected = recent_trajectory[-5:] if len(recent_trajectory) > 5 else recent_trajectory[:]
        recent = []
        for i, (poi, time) in enumerate(rec_selected):
            if i == 0:
                travel_distance = 0
            else:
                prev_poi = rec_selected[i - 1][0]
                travel_distance = invoke_tool(
                    "DisCalculate",
                    lat1=self.mapoi.poiInfos[prev_poi]["latitude"],
                    lon1=self.mapoi.poiInfos[prev_poi]["longitude"],
                    lat2=self.mapoi.poiInfos[poi]["latitude"],
                    lon2=self.mapoi.poiInfos[poi]["longitude"],
                )
            recent.append((poi, self.mapoi.poiInfos[poi]["category"], time2day(time), time2period(time), travel_distance))

        candidates = []
        if current_poi:
            for poi in candidateSet:
                if poi in self.mapoi.poiInfos and current_poi in self.mapoi.poiInfos:
                    try:
                        dist = invoke_tool(
                            "DisCalculate",
                            lat1=self.mapoi.poiInfos[poi]["latitude"],
                            lon1=self.mapoi.poiInfos[poi]["longitude"],
                            lat2=self.mapoi.poiInfos[current_poi]["latitude"],
                            lon2=self.mapoi.poiInfos[current_poi]["longitude"],
                        )
                        candidates.append((poi, dist, self.mapoi.poiInfos[poi]["category"]))
                    except Exception:
                        candidates.append((poi, 999, self.mapoi.poiInfos[poi]["category"]))
            candidates.sort(key=lambda x: x[1])
        else:
            candidates = [(poi, 0, self.mapoi.poiInfos[poi]["category"]) for poi in candidateSet if poi in self.mapoi.poiInfos]

        current_time = recent_trajectory[-1][1] if recent_trajectory else None
        next_time = groundTruth[1]
        next_time_day = time2day(next_time)
        next_time_period = time2period(next_time)

        return user_id, longterm_groups, longterm, recent, next_time_day, next_time_period, candidates, current_poi, current_time

    def generate_user_profile(self, user_id, longterm, recent, next_time_day, next_time_period):
        profile_prompt = f"""\
<long-term check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {longterm}
<recent check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {recent}
Your task is to generate a standardized JSON user profile to be used as input for another LLM to predict future check-in behavior based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Category" shows the semantic information of the POI, "Day of week" shows the day of the week when the user visited the POI, "Period of day" shows the time period when the user visited the POI, and "Travel distance" shows the distance (kilometers) between the last POI and current POI.

IMPORTANT: For the fields "frequent_locations", "favorite_categories", "common_location_sequences", "common_category_sequences", "new_locations", "increased_frequency_locations", "decreased_frequency_locations", "potential_locations", and "potential_categories", please include no more than 5 items (if there are more, only list the top 5 by frequency as appropriate).

The output JSON should strictly follow this schema:
{{
  "user_basic_info": {{
    "user_id": {user_id},
    "likely_occupation": "string",
    "age_stage": "string"
  }},
  "location_preferences": {{
    "frequent_locations": ["POIID"],
    "favorite_categories": ["Category"],
    "home_location": {{"poi_id": "string", "category": "string", "confidence": "number"}}
  }},
  "behavior_patterns": {{
    "common_location_sequences": [{{"sequence": ["POIID", "POIID"]}}],
    "common_category_sequences": [{{"sequence": ["Category", "Category"]}}],
    "travel_radius": {{"average_km": "number", "max_km": "number"}}
  }},
  "recent_changes": {{
    "new_locations": [{{"poi_id": "string", "category": "string"}}],
    "increased_frequency_locations": [{{"poi_id": "string", "category": "string"}}],
    "decreased_frequency_locations": [{{"poi_id": "string", "category": "string"}}]
  }},
  "target_date_temporal_patterns": {{
    "target_day": {json.dumps(next_time_day)},
    "potential_locations": ["POIID"],
    "potential_categories": ["Category"]
  }},
  "target_time_period_temporal_patterns": {{
    "target_time_period": {json.dumps(next_time_period)},
    "potential_locations": ["POIID"],
    "potential_categories": ["Category"]
  }}
}}
"""

        profile_messages = [{"role": "user", "content": profile_prompt}]
        profile_response = openaiAPIcall(
            model=LLM_NAME,
            messages=profile_messages,
            temperature=0,
            top_p=0.95,
            presence_penalty=0,
            frequency_penalty=0,
            extra_body={"top_k": 50},
        )
        profile_content = profile_response.choices[0].message.content

        try:
            profile = parse_profile(profile_content)
        except Exception as parse_error:
            print(f"Memory analyst: JSON parsing failed {parse_error}, using raw content")
            print(f"Raw content: {profile_content}")
            profile = profile_content

        return profile_prompt, profile

    def update_user_profile(self, user_id, profile, recent, next_time_day, next_time_period):
        profile_str = json.dumps(profile, indent=2)
        profile_str = profile_str.replace('\\n', '\n')
        profile_prompt = f"""\
You are given a user's previous JSON profile and a sequence of their recent check-ins (recent trajectory). Your task is to update and refine the user's standardized JSON profile according to the latest behavioral information.

The previous profile is as follows:
<old_profile_json>: {profile_str}

The user's recent check-ins are as follows:
<recent check-ins> [Format: (POIID, Category, Day of week, Period of day, Travel distance)]: {recent}

IMPORTANT: For the fields "frequent_locations", "favorite_categories", "common_location_sequences", "common_category_sequences", "new_locations", "increased_frequency_locations", "decreased_frequency_locations", "potential_locations", and "potential_categories", please include no more than 5 items (if there are more, only list the top 5 by frequency as appropriate).

Instructions:
- Analyze the recent check-ins and compare them with the old profile.
- Update or revise all relevant fields in the profile to reflect new patterns, changes in preferences, and any other detected behavioral shifts.
- For fields such as "frequent_locations", "favorite_categories", "common_location_sequences", "common_category_sequences", "new_locations", "increased_frequency_locations", "decreased_frequency_locations", "potential_locations", and "potential_categories", only include up to 5 items (keep the top 5 by frequency if more exist).
- If new locations or categories appear in recent check-ins, update the corresponding fields.
- If location/category frequencies increase or decrease compared to the old profile, update "increased_frequency_locations" or "decreased_frequency_locations" accordingly.
- If the user's travel radius, home location, or temporal patterns show significant changes, update those fields as well.
- Ensure the output strictly follows the schema below and remains consistent and coherent.

The output JSON should strictly follow this schema:
{{
  "user_basic_info": {{
    "user_id": {user_id},
    "likely_occupation": "string",
    "age_stage": "string"
  }},
  "location_preferences": {{
    "frequent_locations": ["POIID"],
    "favorite_categories": ["Category"],
    "home_location": {{"poi_id": "string", "category": "string", "confidence": "number"}}
  }},
  "behavior_patterns": {{
    "common_location_sequences": [{{"sequence": ["POIID", "POIID"]}}],
    "common_category_sequences": [{{"sequence": ["Category", "Category"]}}],
    "travel_radius": {{"average_km": "number", "max_km": "number"}}
  }},
  "recent_changes": {{
    "new_locations": [{{"poi_id": "string", "category": "string"}}],
    "increased_frequency_locations": [{{"poi_id": "string", "category": "string"}}],
    "decreased_frequency_locations": [{{"poi_id": "string", "category": "string"}}]
  }},
  "target_date_temporal_patterns": {{
    "target_day": {json.dumps(next_time_day)},
    "potential_locations": ["POIID"],
    "potential_categories": ["Category"]
  }},
  "target_time_period_temporal_patterns": {{
    "target_time_period": {json.dumps(next_time_period)},
    "potential_locations": ["POIID"],
    "potential_categories": ["Category"]
  }}
}}
"""

        profile_messages = [{"role": "user", "content": profile_prompt}]
        profile_response = openaiAPIcall(
            model=LLM_NAME,
            messages=profile_messages,
            temperature=0,
            top_p=0.95,
            presence_penalty=0,
            frequency_penalty=0,
            extra_body={"top_k": 50},
        )
        profile_content = profile_response.choices[0].message.content

        try:
            profile = parse_profile(profile_content)
        except Exception as parse_error:
            print(f"Memory analyst: JSON parsing failed {parse_error}, using raw content")
            print(f"Raw content: {profile_content}")
            profile = profile_content

        return profile_prompt, profile

    def user_profilling(self, user_id, longterm_groups, recent, next_time_day, next_time_period):
        if len(longterm_groups) == 1:
            profile_prompt, profile = self.generate_user_profile(
                user_id, longterm_groups[0], recent, next_time_day, next_time_period
            )
        else:
            profile_prompt, profile = self.generate_user_profile(
                user_id, longterm_groups[0], longterm_groups[1], next_time_day, next_time_period
            )
            if len(longterm_groups) > 2:
                for group in longterm_groups[2:]:
                    profile_prompt, profile = self.update_user_profile(
                        user_id, profile, group, next_time_day, next_time_period
                    )
            profile_prompt, profile = self.update_user_profile(
                user_id, profile, recent, next_time_day, next_time_period
            )
        return profile_prompt, profile

    # ---------------------------
    # Internal: load/save
    # ---------------------------
    def _load_global(self, dataset_name: str) -> dict:
        path = os.path.join(self.memory_dir, dataset_name.lower(), "global_memory.json")
        if os.path.exists(path):
            return self._load_json(path)
        return self._empty_knowledge()

    def _empty_knowledge(self) -> dict:
        return {
            "popular_pois": [],
            "popular_categories": [],
            "category_distribution": {},
            "period_category_distribution": {},
            "date_type_category_distribution": {},
            "weather_category_distribution": {},
            "meta": {
                "version": 2,
                "desc": "Auto-built memory from dataset; includes global popular categories.",
            }
        }

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, path: str, obj: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # ---------------------------
    # Internal: data loading and enhancement
    # ---------------------------
    def _load_any_dataset_csv(self, csv_path: str, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Support two input formats:
        1) Preprocessed/enhanced CSV with columns such as UserId, PoiId,
           PoiCategoryName, Latitude, Longitude, UTCTimeOffsetEpoch or datetime,
           and optional date_type, time_period, weather_type.
        2) Raw new_*_sample.csv (no header or different headers), parsed by
           positional mapping consistent with main.py.
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[MemoryMaster] Read failed: {csv_path}, err={e}")
            return None

        cols = set(df.columns)

        # Enhanced input: normalize fields directly and keep enhanced columns.
        if {"UserId", "PoiId", "PoiCategoryName"}.issubset(cols):
            # Parse/normalize time column.
            if "datetime" not in cols and "UTCTimeOffsetEpoch" in cols:
                df["datetime"] = pd.to_datetime(df["UTCTimeOffsetEpoch"], unit="s")
            elif "datetime" in cols:
                try:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                except Exception:
                    # Try inferring from other candidate columns.
                    for c in ["Time", "time", "timestamp", "UTCTimeOffsetEpoch"]:
                        if c in cols:
                            try:
                                if c == "UTCTimeOffsetEpoch":
                                    df["datetime"] = pd.to_datetime(df[c], unit="s")
                                else:
                                    df["datetime"] = pd.to_datetime(df[c])
                                break
                            except Exception:
                                pass

            # Normalize latitude/longitude naming.
            for a, b in [("Latitude", "lat"), ("Longitude", "lon")]:
                if a not in df.columns and b in df.columns:
                    df[a] = df[b]

            # Keep enhanced columns whenever possible to avoid recomputation.
            keep_cols = [
                "UserId", "PoiId", "PoiCategoryName", "Latitude", "Longitude", "datetime",
                "date_type", "time_period", "weather_type"
            ]
            present = [c for c in keep_cols if c in df.columns]
            if not present:
                # Fallback: keep at least the minimum required fields.
                present = [c for c in ["UserId","PoiId","PoiCategoryName","Latitude","Longitude","datetime"] if c in df.columns]
            return df[present].copy()

        # Raw input: use positional mapping based on main.py (nyc/tky format).
        try:
            if df.shape[1] < 11:
                print(f"[MemoryMaster] Not enough columns for positional parsing: {csv_path}")
                return None
            out = pd.DataFrame()
            out["datetime"] = pd.to_datetime(df.iloc[:, 1])
            out["UserId"] = df.iloc[:, 5].astype(str)
            out["Latitude"] = pd.to_numeric(df.iloc[:, 6], errors="coerce")
            out["Longitude"] = pd.to_numeric(df.iloc[:, 7], errors="coerce")
            out["PoiId"] = df.iloc[:, 8].astype(str)
            cat_series = df.iloc[:, 10].astype(str)
            if cat_series.isna().all():
                cat_series = df.iloc[:, 9].astype(str)
            out["PoiCategoryName"] = cat_series
            return out
        except Exception as e:
            print(f"[MemoryMaster] Positional parsing failed: {csv_path}, err={e}")
            return None

    def _ensure_enhanced_columns(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Ensure date_type, time_period, and weather_type columns exist.
        Missing columns are filled using preprocess-compatible logic.
        """
        df = df.copy()
        # Time column.
        if "datetime" not in df.columns:
            raise ValueError("datetime column is required for enhancement.")

        # Date type.
        if "date_type" not in df.columns:
            tqdm.pandas(desc="Memory: date_type")
            df["date_type"] = df["datetime"].progress_apply(
                lambda x: get_date_info(x.strftime("%Y-%m-%d %H:%M:%S"), dataset_name)
            )

        # Time period.
        if "time_period" not in df.columns:
            tqdm.pandas(desc="Memory: time_period")
            df["time_period"] = df["datetime"].progress_apply(
                lambda x: time2period(x.strftime("%Y-%m-%d %H:%M:%S"))
            )

        # Weather type.
        if "weather_type" not in df.columns:
            if {"Latitude", "Longitude"}.issubset(set(df.columns)):
                tqdm.pandas(desc="Memory: weather_type")
                weather_types: List[str] = []
                # Cache by latitude/longitude + date.
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Memory: weather fetch"):
                    lat = row.get("Latitude", np.nan)
                    lon = row.get("Longitude", np.nan)
                    date = row["datetime"].date()
                    wt = self._get_weather_type(lat, lon, date, precision=4)
                    weather_types.append(wt)
                df["weather_type"] = weather_types
            else:
                df["weather_type"] = "Unknown"
        return df

    # ---------------------------
    # Internal: memory construction (statistics)
    # ---------------------------
    def _compute_global_memory(self, df: pd.DataFrame) -> dict:
        knowledge = self._empty_knowledge()

        # Popular POIs.
        if "PoiId" in df.columns:
            poi_counts = df["PoiId"].value_counts().to_dict()
            # Sort and convert to list[(poi, count)].
            knowledge["popular_pois"] = sorted(poi_counts.items(), key=lambda x: (-x[1], x[0]))

        # period -> category distribution.
        pdc: Dict[str, Counter] = defaultdict(Counter)
        # date_type -> category distribution.
        ddc: Dict[str, Counter] = defaultdict(Counter)
        # weather_type -> category distribution.
        wdc: Dict[str, Counter] = defaultdict(Counter)

        cat_col: Optional[str] = "PoiCategoryName" if "PoiCategoryName" in df.columns else None
        if cat_col is None:
            # If category name is missing, try categoryID.
            if "categoryID" in df.columns:
                cat_col = "categoryID"

        # Global popular categories + global category distribution.
        if cat_col:
            category_counts = df[cat_col].astype(str).value_counts().to_dict()
            knowledge["popular_categories"] = sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))
            total_c = sum(category_counts.values())
            knowledge["category_distribution"] = {k: v / total_c for k, v in category_counts.items()} if total_c > 0 else {}

            for _, row in df.iterrows():
                cat = str(row[cat_col]) if pd.notna(row[cat_col]) else "Unknown"
                period = row.get("time_period", "Unknown")
                date_type = row.get("date_type", "Unknown")
                weather = row.get("weather_type", "Unknown")
                pdc[period][cat] += 1
                ddc[date_type][cat] += 1
                wdc[weather][cat] += 1

            knowledge["period_category_distribution"] = {
                k: self._normalize_counter(v) for k, v in pdc.items()
            }
            knowledge["date_type_category_distribution"] = {
                k: self._normalize_counter(v) for k, v in ddc.items()
            }
            knowledge["weather_category_distribution"] = {
                k: self._normalize_counter(v) for k, v in wdc.items()
            }

        return knowledge

    def _save_transitions(self, df: pd.DataFrame, dataset_name: str):
        """
        Save simple transition statistics (usable as next-step priors).
        Two granularities:
          - poi_to_poi: (prev_poi -> next_poi)
          - cat_to_cat: (prev_cat -> next_cat)
        """
        mem_root = os.path.join(self.memory_dir, dataset_name.lower())
        path = os.path.join(mem_root, "transitions.json")

        # Requires per-user sequences; sort by (UserId, datetime) when possible.
        if "UserId" not in df.columns or "datetime" not in df.columns:
            return

        cat_col: Optional[str] = "PoiCategoryName" if "PoiCategoryName" in df.columns else None
        if cat_col is None and "categoryID" in df.columns:
            cat_col = "categoryID"

        poi_to_poi: Dict[str, Counter] = defaultdict(Counter)
        cat_to_cat: Dict[str, Counter] = defaultdict(Counter)

        # Aggregate by user.
        for _, g in df.groupby("UserId"):
            g = g.sort_values("datetime")
            prev_poi, prev_cat = None, None
            for _, row in g.iterrows():
                cur_poi = str(row["PoiId"]) if "PoiId" in row else None
                cur_cat = str(row[cat_col]) if cat_col and pd.notna(row[cat_col]) else None
                if prev_poi and cur_poi:
                    poi_to_poi[prev_poi][cur_poi] += 1
                if prev_cat and cur_cat:
                    cat_to_cat[prev_cat][cur_cat] += 1
                prev_poi, prev_cat = cur_poi, cur_cat

        data = {
            "poi_to_poi": {k: self._normalize_counter(v) for k, v in poi_to_poi.items()},
            "cat_to_cat": {k: self._normalize_counter(v) for k, v in cat_to_cat.items()},
        }
        self._save_json(path, data)

    @staticmethod
    def _normalize_counter(counter: Counter) -> dict:
        total = sum(counter.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in counter.items()}

    # ---------------------------
    # Utility adapted from preprocess.py (internal method)
    # ---------------------------
    def _get_weather_type(self, lat: float, lon: float, date, precision: int = 4) -> str:
        if pd.isna(lat) or pd.isna(lon):
            return "Unknown"
        key = f"{str(date)}_{lat:.{precision}f}_{lon:.{precision}f}"
        if key in self._weather_cache:
            weather_type = self._weather_cache[key]
        else:
            date_str = datetime(date.year, date.month, date.day).strftime("%Y-%m-%d %H:%M:%S")
            weather_type = invoke_tool(
                "WeatherSearch",
                latitude=lat,
                longitude=lon,
                date_str=date_str,
            ) or "Unknown"
            self._weather_cache[key] = weather_type

        return weather_type or "Unknown"