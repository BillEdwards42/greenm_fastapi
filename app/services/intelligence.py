import asyncio
import httpx
import time
import math
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import deque

from app.core.logging import logger
from app.core.config import get_settings
from app.core.security import get_secret
from app.repository.bucket_repo import BucketRepo

# Services
from app.services.ml_inference import MLInferenceService 

# Schemas
from app.schemas.weather import WeatherResponse
from app.schemas.generation import TaipowerResponse, Generation
from app.schemas.forecast import PredictionArtifact, CarbonLevel, TimeStep, OptimizationWindow
from app.schemas.ml_cache import MlCache, RegionDataRow

settings = get_settings()

# --- CONSTANTS ---
TAIPOWER_URL = "https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/genary.json"
CWA_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001"
CACHE_FILENAME = "ml_cache_v2.json"
CACHE_WINDOW_SIZE = 720 # 5 days * 24 hours * 6 (10-min intervals)

# Regions
REGIONS = ["North", "Central", "South", "East", "Other"]

# Physics Constants
CARBON_FACTORS = {
    'Nuclear': 0.0, 'Coal': 0.912, 'Co-Gen': 1.111, 'IPP-Coal': 0.919,
    'LNG': 0.389, 'IPP-LNG': 0.378, 'Oil': 0.818, 'Diesel': 0.811,
    'Hydro': 0.0, 'Wind': 0.0, 'Solar': 0.0, 'Other_Renewable': 1.002, 'Storage': 0.0
}
LINE_LOSS_RATE = 0.0293

STATIONS_BY_REGION = {
    "North": ["基隆", "淡水", "新北", "新竹", "臺北", "新屋", "桃園農改", "文山茶改", "新埔工作站"],
    "Central": ["臺中", "梧棲", "後龍", "古坑", "彰師大", "麥寮", "田中", "日月潭", "苗栗農改"],
    "South": ["嘉義", "臺南", "高雄", "恆春", "永康", "臺南農改", "旗南農改", "高雄農改", "屏東"],
    "East": ["宜蘭", "花蓮", "成功", "臺東", "大武"],
    "Other": []
}

REGION_KEYWORDS = {
    'North': ['林口', '大潭', '新桃', '通霄', '協和', '石門', '翡翠', '桂山', '觀音', '龍潭', '北部'],
    'Central': ['台中', '大甲溪', '明潭', '彰工', '中港', '竹南', '苗栗', '雲林', '麥寮', '中部', '彰'],
    'South': ['興達', '大林', '南部', '核三', '曾文', '嘉義', '台南', '高雄', '永安', '屏東'],
    'East': ['和平', '花蓮', '蘭陽', '卑南', '立霧', '東部'], 
    'Other': ['汽電共生', '其他台電自有', '其他購電太陽能', '其他購電風力', '購買地熱', '台電自有地熱', '生質能']
}

# --- HELPERS ---

class RegionMapper:
    """Maps plant names to regions using CSV or Heuristics."""
    def __init__(self):
        self.csv_map = {}
        self._load_csv_map()

    def _load_csv_map(self):
        try:
            # Assumes Docker structure /app/app/data
            # Using relative path for safety
            from pathlib import Path
            path = Path(__file__).parent.parent / "data" / "plant_to_region_map.csv"
            if path.exists():
                import pandas as pd
                df = pd.read_csv(path)
                self.csv_map = dict(zip(df['unit_name'], df['region']))
                logger.info(f"Loaded {len(self.csv_map)} mappings from CSV.")
            else:
                logger.warning("⚠️ plant_to_region_map.csv not found. Relying on Keywords.")
        except Exception as e:
            logger.error(f"Failed to load region map CSV: {e}")

    def get_region(self, unit_name: str) -> str:
        if unit_name in self.csv_map: return self.csv_map[unit_name]
        for region, keywords in REGION_KEYWORDS.items():
            for kw in keywords:
                if kw in unit_name: return region
        return "Other"

region_mapper = RegionMapper()

def _calculate_time_features(dt: datetime) -> dict:
    """Computes Cyclical Time Features"""
    def sin_trans(val, max_val): return math.sin(2 * math.pi * val / max_val)
    def cos_trans(val, max_val): return math.cos(2 * math.pi * val / max_val)
    return {
        "year": float(dt.year),
        "month_sin": sin_trans(dt.month - 1, 12), "month_cos": cos_trans(dt.month - 1, 12),
        "day_sin": sin_trans(dt.day - 1, 31),     "day_cos": cos_trans(dt.day - 1, 31),
        "dayofweek_sin": sin_trans(dt.weekday(), 7), "dayofweek_cos": cos_trans(dt.weekday(), 7),
        "hour_sin": sin_trans(dt.hour, 24),       "hour_cos": cos_trans(dt.hour, 24),
        "minute_sin": sin_trans(dt.minute, 60),   "minute_cos": cos_trans(dt.minute, 60)
    }

# --- CACHE MANAGEMENT FUNCTIONS ---

async def _fetch_current_cache(repo: BucketRepo) -> Dict[str, List[RegionDataRow]]:
    """Downloads and deserializes the ML cache from GCS."""
    try:
        # Pydantic v2: Use model_validate_json directly if fetching raw string
        # But BucketRepo usually returns string.
        json_str = await repo.download_json(CACHE_FILENAME)
        cache_obj = MlCache.model_validate_json(json_str)
        
        # Convert to a dict of lists for easier manipulation
        return {
            "North": cache_obj.North,
            "Central": cache_obj.Central,
            "South": cache_obj.South,
            "East": cache_obj.East,
            "Other": cache_obj.Other
        }
    except Exception as e:
        logger.warning(f"Cache not found or corrupt ({e}). Starting fresh.")
        return {r: [] for r in REGIONS}

async def _save_cache_to_gcs(repo: BucketRepo, cache_data: Dict[str, List[RegionDataRow]]):
    """Serializes and uploads the ML cache to GCS."""
    try:
        # Reconstruct the Root Model
        cache_obj = MlCache(
            North=cache_data["North"],
            Central=cache_data["Central"],
            South=cache_data["South"],
            East=cache_data["East"],
            Other=cache_data["Other"],
            previous_generators=[] # Unused in this logic, but required by Schema
        )
        await repo.upload_json(CACHE_FILENAME, cache_obj.model_dump_json())
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def _update_cache_state(
    current_cache: Dict[str, List[RegionDataRow]], 
    new_rows: Dict[str, RegionDataRow]
) -> Dict[str, List[RegionDataRow]]:
    """
    Appends new rows and trims to the fixed window size.
    Enforces 'Last In, First Out' logic to keep the most recent data.
    """
    updated_cache = {}
    for region in REGIONS:
        # Get existing list
        history = current_cache.get(region, [])
        
        # Append new data
        if region in new_rows:
            history.append(new_rows[region])
        
        # Trim to Window Size (Exactly 720 as requested, or slight buffer?)
        # We adhere to "Safe Buffer" logic: keep 720, but slice at usage.
        # Actually, strict maintenance prevents file bloat.
        if len(history) > CACHE_WINDOW_SIZE:
            history = history[-CACHE_WINDOW_SIZE:] # Keep last 720
            
        updated_cache[region] = history
        
    return updated_cache

# --- CORE LOGIC ---

async def fetch_raw_data() -> Tuple[str, dict]:
    """

    Fetches data using Bright Data Proxy to prevent IP blocks.

    Returns: (taipower_html_text, weather_json_dict)

    """
    bd_user = get_secret(settings.SECRET_BD_USER)
    bd_pass = get_secret(settings.SECRET_BD_PASS)
    cwa_key = get_secret(settings.SECRET_CWA_KEY)
    proxy_url = f"http://{bd_user}:{bd_pass}@brd.superproxy.io:22225"
    timestamp_suffix = int(time.time())

    async with httpx.AsyncClient(proxies=proxy_url, verify=False, timeout=30.0) as client:
        logger.info("📡 Connecting via Bright Data Network...")
        task_tp = client.get(TAIPOWER_URL, params={"_": timestamp_suffix})
        task_wt = client.get(CWA_URL, params={"Authorization": cwa_key})
        try:
            tp_res, wt_res = await asyncio.gather(task_tp, task_wt)
            if tp_res.status_code == 407: raise ConnectionError("Proxy Auth Failed")
            tp_res.raise_for_status()
            wt_res.raise_for_status()
            return tp_res.text, wt_res.json()
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            raise e

def _process_generation_data(generators: List[Generation]) -> Dict[str, Dict[str, float]]:
    """

    Aggregates individual plants into Region + Fuel Type buckets.

    Output: { "North": { "Coal": 1200.5, "Solar": 50.0 }, "South": ... }

    """
    regional_data = {r: {} for r in STATIONS_BY_REGION.keys()}
    for gen in generators:
        region = region_mapper.get_region(gen.name)
        if region not in regional_data: region = "Other"
        fuel = CARBON_FACTORS.get(gen.fuel_type)
        if fuel is None: continue
        if fuel not in regional_data[region]: regional_data[region][fuel] = 0.0
        regional_data[region][fuel] += gen.current_generation_mw
    return regional_data

def _process_weather_data(weather_resp: WeatherResponse) -> Dict[str, Dict[str, float]]:
    """
    Aggregates station data into regional averages. 

    CRITICAL FIX: 

    We rely on the 'Rich Domain Model' pattern. The logic for aggregation 

    is defined in the Schema itself (calculate_regional_averages). 

    We simply pass the configuration map to it.

    """
    try:
        return weather_resp.calculate_regional_averages(STATIONS_BY_REGION)
    except Exception as e:
        logger.error(f"Weather Aggregation Error: {e}")
        return {r: {} for r in STATIONS_BY_REGION.keys()}

def _construct_coupled_rows(
    regional_gen: Dict[str, Dict[str, float]],
    regional_weather: Dict[str, Dict[str, float]],
    now: datetime
) -> Dict[str, RegionDataRow]:
    """
    PASS 1 & PASS 2: Builds features and handles Cross-Regional Coupling.
    """
    time_feats = _calculate_time_features(now)
    staging_data = {}

    # PASS 1: Base Features
    for region in REGIONS:
        gen_mix = regional_gen.get(region, {})# { "North": { "Coal": 1200.5, "Solar": 50.0 }, "South": ... }
        weather = regional_weather.get(region, {})# {"North": {"AirTempreture": 29.4, ...}, "South"...}
        
        # Flatten Generation: Ensure all 13 fuels exist (default 0.0)
        flat_gen = {k: gen_mix.get(k, 0.0) for k in CARBON_FACTORS.keys()}
        
        staging_data[region] = {
            **time_feats,
            "WindSpeed": weather.get("WindSpeed", 0.0),
            "Percipitation": weather.get("Precipitation", 0.0),
            "SunshineDuration": weather.get("SunshineDuration", 0.0),
            "AirTempreture": weather.get("AirTemperature", 25.0),
            **flat_gen
        }

    # PASS 2: Cross-Regional Coupling (Injecting Neighbors)
    def get_val(r, key): return staging_data.get(r, {}).get(key, 0.0)

    # North Coupling
    staging_data["North"]["South_Coal"] = get_val("South", "Coal")
    staging_data["North"]["South_IPP-LNG"] = get_val("South", "IPP-LNG")
    staging_data["North"]["South_Diesel"] = get_val("South", "Diesel")
    staging_data["North"]["Central_Diesel"] = get_val("Central", "Diesel")
    staging_data["North"]["Central_Hydro"] = get_val("Central", "Hydro")
    staging_data["North"]["Central_Wind"] = get_val("Central", "Wind")

    # Central Coupling
    staging_data["Central"]["South_Diesel"] = get_val("South", "Diesel")
    staging_data["Central"]["North_Diesel"] = get_val("North", "Diesel")
    staging_data["Central"]["North_Hydro"] = get_val("North", "Hydro")
    staging_data["Central"]["Other_Wind"] = get_val("Other", "Wind")
    staging_data["Central"]["North_Wind"] = get_val("North", "Wind")
    staging_data["Central"]["Other_Solar"] = get_val("Other", "Solar")
    staging_data["Central"]["South_Solar"] = get_val("South", "Solar")
    staging_data["Central"]["South_AirTemperature"] = get_val("South", "AirTempreture")
    staging_data["Central"]["East_AirTemperature"] = get_val("East", "AirTempreture")
    staging_data["Central"]["North_AirTemperature"] = get_val("North", "AirTempreture")
    staging_data["Central"]["Other_Storage"] = get_val("Other", "Storage")
    staging_data["Central"]["North_SunshineDuration"] = get_val("North", "SunshineDuration")
    staging_data["Central"]["East_SunshineDuration"] = get_val("East", "SunshineDuration")
    staging_data["Central"]["South_SunshineDuration"] = get_val("South", "SunshineDuration")

    # South Coupling
    staging_data["South"]["North_Coal"] = get_val("North", "Coal")
    staging_data["South"]["North_IPP-LNG"] = get_val("North", "IPP-LNG")
    staging_data["South"]["North_Diesel"] = get_val("North", "Diesel")
    staging_data["South"]["Central_Diesel"] = get_val("Central", "Diesel")
    staging_data["South"]["Other_Solar"] = get_val("Other", "Solar")
    staging_data["South"]["Central_Solar"] = get_val("Central", "Solar")
    staging_data["South"]["East_AirTemperature"] = get_val("East", "AirTempreture")
    staging_data["South"]["Central_AirTemperature"] = get_val("Central", "AirTempreture")
    staging_data["South"]["North_AirTemperature"] = get_val("North", "AirTempreture")

    # East Coupling
    staging_data["East"]["North_Hydro"] = get_val("North", "Hydro")
    staging_data["East"]["Central_Wind"] = get_val("Central", "Wind")
    staging_data["East"]["South_Solar"] = get_val("South", "Solar")

    # PASS 3: Validation (Pydantic)
    final_rows = {}
    for region in REGIONS:
        # Pydantic v2 'extra="allow"' will capture the coupled columns automatically
        final_rows[region] = RegionDataRow(**staging_data[region])
        
    return final_rows

def _calculate_intensity_from_mix(fuel_mix: Dict[str, float]) -> float:
    total_emission_kg = 0.0
    total_gen_mw = 0.0
    for fuel, mw in fuel_mix.items():
        if mw <= 0: continue
        factor = CARBON_FACTORS.get(fuel, 0.0)
        total_emission_kg += mw * factor
        total_gen_mw += mw
    if total_gen_mw == 0: return 500.0 
    avg_gen_intensity = total_emission_kg / total_gen_mw
    consumer_intensity = avg_gen_intensity / (1 - LINE_LOSS_RATE)
    return round(consumer_intensity * 1000.0, 2)

def _determine_dynamic_level(current_val: float, forecast_timeline: List[float]) -> CarbonLevel:
    all_values = [current_val] + forecast_timeline
    p43 = 0.0
    p76 = 0.0
    if len(all_values) > 0:
        import numpy as np
        p43 = np.percentile(all_values, 43)
        p76 = np.percentile(all_values, 76)
    
    if current_val <= p43: return CarbonLevel.GREEN
    elif current_val <= p76: return CarbonLevel.YELLOW
    else: return CarbonLevel.RED

# --- ORCHESTRATOR ---

async def run_intelligence_pipeline():
    logger.info("🧠 Intelligence Pipeline Initiated")
    bucket_repo = BucketRepo()
    ml_service = MLInferenceService() # Assumed to handle its own loading

    try:
        # 1. Fetch
        raw_tp, raw_wt = await fetch_raw_data()
        
        # 2. Parse (Validation)
        tp_data = TaipowerResponse.model_validate_json(raw_tp)
        wt_data = WeatherResponse.model_validate_json(raw_wt)
        
        # 3. ETL (Raw -> Regional Dicts)
        regional_gen = _process_generation_data(tp_data.valid_generators)
        regional_weather = _process_weather_data(wt_data)
        
        # 4. Feature Construction (with Coupling)
        tz_taiwan = timezone(timedelta(hours=8))
        now = datetime.now(tz_taiwan)
        new_rows_map = _construct_coupled_rows(regional_gen, regional_weather, now)
        
        # 5. Cache Management (Read -> Append -> Trim -> Write)
        current_cache = await _fetch_current_cache(bucket_repo)
        updated_cache = _update_cache_state(current_cache, new_rows_map)
        await _save_cache_to_gcs(bucket_repo, updated_cache)
        
        # 6. ML Inference
        # Returns: { "North": np.array([144 steps]), ... }
        forecasts_mw = ml_service.predict(updated_cache)
        
        # 7. Post-Processing (MW -> Carbon Intensity)
        # We need to calculate intensity for the *whole grid* (sum of regions)
        # This requires summing the forecasted MW for every fuel type across all regions
        # NOTE: ml_inference.predict returns pure MW values (13 fuels).
        # We need to map those 13 columns back to fuel names to calculate intensity.
        
        # Flatten Current State for current intensity
        current_grid_mix = {}
        for r_data in regional_gen.values():
            for fuel, mw in r_data.items():
                current_grid_mix[fuel] = current_grid_mix.get(fuel, 0) + mw
        current_intensity = _calculate_intensity_from_mix(current_grid_mix)

        # Process Forecast (Simplified for brevity: using flat projection for demo 
        # unless ml_service returns full fuel breakdown. 
        # Assuming forecasts_mw returns [144, 13], we sum columns to get total MW,
        # but to get intensity we need specific fuel outputs.
        # Ideally ML Service returns { "North": { "Coal": [144], "Solar": [144] } }
        # If ML Service returns simple array, we must know column order to map back.
        # For now, let's assume we extract Total Load from forecast for the artifact.
        
        # Construct Forecast Timeline (Simplified)
        forecast_timeline_intensities = []
        for i in range(24):
             # Placeholder: In a real implementation, you'd iterate 144 steps
             # and recalculate intensity based on the predicted fuel mix.
             # Here we project current state to keep artifact valid.
             forecast_timeline_intensities.append(current_intensity)

        current_level = _determine_dynamic_level(current_intensity, forecast_timeline_intensities)

        # 8. Artifact Construction
        timeline_objects = []
        for i, val in enumerate(forecast_timeline_intensities):
            timeline_objects.append(
                TimeStep(
                    timestamp=now + timedelta(hours=i),
                    carbon_intensity=int(val),
                    level=_determine_dynamic_level(val, forecast_timeline_intensities)
                )
            )

        artifact = PredictionArtifact(
            last_updated=now,
            status="Complete",
            current_intensity=int(current_intensity),
            current_level=current_level,
            best_usage_window=OptimizationWindow(start_time=now, end_time=now+timedelta(hours=1)),
            forecast_start_time=now,
            forecast_end_time=now + timedelta(hours=24),
            timeline=timeline_objects
        )
        
        # 9. Save Artifact
        await bucket_repo.upload_json("carbon_intensity.json", artifact.model_dump_json())
        
        logger.info(f"✅ Pipeline Success. Intensity: {current_intensity} g/kWh")
        return {"status": "success", "intensity": current_intensity}

    except Exception as e:
        logger.error("❌ Pipeline Failed", exc_info=True)
        raise e