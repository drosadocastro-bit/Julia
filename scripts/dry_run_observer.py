"""
scripts.dry_run_observer — Phase 15: 30-Day Shadow Mode Dry Run

Fetches LIVE Puerto Rico weather from the NWS API (free, no key needed),
runs both v0 (deterministic) and v1 (ML) risk engines, calculates water
multipliers, and logs everything — WITHOUT actually watering.

Designed to run once per day (e.g., via Task Scheduler or cron).
Appends results to julia/logs/dry_run_log.csv.

Usage:
    python scripts/dry_run_observer.py           # Single daily observation
    python scripts/dry_run_observer.py --loop 30  # Continuous 30-day loop (testing)
"""

import sys
import json
import logging
import time
import csv
import math
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

import requests

sys.path.append(str(Path(__file__).parent.parent))
from julia.core.risk_engine import ClimateRiskEngine

logger = logging.getLogger("dry_run")

LOGS_DIR = Path(__file__).parent.parent / "julia" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "dry_run_log.csv"

# NWS API Configuration for San Juan, PR
# NWS gridpoint for San Juan area
NWS_STATION = "SJU"  # San Juan Weather Forecast Office
NWS_GRID_X = 33
NWS_GRID_Y = 34
NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {"User-Agent": "(Julia CropTaker, julia@croptaker.dev)", "Accept": "application/geo+json"}

# Puerto Rico annual daylight average (minutes)
PR_ANNUAL_MEAN_DAYLIGHT = 720


def fetch_nws_observation() -> Optional[Dict]:
    """
    Fetch the latest weather observation from the NWS API.
    Uses the San Juan, PR station.
    Returns a dict with temperature, humidity, precipitation, wind, etc.
    """
    try:
        # Fetch latest observations from SJU station
        url = f"{NWS_BASE}/stations/TJSJ/observations/latest"
        resp = requests.get(url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        props = data.get("properties", {})
        
        # Extract values (NWS returns in metric by default)
        temp_c = props.get("temperature", {}).get("value")
        humidity = props.get("relativeHumidity", {}).get("value")
        wind_speed = props.get("windSpeed", {}).get("value")  # m/s
        precip_1h = props.get("precipitationLastHour", {}).get("value")  # mm
        precip_6h = props.get("precipitationLast6Hours", {}).get("value")
        description = props.get("textDescription", "Unknown")
        timestamp = props.get("timestamp", "")
        
        # Handle null values gracefully
        temp_c = temp_c if temp_c is not None else 28.0
        humidity = humidity if humidity is not None else 75.0
        precip_1h = precip_1h if precip_1h is not None else 0.0
        precip_6h = precip_6h if precip_6h is not None else 0.0
        
        logger.info(f"  🌤️ NWS Live: {description} | {temp_c:.1f}°C | Humidity {humidity:.0f}% | Precip 1h: {precip_1h}mm")
        
        return {
            "temp_c": temp_c,
            "humidity": humidity,
            "wind_speed_ms": wind_speed or 0,
            "precip_1h_mm": precip_1h,
            "precip_6h_mm": precip_6h,
            "description": description,
            "timestamp": timestamp
        }
        
    except Exception as e:
        logger.warning(f"  ⚠️ NWS API failed: {e}. Using PR default fallback.")
        return {
            "temp_c": 28.0,
            "humidity": 75.0,
            "wind_speed_ms": 3.0,
            "precip_1h_mm": 0.0,
            "precip_6h_mm": 0.0,
            "description": "API Unavailable - PR Default",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def fetch_nws_forecast() -> Dict:
    """
    Fetch the 7-day forecast from NWS to estimate rain probability.
    """
    try:
        url = f"{NWS_BASE}/gridpoints/{NWS_STATION}/{NWS_GRID_X},{NWS_GRID_Y}/forecast"
        resp = requests.get(url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        periods = data.get("properties", {}).get("periods", [])
        
        if not periods:
            return {"rain_chance_today": 0, "rain_chance_week": 0, "forecast_desc": "N/A"}
        
        # First period is today
        today = periods[0]
        rain_today = today.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        
        # Average across all periods for weekly
        rain_chances = [p.get("probabilityOfPrecipitation", {}).get("value", 0) or 0 for p in periods]
        rain_week = sum(rain_chances) / len(rain_chances) if rain_chances else 0
        
        return {
            "rain_chance_today": rain_today,
            "rain_chance_week": rain_week,
            "forecast_desc": today.get("shortForecast", "Unknown")
        }
    except Exception as e:
        logger.warning(f"  ⚠️ NWS Forecast API failed: {e}")
        return {"rain_chance_today": 0, "rain_chance_week": 0, "forecast_desc": "Unavailable"}


def estimate_day_length(date: datetime) -> float:
    """Estimate daylight minutes for PR based on day of year."""
    doy = date.timetuple().tm_yday
    # PR is ~18°N latitude. Day length varies ~660-780 min across the year.
    return 720 + 40 * math.sin(2 * math.pi * (doy - 80) / 365)


def build_rolling_state(log_file: Path) -> Dict:
    """
    Read the existing dry_run_log.csv to compute rolling 7d/30d stats.
    """
    rain_7d = 0.0
    rain_30d = 0.0
    
    if not log_file.exists():
        return {"rain_7d": 0, "rain_30d": 0, "days_logged": 0}
    
    try:
        df_rows = []
        with open(log_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                df_rows.append(row)
        
        # Last 7 days
        recent_7 = df_rows[-7:] if len(df_rows) >= 7 else df_rows
        rain_7d = sum(float(r.get("precip_mm", 0)) for r in recent_7)
        
        # Last 30 days
        recent_30 = df_rows[-30:] if len(df_rows) >= 30 else df_rows
        rain_30d = sum(float(r.get("precip_mm", 0)) for r in recent_30)
        
        return {"rain_7d": rain_7d, "rain_30d": rain_30d, "days_logged": len(df_rows)}
    except Exception:
        return {"rain_7d": 0, "rain_30d": 0, "days_logged": 0}


def run_single_observation(engine: ClimateRiskEngine):
    """Run a single daily dry-run observation cycle."""
    now = datetime.now(timezone.utc)
    logger.info(f"\n{'='*50}")
    logger.info(f"  JULIA DRY RUN — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info(f"{'='*50}")
    
    # 1. Fetch live weather
    logger.info("\n  📡 Fetching live PR weather from NWS API...")
    obs = fetch_nws_observation()
    forecast = fetch_nws_forecast()
    
    # 2. Build rolling state from previous logs
    rolling = build_rolling_state(LOG_FILE)
    logger.info(f"  📊 Rolling state: {rolling['days_logged']} days logged, rain_7d={rolling['rain_7d']:.1f}mm, rain_30d={rolling['rain_30d']:.1f}mm")
    
    # 3. Estimate features
    day_len = estimate_day_length(now)
    month = now.month
    
    # Rainfall anomaly: compare rolling 30d to PR normal (~120mm/month)
    normal_30d = 120.0
    rain_total_30d = rolling["rain_30d"] + obs["precip_6h_mm"]
    rain_anomaly_pct = ((rain_total_30d - normal_30d) / normal_30d) * 100 if normal_30d > 0 else 0
    
    # Simple drought proxy from rain anomaly
    drought_idx = 0.0
    if rain_anomaly_pct < -30:
        drought_idx = -1.0
    if rain_anomaly_pct < -50:
        drought_idx = -2.0
    if rain_anomaly_pct < -70:
        drought_idx = -3.0
    
    hurricane_flag = 1 if month in [6,7,8,9,10,11] else 0
    dry_flag = 1 if month in [12,1,2,3,4] else 0
    heat_flag = 1 if obs["temp_c"] > 32 else 0
    
    # 4. Run v0 Deterministic Engine
    v0_result = engine.evaluate(
        storm_dist_km=9999,     # No storm tracking in dry run (no HURDAT real-time)
        storm_vmax=0,
        storm_count=0,
        storm_days_since=7,
        rain_anomaly_pct=rain_anomaly_pct,
        drought_idx=drought_idx,
        day_length_min=day_len,
        annual_mean_min=PR_ANNUAL_MEAN_DAYLIGHT,
        enso_phase=0            # Default neutral (could fetch from CPC later)
    )
    
    v0_risk = v0_result["composite"]["final_risk"]
    v0_cat = v0_result["composite"]["category"]
    
    logger.info(f"\n  🔬 v0 Deterministic: {v0_risk:.3f} → {v0_cat}")
    for k, v in v0_result["subscores"].items():
        logger.info(f"     {k}: {v:.3f}")
    
    # 5. Run v1 ML Engine
    ml_payload = {
        "hurricane_season_flag": hurricane_flag,
        "dry_season_flag": dry_flag,
        "day_length_minutes": day_len,
        "rainfall_last_7_days": rolling["rain_7d"] + obs["precip_1h_mm"],
        "rainfall_last_30_days": rain_total_30d,
        "rainfall_anomaly_percent": rain_anomaly_pct,
        "TMAX": obs["temp_c"],
        "heat_stress_flag": heat_flag,
        "drought_index": drought_idx,
        "drought_severity_flag": 1 if drought_idx <= -2 else 0,
        "enso_phase_encoded": 0,
        "enso_strength": 0,
        "min_distance_to_PR_last_7d": 9999,
        "storm_vmax": 0,
        "storm_count_last_30_days": 0
    }
    
    v1_result = engine.evaluate_v1(ml_payload)
    ml_week = v1_result["horizons"]["weekly_risk_score"]
    ml_month = v1_result["horizons"]["monthly_risk_score"]
    ml_comp = v1_result["composite"]["final_risk"]
    ml_cat = v1_result["composite"]["category"]
    advisory = v1_result["composite"]["final_advisory"]
    
    logger.info(f"\n  🧠 v1 ML Engine:")
    logger.info(f"     Weekly:  {ml_week:.3f} → {v1_result['horizons']['weekly_category']}")
    logger.info(f"     Monthly: {ml_month:.3f} → {v1_result['horizons']['monthly_category']}")
    logger.info(f"     Composite: {ml_comp:.3f} → {ml_cat}")
    logger.info(f"     Advisory: {advisory}")
    
    # 6. Calculate water multiplier (but DON'T water)
    base_water_ml = 200  # Standard basil watering
    multiplier = 1.0
    if ml_cat in ["HIGH", "CRITICAL"]:
        multiplier = 1.3
    if obs["temp_c"] > 30:
        hot_factor = 1.0 + (obs["temp_c"] - 30) * 0.04
        hot_factor = min(hot_factor, 1.3)
        multiplier *= hot_factor
    
    calculated_ml = int(base_water_ml * multiplier)
    
    logger.info(f"\n  💧 SHADOW WATERING CALCULATION (NOT EXECUTED):")
    logger.info(f"     Base: {base_water_ml}ml × {multiplier:.2f} = {calculated_ml}ml")
    logger.info(f"     Would water: {'YES' if v0_risk > 0.25 or ml_comp > 0.33 else 'NO'}")
    
    # 7. Append to CSV log
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "date", "time_utc", "temp_c", "humidity", "precip_mm", 
                "description", "rain_anomaly_pct", "drought_idx",
                "v0_risk", "v0_category",
                "ml_weekly", "ml_monthly", "ml_composite", "ml_category", "ml_advisory",
                "water_multiplier", "calculated_ml", "would_water",
                "rain_7d", "rain_30d", "day_length_min"
            ])
        writer.writerow([
            now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
            round(obs["temp_c"], 1), round(obs["humidity"], 0), round(obs["precip_6h_mm"], 1),
            obs["description"], round(rain_anomaly_pct, 1), drought_idx,
            v0_risk, v0_cat,
            ml_week, ml_month, ml_comp, ml_cat, advisory,
            round(multiplier, 2), calculated_ml,
            "YES" if v0_risk > 0.25 or ml_comp > 0.33 else "NO",
            round(rolling["rain_7d"], 1), round(rain_total_30d, 1), round(day_len, 0)
        ])
    
    logger.info(f"\n  📝 Logged to {LOG_FILE}")
    logger.info(f"  📅 Day {rolling['days_logged'] + 1} of 30-day observation window")
    
    remaining = 30 - (rolling["days_logged"] + 1)
    if remaining > 0:
        logger.info(f"  ⏳ {remaining} days remaining in dry run")
    else:
        logger.info(f"  🎉 30-DAY DRY RUN COMPLETE! Review {LOG_FILE} for baseline behavior.")
    
    # 8. Rolling 7-Day Temperament Report (unlocks after 7 days)
    if rolling["days_logged"] + 1 >= 7:
        _print_temperament_report()
    
    return {
        "v0_risk": v0_risk, "ml_composite": ml_comp, 
        "advisory": advisory, "would_water": v0_risk > 0.25 or ml_comp > 0.33
    }


def _print_temperament_report():
    """Rolling 7-day temperament analysis — auto-generates after day 7."""
    try:
        rows = []
        with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        
        if len(rows) < 7:
            return
        
        last_7 = rows[-7:]
        
        # Extract scores
        v0_scores = [float(r["v0_risk"]) for r in last_7]
        ml_scores = [float(r["ml_composite"]) for r in last_7]
        
        # Compute stats
        import statistics
        v0_mean = statistics.mean(v0_scores)
        v0_var = statistics.variance(v0_scores) if len(v0_scores) > 1 else 0
        ml_mean = statistics.mean(ml_scores)
        ml_var = statistics.variance(ml_scores) if len(ml_scores) > 1 else 0
        
        # Classify temperament based on variance
        if ml_var < 0.001:
            temperament = "🧘 ZEN"
            desc = "Rock-solid. Almost no variation. Julia is completely calm."
        elif ml_var < 0.005:
            temperament = "😌 STEADY"
            desc = "Mild, natural fluctuations. Healthy baseline behavior."
        elif ml_var < 0.02:
            temperament = "😟 NERVOUS"
            desc = "Noticeable swings. Monitor for overcorrection patterns."
        else:
            temperament = "😰 ANXIOUS"
            desc = "High variance! Risk scores bouncing significantly. Investigate."
        
        logger.info(f"\n  {'─'*45}")
        logger.info(f"  📊 ROLLING 7-DAY TEMPERAMENT REPORT")
        logger.info(f"  {'─'*45}")
        logger.info(f"  v0 Deterministic:")
        logger.info(f"    Mean Risk:  {v0_mean:.4f}")
        logger.info(f"    Variance:   {v0_var:.6f}")
        logger.info(f"  ML Composite:")
        logger.info(f"    Mean Risk:  {ml_mean:.4f}")
        logger.info(f"    Variance:   {ml_var:.6f}")
        logger.info(f"  Temperament:  {temperament}")
        logger.info(f"    {desc}")
        
        # Trend direction
        if len(ml_scores) >= 3:
            first_half = statistics.mean(ml_scores[:3])
            second_half = statistics.mean(ml_scores[-3:])
            delta = second_half - first_half
            if delta > 0.02:
                logger.info(f"  Trend: 📈 RISING (+{delta:.3f})")
            elif delta < -0.02:
                logger.info(f"  Trend: 📉 FALLING ({delta:.3f})")
            else:
                logger.info(f"  Trend: ➡️  FLAT ({delta:+.3f})")
        
    except Exception as e:
        logger.warning(f"  ⚠️ Temperament report failed: {e}")


def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Julia Dry Run Observer")
    parser.add_argument("--loop", type=int, default=0, 
                        help="Number of simulated days to run in rapid succession (for testing)")
    args = parser.parse_args()
    
    engine = ClimateRiskEngine()
    
    if args.loop > 0:
        logger.info(f"🔄 Running {args.loop} rapid observation cycles (test mode)...")
        for i in range(args.loop):
            logger.info(f"\n{'━'*50}")
            logger.info(f"  TEST CYCLE {i+1}/{args.loop}")
            run_single_observation(engine)
            if i < args.loop - 1:
                time.sleep(2)  # Small delay between test cycles
    else:
        # Single daily observation
        run_single_observation(engine)


if __name__ == "__main__":
    run()
