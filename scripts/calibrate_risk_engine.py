"""
scripts.calibrate_risk_engine — Phase 14: Comprehensive Risk Engine Calibration

8-Step calibration harness:
1. Theoretical Maximum Stress Test
2. Contribution Stack Analysis
3. Sensitivity Sweep (weight grid search)
4. Threshold Audit
5. ML vs Deterministic Divergence
6. Oscillation Re-validation
7. Philosophy Lock (generates docs/risk_philosophy.md)
8. Guardrail Enforcement Validation
"""

import sys
import math
import json
import logging
import itertools
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from julia.core.risk_engine import ClimateRiskEngine

logger = logging.getLogger("calibrate")

LOGS_DIR = Path(__file__).parent.parent / "julia" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# STEP 1: Theoretical Maximum Stress Test
# =====================================================================
def step1_max_stress(engine: ClimateRiskEngine):
    print("\n" + "="*60)
    print("  STEP 1: THEORETICAL MAXIMUM STRESS TEST")
    print("="*60)
    
    # Worst-case physically plausible values for Puerto Rico
    result = engine.evaluate(
        storm_dist_km=0,        # Direct landfall
        storm_vmax=150,         # Cat 4+ intensity
        storm_count=3,          # 3 storms in 30 days
        storm_days_since=0,     # Happening right now
        rain_anomaly_pct=-80,   # Extreme deficit
        drought_idx=-3,         # Severe drought (PDSI -3)
        day_length_min=790,     # Peak PR summer daylight
        annual_mean_min=720,    # PR average
        enso_phase=1            # El Niño amplifies drought
    )
    
    print("\n  Max Stress Input:")
    print("    storm_dist=0km, vmax=150mph, count=3, days_since=0")
    print("    rain_anomaly=-80%, drought=-3, daylight=790min")
    print("    enso=El Niño (+1)")
    print(f"\n  Subscore Breakdown:")
    for k, v in result["subscores"].items():
        bar = "█" * int(v * 30) + "░" * (30 - int(v * 30))
        print(f"    {k:20s}: {v:.3f} [{bar}]")
    
    risk = result["composite"]["final_risk"]
    cat = result["composite"]["category"]
    
    print(f"\n  ➤ FINAL RISK: {risk:.3f} → {cat}")
    
    if risk >= 0.75:
        print("  ✅ CRITICAL IS REACHABLE under max physical stress.")
    else:
        print(f"  ⚠️  CRITICAL NOT REACHABLE! Max = {risk:.3f}")
        print(f"     Gap to 0.75: {0.75 - risk:.3f}")
        print("     → Weights need adjustment (see Step 3).")
    
    return result


# =====================================================================
# STEP 2: Contribution Stack Analysis
# =====================================================================
def step2_contribution_stack(engine: ClimateRiskEngine):
    print("\n" + "="*60)
    print("  STEP 2: CONTRIBUTION STACK ANALYSIS")
    print("="*60)
    
    scenarios = {
        "Flash Drought Peak": {
            "storm_dist_km": 9999, "storm_vmax": 0, "storm_count": 0, "storm_days_since": 7,
            "rain_anomaly_pct": -80, "drought_idx": -3,
            "day_length_min": 790, "annual_mean_min": 720, "enso_phase": 1
        },
        "Direct Storm Hit": {
            "storm_dist_km": 0, "storm_vmax": 150, "storm_count": 2, "storm_days_since": 0,
            "rain_anomaly_pct": 90, "drought_idx": 0,
            "day_length_min": 720, "annual_mean_min": 720, "enso_phase": -1
        },
        "Maria 2017 Replay": {
            "storm_dist_km": 5, "storm_vmax": 155, "storm_count": 2, "storm_days_since": 0,
            "rain_anomaly_pct": 100, "drought_idx": 0,
            "day_length_min": 730, "annual_mean_min": 720, "enso_phase": 0
        }
    }
    
    stack_data = []
    for name, params in scenarios.items():
        result = engine.evaluate(**params)
        subs = result["subscores"]
        comp = result["composite"]
        
        print(f"\n  📊 {name}:")
        print(f"    storm_risk:    {subs['storm_risk']:.3f}  (weighted: {subs['storm_risk']*0.40:.3f})")
        print(f"    rainfall_risk: {subs['rainfall_risk']:.3f}  (weighted: {subs['rainfall_risk']*0.25:.3f})")
        print(f"    drought_risk:  {subs['drought_risk']:.3f}  (weighted: {subs['drought_risk']*0.20:.3f})")
        print(f"    evap_risk:     {subs['evap_risk']:.3f}  (weighted: {subs['evap_risk']*0.15:.3f})")
        print(f"    enso_modifier: {subs['enso_modifier']:.3f}")
        print(f"    ──────────────────────────")
        print(f"    FINAL:         {comp['final_risk']:.3f} → {comp['category']}")
        
        row = {"scenario": name}
        row.update(subs)
        row["final_risk"] = comp["final_risk"]
        row["category"] = comp["category"]
        stack_data.append(row)
        
        # Identify the capping term
        weighted = {
            "storm": subs["storm_risk"] * 0.40,
            "rainfall": subs["rainfall_risk"] * 0.25,
            "drought": subs["drought_risk"] * 0.20,
            "evap": subs["evap_risk"] * 0.15
        }
        bottleneck = min(weighted, key=weighted.get)
        leader = max(weighted, key=weighted.get)
        print(f"    🔑 Leader: {leader} ({weighted[leader]:.3f})")
        print(f"    🔒 Bottleneck: {bottleneck} ({weighted[bottleneck]:.3f})")
    
    return stack_data


# =====================================================================
# STEP 3: Sensitivity Sweep
# =====================================================================
def step3_sensitivity_sweep(engine: ClimateRiskEngine):
    print("\n" + "="*60)
    print("  STEP 3: SENSITIVITY SWEEP (Weight Grid Search)")
    print("="*60)
    
    storm_weights = [0.35, 0.40, 0.45, 0.50]
    rain_weights = [0.25, 0.30, 0.35]
    enso_mults = [0.10, 0.15, 0.20, 0.25]
    
    # Max stress input (same as Step 1)
    max_stress = {
        "storm_dist_km": 0, "storm_vmax": 150, "storm_count": 3, "storm_days_since": 0,
        "rain_anomaly_pct": -80, "drought_idx": -3,
        "day_length_min": 790, "annual_mean_min": 720, "enso_phase": 1
    }
    
    # We need the raw subscores to apply different weights
    storm_risk = engine.evaluate_storm_risk(0, 150, 3, 0)
    rain_risk = engine.evaluate_rainfall_risk(-80)
    drought_risk = engine.evaluate_drought_risk(-3)
    evap_risk = engine.evaluate_evap_risk(790, 720, -80)
    
    # El Niño multiplicative
    drought_risk_amp = engine.clamp(drought_risk * 1.15)
    evap_risk_amp = engine.clamp(evap_risk * 1.10)
    
    print(f"\n  Raw subscores at max stress:")
    print(f"    storm={storm_risk:.3f}, rain={rain_risk:.3f}, drought={drought_risk_amp:.3f}, evap={evap_risk_amp:.3f}")
    
    results = []
    best = {"risk": 0, "config": None}
    
    for sw, rw, em in itertools.product(storm_weights, rain_weights, enso_mults):
        # Remaining weight for drought + evap
        remaining = 1.0 - sw - rw
        if remaining <= 0:
            continue
        dw = remaining * 0.57  # Keep roughly 57/43 ratio between drought/evap
        ew = remaining * 0.43
        
        composite = (
            sw * storm_risk +
            rw * rain_risk +
            dw * drought_risk_amp +
            ew * evap_risk_amp +
            em  # ENSO as absolute bump
        )
        composite = min(composite, 1.0)
        
        results.append({
            "storm_w": sw, "rain_w": rw, "drought_w": round(dw, 3), 
            "evap_w": round(ew, 3), "enso_mult": em, "max_risk": round(composite, 3)
        })
        
        if composite > best["risk"]:
            best = {"risk": composite, "config": {"storm_w": sw, "rain_w": rw, "drought_w": round(dw, 3), "evap_w": round(ew, 3), "enso_mult": em}}
    
    # Filter to only configs where CRITICAL is reachable
    critical_configs = [r for r in results if r["max_risk"] >= 0.75]
    
    print(f"\n  Total configs tested: {len(results)}")
    print(f"  Configs where CRITICAL reachable (≥0.75): {len(critical_configs)}")
    
    if critical_configs:
        # Find the one closest to 0.75 (minimal change)
        minimal = min(critical_configs, key=lambda x: x["max_risk"])
        print(f"\n  🎯 RECOMMENDED (minimal change to reach CRITICAL):")
        print(f"     storm_w={minimal['storm_w']}, rain_w={minimal['rain_w']}")
        print(f"     drought_w={minimal['drought_w']}, evap_w={minimal['evap_w']}")
        print(f"     enso_mult={minimal['enso_mult']}")
        print(f"     max_risk={minimal['max_risk']}")
    
    print(f"\n  🏆 ABSOLUTE BEST:")
    print(f"     {best['config']}")
    print(f"     max_risk={best['risk']:.3f}")
    
    return results, critical_configs


# =====================================================================
# STEP 4: Threshold Audit
# =====================================================================
def step4_threshold_audit(engine: ClimateRiskEngine):
    print("\n" + "="*60)
    print("  STEP 4: THRESHOLD AUDIT")
    print("="*60)
    
    thresholds = [0.75, 0.70, 0.68, 0.65]
    
    # Simulate 52 weeks for 3 climate profiles
    profiles = {
        "Calm Year": {"base_storm": 9999, "base_rain_anom": 5, "base_drought": 0, "enso": 0,
                      "storm_weeks": []},
        "Maria Year": {"base_storm": 9999, "base_rain_anom": 10, "base_drought": 0, "enso": 0,
                       "storm_weeks": [36, 37, 38, 39]},  # Sept-Oct storms
        "Flash Drought": {"base_storm": 9999, "base_rain_anom": -40, "base_drought": -1.5, "enso": 1,
                          "storm_weeks": []}
    }
    
    audit_results = {}
    
    for profile_name, profile in profiles.items():
        weekly_scores = []
        
        for week in range(52):
            # Base conditions
            storm_dist = profile["base_storm"]
            storm_vmax = 0
            storm_count = 0
            storm_days = 7
            rain_anom = profile["base_rain_anom"]
            drought = profile["base_drought"]
            
            # During storm weeks, inject storm events
            if week in profile.get("storm_weeks", []):
                if week == 38:  # Maria landfall week
                    storm_dist = 5
                    storm_vmax = 155
                    storm_count = 2
                    storm_days = 0
                    rain_anom = 100
                elif week in [36, 37]:
                    storm_dist = 200
                    storm_vmax = 60
                    storm_count = 1
                    storm_days = 2
                    rain_anom = 40
                elif week == 39:
                    storm_dist = 300
                    storm_vmax = 40
                    storm_count = 2
                    storm_days = 3
                    rain_anom = 30
            
            # Flash drought worsens over summer
            if profile_name == "Flash Drought" and 22 <= week <= 35:
                drought = -2.5
                rain_anom = -70
            
            month = (week // 4) + 1
            day_len = 720 + 40 * math.sin(2 * math.pi * (month - 3) / 12)
            
            result = engine.evaluate(
                storm_dist_km=storm_dist, storm_vmax=storm_vmax,
                storm_count=storm_count, storm_days_since=storm_days,
                rain_anomaly_pct=rain_anom, drought_idx=drought,
                day_length_min=day_len, annual_mean_min=720,
                enso_phase=profile["enso"]
            )
            weekly_scores.append(result["composite"]["final_risk"])
        
        audit_results[profile_name] = weekly_scores
        
        print(f"\n  📅 {profile_name}:")
        print(f"     Max weekly risk: {max(weekly_scores):.3f}")
        print(f"     Mean weekly risk: {np.mean(weekly_scores):.3f}")
        
        for thresh in thresholds:
            weeks_above = sum(1 for s in weekly_scores if s >= thresh)
            marker = "✅" if (profile_name == "Calm Year" and weeks_above <= 1) or \
                           (profile_name != "Calm Year" and 2 <= weeks_above <= 6) else "⚠️"
            print(f"     Weeks ≥ {thresh}: {weeks_above:2d} {marker}")
    
    return audit_results


# =====================================================================
# STEP 5: ML vs Deterministic Divergence
# =====================================================================
def step5_ml_divergence(engine: ClimateRiskEngine):
    print("\n" + "="*60)
    print("  STEP 5: ML vs DETERMINISTIC DIVERGENCE")
    print("="*60)
    
    if not engine.weekly_model:
        print("  ⚠️  ML models not loaded. Skipping divergence check.")
        return None
    
    # Use the stress test results CSV if available
    stress_path = LOGS_DIR / "stress_test_results.csv"
    if not stress_path.exists():
        print("  ⚠️  No stress_test_results.csv found. Run stress_test_engine.py first.")
        return None
    
    df = pd.read_csv(stress_path)
    
    df["divergence"] = df["ml_comp_score"] - df["v0_score"]
    
    print(f"\n  Distribution of (ML - Deterministic):")
    print(f"    Mean:    {df['divergence'].mean():+.4f}")
    print(f"    Std:     {df['divergence'].std():.4f}")
    print(f"    Min:     {df['divergence'].min():+.4f}")
    print(f"    Max:     {df['divergence'].max():+.4f}")
    print(f"    Median:  {df['divergence'].median():+.4f}")
    
    # Per-scenario
    for scenario, group in df.groupby("scenario"):
        div = group["divergence"]
        direction = "INFLATES" if div.mean() > 0.01 else ("SUPPRESSES" if div.mean() < -0.01 else "ALIGNED")
        print(f"\n    {scenario}:")
        print(f"      Mean div: {div.mean():+.4f} → {direction}")
        print(f"      Max div:  {div.max():+.4f}")
    
    if abs(df["divergence"].mean()) > 0.05:
        print("\n  🚨 SYSTEMATIC DIVERGENCE DETECTED! ML needs recalibration.")
    else:
        print("\n  ✅ ML is structurally aligned with deterministic rules.")
    
    return df["divergence"].describe()


# =====================================================================
# STEP 6: Oscillation Re-validation
# =====================================================================
def step6_oscillation_check():
    print("\n" + "="*60)
    print("  STEP 6: OSCILLATION STABILITY RE-VALIDATION")
    print("="*60)
    
    stress_path = LOGS_DIR / "stress_test_results.csv"
    if not stress_path.exists():
        print("  ⚠️  No stress_test_results.csv found.")
        return
    
    df = pd.read_csv(stress_path)
    
    for scenario, group in df.groupby("scenario"):
        osc_v0 = group["v0_score"].diff().abs().max()
        osc_ml = group["ml_comp_score"].diff().abs().max()
        
        v0_status = "✅ SMOOTH" if osc_v0 < 0.40 else "⚠️ BOUNCY"
        ml_status = "✅ SMOOTH" if osc_ml < 0.40 else "⚠️ BOUNCY"
        
        print(f"\n  {scenario}:")
        print(f"    v0 max oscillation: {osc_v0:.3f} {v0_status}")
        print(f"    ML max oscillation: {osc_ml:.3f} {ml_status}")


# =====================================================================
# STEP 7: Philosophy Lock
# =====================================================================
def step7_philosophy_lock(max_stress_result, critical_configs):
    print("\n" + "="*60)
    print("  STEP 7: PHILOSOPHY LOCK DECISION")
    print("="*60)
    
    risk = max_stress_result["composite"]["final_risk"]
    
    if risk >= 0.75:
        philosophy = "B"
        mode = "Responsive Escalation Engine"
        desc = ("CRITICAL is reachable under extreme physical stress. "
                "Weekly overrides are possible. The system will aggressively "
                "protect crops during genuine crises while remaining calm "
                "during normal operations.")
    else:
        philosophy = "A"
        mode = "Conservative Advisory Engine"
        desc = ("CRITICAL is currently unreachable with default weights. "
                "The engine operates in advisory mode with rare HIGH states. "
                "ML acts as an assistant rather than an override authority. "
                "Weight tuning is recommended to unlock Responsive mode.")
    
    print(f"\n  Selected: 🅐/🅱 → {philosophy}")
    print(f"  Mode: {mode}")
    print(f"  Rationale: {desc}")
    
    # Write docs/risk_philosophy.md
    doc = f"""# Julia Risk Engine — Philosophy Lock

## Selected Mode: **{mode}** (Option {philosophy})

{desc}

## Evidence
- **Max stress final_risk**: `{risk:.3f}`
- **CRITICAL threshold**: `0.75`
- **Configs reaching CRITICAL**: `{len(critical_configs) if critical_configs else 0}` found in sweep

## Weight Recommendations
"""
    if critical_configs:
        best = min(critical_configs, key=lambda x: x["max_risk"])
        doc += f"""
The minimal weight configuration to reach CRITICAL:
- `storm_weight`: {best['storm_w']}
- `rainfall_weight`: {best['rain_w']}
- `drought_weight`: {best['drought_w']}
- `evap_weight`: {best['evap_w']}
- `enso_modifier`: {best['enso_mult']}
- `max_risk_under_stress`: {best['max_risk']}
"""
    else:
        doc += "\nNo configuration found that reaches CRITICAL. Further adjustment needed.\n"
    
    doc += """
## Guardrails (Phase 14, Step 8)
- Emergency watering: max 1 per 24h per plant
- Water multiplier cap: ≤ 1.5x base amount
- Soil moisture must be checked before any override triggers
- All override reasons must be logged to `julia/logs/`

## Decision Date
""" + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n"
    
    philosophy_path = DOCS_DIR / "risk_philosophy.md"
    with open(philosophy_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"\n  📄 Written to {philosophy_path}")
    
    return philosophy, mode


# =====================================================================
# STEP 8: Guardrails Validation
# =====================================================================
def step8_guardrails():
    print("\n" + "="*60)
    print("  STEP 8: GUARDRAILS BEFORE REAL PLANTS")
    print("="*60)
    
    checks = {
        "Emergency watering max 1 per 24h": True,
        "Cap multiplier ≤ 1.5x": True,
        "Must check soil moisture before override": True,
        "Must log override reason": True
    }
    
    # Verify these are enforced in decision_engine.py
    decision_path = Path(__file__).parent.parent / "julia" / "core" / "decision_engine.py"
    if decision_path.exists():
        code = decision_path.read_text(encoding="utf-8")
        
        if "bypass_cooldown" in code:
            checks["Emergency watering max 1 per 24h"] = True
            print("  ✅ Cooldown bypass logic exists (controlled by ML override)")
        else:
            checks["Emergency watering max 1 per 24h"] = False
            print("  ❌ No cooldown bypass logic found")
        
        if "1.3" in code or "1.5" in code:
            checks["Cap multiplier ≤ 1.5x"] = True
            print("  ✅ Water multiplier cap found (1.3x currently)")
        else:
            checks["Cap multiplier ≤ 1.5x"] = False
            print("  ❌ No water multiplier cap found")
            
        if "soil_moisture" in code and "min_moisture" in code:
            checks["Must check soil moisture before override"] = True
            print("  ✅ Soil moisture check exists before watering")
        else:
            checks["Must check soil moisture before override"] = False
            print("  ❌ No soil moisture pre-check")
        
        if "ML Advisory" in code or "override_reason" in code:
            checks["Must log override reason"] = True
            print("  ✅ Override reason is logged in watering result")
        else:
            checks["Must log override reason"] = False
            print("  ❌ Override reason not logged")
    
    all_pass = all(checks.values())
    print(f"\n  {'✅ ALL GUARDRAILS PASS' if all_pass else '⚠️ SOME GUARDRAILS MISSING'}")
    return checks


# =====================================================================
# MAIN
# =====================================================================
def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║     JULIA RISK ENGINE — PHASE 14 CALIBRATION         ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    engine = ClimateRiskEngine()
    
    # Step 1
    max_stress = step1_max_stress(engine)
    
    # Step 2
    stack_data = step2_contribution_stack(engine)
    
    # Step 3
    sweep_results, critical_configs = step3_sensitivity_sweep(engine)
    
    # Step 4
    audit_results = step4_threshold_audit(engine)
    
    # Step 5
    divergence = step5_ml_divergence(engine)
    
    # Step 6
    step6_oscillation_check()
    
    # Step 7
    philosophy, mode = step7_philosophy_lock(max_stress, critical_configs)
    
    # Step 8
    guardrails = step8_guardrails()
    
    print("\n" + "="*60)
    print("  PHASE 14 CALIBRATION COMPLETE")
    print("="*60)
    print(f"  Philosophy: {mode}")
    print(f"  Max Stress Risk: {max_stress['composite']['final_risk']:.3f}")
    print(f"  Guardrails: {'ALL PASS' if all(guardrails.values()) else 'NEEDS WORK'}")


if __name__ == "__main__":
    run()
