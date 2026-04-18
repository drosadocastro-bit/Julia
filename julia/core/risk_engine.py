"""
julia.core.risk_engine — Deterministic Climate Risk Score Engine v0

Computes a synthetic 0-1 risk score indicating the probability that
crop stress or loss will increase based on macro-climate factors:
storm impact, rainfall anomalies, drought, solar stress, and ENSO state.
"""

import math
import pickle
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger("risk_engine")

class ClimateRiskEngine:
    """
    Evaluates current and historical climate data to produce a risk score.
    v0: Deterministic ground-truth.
    v1: Dual-Horizon ML models (Weekly/Monthly) with Override Logic.
    """

    CATEGORIES = {
        "LOW": (0.00, 0.33),
        "MODERATE": (0.33, 0.66),
        "HIGH": (0.66, 0.85),
        "CRITICAL": (0.85, 1.00)
    }
    
    # Keep v0 categories for backward compatibility
    CATEGORIES_V0 = {
        "LOW": (0.00, 0.25),
        "MODERATE": (0.25, 0.50),
        "HIGH": (0.50, 0.75),
        "CRITICAL": (0.75, 1.00)
    }

    def __init__(self):
        self.weekly_model = None
        self.monthly_model = None
        self.features_schema = []
        self._load_v1_models()

    def _load_v1_models(self):
        """Attempt to load v1 ML models if available."""
        models_dir = Path(__file__).parent.parent / "models"
        weekly_path = models_dir / "weekly_model.pkl"
        monthly_path = models_dir / "monthly_model.pkl"
        schema_path = models_dir / "feature_schema.json"
        
        try:
            import json
            if schema_path.exists():
                with open(schema_path, "r") as f:
                    schema = json.load(f)
                    self.features_schema = schema.get("features", [])
                    
            if weekly_path.exists() and monthly_path.exists():
                with open(weekly_path, "rb") as f:
                    self.weekly_model = pickle.load(f)
                with open(monthly_path, "rb") as f:
                    self.monthly_model = pickle.load(f)
                logger.info("Successfully loaded v1 ML Dual-Horizon Risk Models.")
        except Exception as e:
            logger.warning(f"Could not load v1 ML models. Operating in pure v0 deterministic mode. Error: {e}")

    @staticmethod
    def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp a value between min_val and max_val."""
        return max(min_val, min(val, max_val))

    def evaluate_storm_risk(self, distance_km: float, vmax: float, storm_count: int, days_since: float) -> float:
        """
        Calculate storm threat using exponential decay for distance and linear recency.
        
        Args:
            distance_km: Distance to closest storm in last 7 days. Use high value (e.g. 9999) if none.
            vmax: Maximum sustained wind speed in mph (intensity). Use 0 if none.
            storm_count: Number of storms in last 30 days.
            days_since: Days since closest storm pass (0-7). Use 7 if none.
        """
        if distance_km >= 1000:
            return 0.0

        # Exponential decay for distance: e^(-d / 150)
        # 0km -> 1.0, 150km -> ~0.37, 300km -> ~0.14
        distance_factor = math.exp(-distance_km / 150.0)
        
        # Intensity factor (clamp at Category 4 ~130-150mph)
        intensity_factor = self.clamp(vmax / 150.0)
        
        # Frequency factor (3+ storms in a month is max risk)
        frequency_factor = self.clamp(storm_count / 3.0)
        
        # Recency decay (7 days max memory for acute storm impact)
        recency_factor = self.clamp(1.0 - (days_since / 7.0))
        
        base_risk = (0.5 * distance_factor) + (0.3 * intensity_factor) + (0.2 * frequency_factor)
        
        return self.clamp(base_risk * recency_factor)

    def evaluate_rainfall_risk(self, anomaly_percent: float, _recent_rainfall_mm: float = 0) -> float:
        """
        Calculate risk from anomalous rainfall (both deficit and excess).
        
        Args:
            anomaly_percent: Percentage deviation from normal (-100 to +100 or more).
                             Negative = deficit, Positive = excess.
            _recent_rainfall_mm: Unused in v0 but reserved for absolute checks (e.g., flooding).
        """
        if anomaly_percent < 0:
            # Deficit: 50% below normal is max risk
            deficit_factor = self.clamp(abs(anomaly_percent) / 50.0)
            return deficit_factor
        else:
            # Excess: 70% above normal is max risk
            flood_factor = self.clamp(anomaly_percent / 70.0)
            return flood_factor

    def evaluate_drought_risk(self, drought_index: float) -> float:
        """
        Continuous mapping of a drought index (e.g., PDSI, SPEI).
        Assume index: 0 = normal, negative = dry, -3 = severe drought.
        """
        if drought_index >= 0:
            return 0.0
            
        # Smooth continuous mapping: index 0 -> 0 risk, index -3 -> 1.0 risk
        return self.clamp(abs(min(drought_index, 0)) / 3.0)

    def evaluate_evap_risk(self, day_length_minutes: float, annual_mean_minutes: float, rainfall_anomaly_percent: float) -> float:
        """
        Solar and evaporation stress based on daylight hours and rainfall deficit.
        """
        # Solar radiation risk (summer stress)
        solar_factor = self.clamp((day_length_minutes - annual_mean_minutes) / 120.0)
        
        # Evaporation is only a severe risk if there's a rain deficit
        if rainfall_anomaly_percent < 0:
            deficit_factor = self.clamp(abs(rainfall_anomaly_percent) / 50.0)
        else:
            deficit_factor = 0.0
            
        return self.clamp(solar_factor * deficit_factor)

    def get_enso_modifier(self, phase_val: int) -> float:
        """
        Return the base modifier value.
        NOTE: Real ENSO interactions are multiplicative with specific subscores in v0.
        
        Args:
            phase_val: -1 (La Niña), 0 (Neutral), 1 (El Niño)
        """
        if phase_val == -1:
            return 0.15  # La Niña tends to bring more storms/flooding to PR
        elif phase_val == 1:
            return 0.10  # El Niño brings more drought
        return 0.0

    def get_category_v0(self, risk_score: float) -> str:
        category = "LOW"
        for cat, (lower, upper) in self.CATEGORIES_V0.items():
            if lower <= risk_score <= upper:
                category = cat
                break
        if risk_score > 1.0: category = "CRITICAL"
        return category

    def get_category_v1(self, risk_score: float) -> str:
        category = "LOW"
        for cat, (lower, upper) in self.CATEGORIES.items():
            if lower <= risk_score <= upper:
                category = cat
                break
        if risk_score > 1.0: category = "CRITICAL"
        return category

    def evaluate(self, 
                 storm_dist_km: float = 9999, storm_vmax: float = 0, storm_count: int = 0, storm_days_since: float = 7,
                 rain_anomaly_pct: float = 0, 
                 drought_idx: float = 0, 
                 day_length_min: float = 720, annual_mean_min: float = 720,
                 enso_phase: int = 0) -> Dict[str, Any]:
        """
        [v0 engine] Compute the final composite risk score and generate a breakdown log.
        """
        # 1. Base Subscores
        storm_risk = self.evaluate_storm_risk(storm_dist_km, storm_vmax, storm_count, storm_days_since)
        rain_risk = self.evaluate_rainfall_risk(rain_anomaly_pct)
        drought_risk = self.evaluate_drought_risk(drought_idx)
        evap_risk = self.evaluate_evap_risk(day_length_min, annual_mean_min, rain_anomaly_pct)
        
        # 2. Multiplicative ENSO Interactions
        if enso_phase == -1:
            storm_risk = self.clamp(storm_risk * 1.15)
            rain_risk = self.clamp(rain_risk * 1.10)
        elif enso_phase == 1:
            drought_risk = self.clamp(drought_risk * 1.15)
            evap_risk = self.clamp(evap_risk * 1.10)
            
        enso_mod_base = self.get_enso_modifier(enso_phase)

        # 3. Weighted Composite
        final_risk = (
            0.40 * storm_risk +
            0.25 * rain_risk +
            0.20 * drought_risk +
            0.15 * evap_risk
        )
        final_risk = self.clamp(final_risk + enso_mod_base)
        
        return {
            "subscores": {
                "storm_risk": round(storm_risk, 3),
                "rainfall_risk": round(rain_risk, 3),
                "drought_risk": round(drought_risk, 3),
                "evap_risk": round(evap_risk, 3),
                "enso_modifier": round(enso_mod_base, 3)
            },
            "composite": {
                "final_risk": round(final_risk, 3),
                "category": self.get_category_v0(final_risk)
            }
        }

    def evaluate_v1(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        [v1 engine] Perform dual-horizon ML prediction.
        Expects a dictionary matching the ML_FEATURES schema.
        """
        if not self.weekly_model or not self.monthly_model:
            raise RuntimeError("ML Models not loaded. Run training script first or fallback to v0.")
            
        # Prepare DataFrame strictly matching the schema
        input_data = {feat: [feature_dict.get(feat, 0.0)] for feat in self.features_schema}
        df_input = pd.DataFrame(input_data)
        
        # Predict Probabilities (Risk Scores 0-1)
        # Using [0] since we only predict for one row
        weekly_prob = self.clamp(float(self.weekly_model.predict(df_input)[0]))
        monthly_prob = self.clamp(float(self.monthly_model.predict(df_input)[0]))
        
        # Categorize
        weekly_cat = self.get_category_v1(weekly_prob)
        monthly_cat = self.get_category_v1(monthly_prob)
        
        # OVERRIDE LOGIC
        # If weekly high -> override monthly
        # If monthly high but weekly low -> advisory mode
        final_advisory = "NORMAL_OPERATIONS"
        override_reason = "None"
        
        if weekly_cat in ["HIGH", "CRITICAL"]:
            final_advisory = "WEEKLY_CRITICAL_OVERRIDE"
            override_reason = "Acute short-term risk detected (e.g., incoming storm/heatwave). Override long-range plans."
        elif monthly_cat in ["HIGH", "CRITICAL"] and weekly_cat in ["LOW", "MODERATE"]:
            final_advisory = "MONTHLY_ADVISORY"
            override_reason = "Long-range macro climate shows elevated risk (e.g. developing drought/ENSO). Prepare resources but no immediate acute danger."
        elif weekly_cat == "MODERATE" or monthly_cat == "MODERATE":
            final_advisory = "ELEVATED_WATCH"
            override_reason = "Moderate risk on horizon."
            
        # Overall risk favors the worst-case horizon to be safe
        overall_max_risk = max(weekly_prob, monthly_prob)
        
        return {
            "horizons": {
                "weekly_risk_score": round(weekly_prob, 3),
                "weekly_category": weekly_cat,
                "monthly_risk_score": round(monthly_prob, 3),
                "monthly_category": monthly_cat
            },
            "composite": {
                "final_risk": round(overall_max_risk, 3),
                "category": self.get_category_v1(overall_max_risk),
                "final_advisory": final_advisory,
                "override_reason": override_reason
            }
        }
