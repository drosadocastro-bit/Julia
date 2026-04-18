# Julia Risk Engine — Philosophy Lock

## Selected Mode: **Responsive Escalation Engine** (Option B)

CRITICAL is reachable under extreme physical stress. Weekly overrides are possible. The system will aggressively protect crops during genuine crises while remaining calm during normal operations.

## Evidence
- **Max stress final_risk**: `1.000`
- **CRITICAL threshold**: `0.75`
- **Configs reaching CRITICAL**: `48` found in sweep

## Weight Recommendations

The minimal weight configuration to reach CRITICAL:
- `storm_weight`: 0.35
- `rainfall_weight`: 0.25
- `drought_weight`: 0.228
- `evap_weight`: 0.172
- `enso_modifier`: 0.1
- `max_risk_under_stress`: 1.0

## Guardrails (Phase 14, Step 8)
- Emergency watering: max 1 per 24h per plant
- Water multiplier cap: ≤ 1.5x base amount
- Soil moisture must be checked before any override triggers
- All override reasons must be logged to `julia/logs/`

## Decision Date
2026-03-02 13:18
