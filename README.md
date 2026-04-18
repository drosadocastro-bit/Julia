# 🌺 JULIA — AI Crop Caretaker

> *"En honor a mi abuela Julia, que amaba las flores y le gustaba plantar"*
>
> *In honor of my grandmother Julia, who loved flowers and enjoyed planting* 🌺

---

**Julia** is an AI-powered crop caretaker system designed for home gardeners in Puerto Rico. She combines soil sensors, weather intelligence, computer vision, and an agentic AI loop to monitor your garden, make intelligent watering decisions, and learn from her mistakes — all while running on edge hardware.

This is not a cloud service. Julia runs **offline-first** on an NVIDIA Jetson Orin Nano, respects your data, and explains every decision she makes.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Smart Watering** | Priority-based decision engine evaluates soil moisture, weather forecasts, plant health, and climate risk before every action |
| **Agentic OODA-L Loop** | Observe → Orient → Decide → Act → **Learn** — Julia doesn't just react, she adapts |
| **Climate Risk Engine** | Dual-horizon (weekly + monthly) ML models evaluate storm, drought, rainfall, and ENSO signals |
| **3-Strike Learning** | When Julia makes the same mistake 3 times under identical conditions, she permanently rewrites her behavior |
| **Caring Invariants** | 5 unbreakable safety rules checked on every output — Julia always explains *why*, offers a monitoring signal, prefers reversible actions, asks for help under uncertainty, and respects your constraints |
| **Empathy Layer** | Emotional state machine (NORMAL → SUPPORTIVE → STORM_PREP → RECOVERY) adjusts communication tone and action conservatism |
| **LLM Brain** | Conversational assistant powered by Qwen3 8B (via LM Studio) with live sensor context, episodic memory, and mistake awareness |
| **Garden Simulator** | Full physics-based simulation of Puerto Rico weather, soil moisture, evaporation, and plant health for testing without hardware |
| **Live Dashboard** | Real-time glassmorphic web UI with Chart.js visualizations, camera feed, and manual controls |
| **Puerto Rico Native** | Plant profiles, weather defaults, pest awareness (🦎 iguanas!), and hurricane season logic tuned for tropical gardening |

## 🏗️ Architecture

```
                        ┌─────────────────────────────┐
                        │      SENSOR LAYER            │
                        │  Haozee Zigbee × 3 + Camera  │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │     COMMUNICATION LAYER      │
                        │  Zigbee2MQTT ↔ Home Assist.  │
                        └─────────────┬───────────────┘
                                      │
        ┌────────────────────────────▼───────────────────────────────┐
        │               JETSON ORIN NANO — "Julia's Brain"           │
        │                                                            │
        │   ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐  │
        │   │ Decision     │  │ Climate     │  │ Agentic Layer   │  │
        │   │ Engine       │  │ Risk Engine │  │ (OODA-L Loop)   │  │
        │   │              │  │ v0 + v1 ML  │  │                 │  │
        │   │ 7-priority   │  │             │  │ • Perception    │  │
        │   │ rule system  │  │ • Storm     │  │ • Caring Layer  │  │
        │   │ + ML edge    │  │ • Drought   │  │ • Planner       │  │
        │   │   cases      │  │ • Rainfall  │  │ • Executor      │  │
        │   │              │  │ • ENSO      │  │ • Learner       │  │
        │   └──────────────┘  └─────────────┘  │ • Invariants    │  │
        │                                      │ • Bitácora      │  │
        │   ┌──────────────┐  ┌─────────────┐  └─────────────────┘  │
        │   │ LLM Brain    │  │ Simulator   │                       │
        │   │ Qwen3 8B     │  │ PR Weather  │  ┌─────────────────┐  │
        │   │ + Context    │  │ + Soil Phys │  │ Web Dashboard   │  │
        │   │ + Memory     │  │ + Health    │  │ Glass UI + Chat │  │
        │   └──────────────┘  └─────────────┘  └─────────────────┘  │
        └───────────────────────────┬────────────────────────────────┘
                                    │
                        ┌───────────▼───────────────┐
                        │   ThirdReality Zigbee Valve │
                        │   💧 Automated Watering     │
                        └───────────────────────────┘
```

## 📁 Project Structure

```
julia/                      # Main package
├── core/                   # Decision engine, config, weather, risk engine, LLM brain
│   ├── decision_engine.py  # 7-priority watering logic with ML climate integration
│   ├── risk_engine.py      # Dual-horizon (weekly/monthly) climate risk scoring
│   ├── llm_brain.py        # Qwen3 8B conversational AI with live context injection
│   ├── llm_config.py       # LM Studio connection settings
│   ├── database.py         # SQLite persistence (sensors, decisions, episodes, chats)
│   ├── weather.py          # OpenWeather API + simulation fallback
│   ├── brain.py            # Lightweight rule-based decision wrapper
│   └── config.py           # Plant profiles loader + system configuration
│
├── agentic/                # Phase 2 — Autonomous agent layer
│   ├── state.py            # WorldState + AgenticContext (conditions hashing)
│   ├── perception.py       # OBSERVE: Sensor fusion + pattern extraction
│   ├── context_engine.py   # ORIENT: Context assembly + risk fusion
│   ├── caring.py           # Empathy state machine (NORMAL → STORM_PREP → RECOVERY)
│   ├── planner.py          # DECIDE: Wraps decision engine + instincts + invariants
│   ├── executor.py         # ACT: Safe action execution with rollback
│   ├── learner.py          # LEARN: Outcome evaluation + 3-Strike autocorrect
│   ├── invariants.py       # 5 unbreakable safety rules (Caring Invariants Firewall)
│   ├── memory.py           # Mistake memory with conditions-hash matching
│   └── bitacora.py         # Decision journal — SQLite + JSONL audit trail
│
├── sensors/                # Sensor I/O
│   ├── ha_client.py        # Home Assistant REST API client
│   ├── sensor_reader.py    # Unified SensorData with validation flags
│   └── data_validator.py   # Range checks, spike detection, staleness guards
│
├── vision/                 # Computer vision (Phase 3 — stub)
│   └── camera.py           # Arducam interface
│
├── actuators/              # Hardware control
│   └── watering.py         # ThirdReality Zigbee valve driver
│
├── simulator/              # Virtual garden for testing
│   ├── sim_engine.py       # Physics simulation (evaporation, rain, health)
│   ├── server.py           # HTTP API for the dashboard
│   └── index.html          # Real-time glassmorphic web dashboard
│
├── notifications/          # Alerts
│   └── notifier.py         # Pushover + MQTT push notifications
│
├── data/                   # Persistence
│   ├── plants.json         # Plant profiles (basil, pepper, tomato, carrot, cilantro, recao)
│   ├── database.py         # Extended DB with episodes, conversations, ML training data
│   └── data_cleaner.py     # Historical data deduplication + anomaly removal
│
├── models/                 # Trained ML models
│   ├── weekly_model.pkl    # Weekly climate risk predictor
│   └── monthly_model.pkl   # Monthly climate risk predictor
│
└── julia_main.py           # Entry point — scheduler, sensor loop, decision cycle

scripts/                    # Operational scripts
├── calibrate_risk_engine.py    # Weight tuning + threshold sweep
├── harvest_oracle.py           # End-to-end season simulation with scoring
├── stress_test_engine.py       # Brute-force edge case testing
├── train_risk_models.py        # ML model training pipeline
├── deep_dive_metrics.py        # 90-day reliability metrics extraction
├── dry_run_observer.py         # Full loop dry-run with human-readable output
├── agentic_stress_test.py      # Agentic layer fuzz testing
└── build_v1_datasets.py        # Synthetic training data generation for risk models

tests/                      # Test suite (19 files, ~100+ test cases)
├── test_decision_v1.py         # Decision engine unit tests
├── test_risk_engine.py         # Climate risk scoring tests
├── test_invariants.py          # Caring Invariants Firewall tests
├── test_agentic_*.py           # Agentic layer tests (caring, context, executor, learner, planner)
├── test_llm_brain.py           # LLM integration tests
├── test_julia_full_suite.py    # Comprehensive integration suite
└── ...

docs/
└── risk_philosophy.md      # Design decision: Responsive Escalation Engine
```

## 🌱 Plant Profiles

Julia ships with profiles tuned for Puerto Rico:

| Plant | Emoji | Moisture Range | Water (ml) | Notes |
|-------|-------|----------------|-----------|-------|
| Basil | 🌿 | 40–70% | 200 | Fast grower, loves sun |
| Pepper | 🌶️ | 35–65% | 250 | Drought tolerant, let soil dry between waterings |
| Tomato | 🍅 | 45–75% | 300 | Consistent moisture prevents blossom end rot |
| Carrot | 🥕 | 45–75% | 150 | BEWARE IGUANAS 🦎 |
| Cilantro | 🌱 | 40–60% | 150 | Plant in partial shade, bolts in heat |
| Recao | 🌱 | 50–80% | 200 | Native to PR, essential for sofrito! |

## ⚙️ Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Compute | NVIDIA Jetson Orin Nano 8GB (40 TOPS) | Julia's brain |
| Sensors | Haozee Zigbee Soil Moisture × 3 | Soil, temp, humidity |
| Watering | ThirdReality Zigbee Smart Watering Kit | Automated irrigation |
| Zigbee Hub | Sonoff Zigbee 3.0 USB Dongle Plus | Mesh coordinator |
| Camera | Arducam | Plant health vision |
| **Total** | | **~$410** |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/drosadocastro-bit/Julia.git
cd Julia
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Fill in:
#   HA_URL, HA_TOKEN         → Home Assistant
#   OPENWEATHER_API_KEY      → Weather data
#   PUSHOVER_USER_KEY (opt)  → Mobile alerts
```

### 3. Run the Simulator (No Hardware Needed)

```bash
python -m julia.simulator.server
```

Open [http://localhost:8888](http://localhost:8888) to see the live dashboard.

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Chat with Julia (Requires LM Studio + Qwen3 8B)

```bash
python -m julia.julia_main --chat
```

## 🧠 How Julia Thinks

Julia's decision engine uses a **7-priority cascade**:

1. 🚨 **EMERGENCY** — Plant wilting + soil dry → water 1.5× immediately
2. ⚠️ **SKIP_INVALID** — Bad sensor data → refuse to act blind
3. 🌧️ **SKIP_RAIN** — Rain >60% in 24h → save water
4. 💧 **SKIP_WET** — Soil above max → prevent root rot
5. ⏰ **SKIP_RECENT** — Watered too recently → respect cooldown
6. 💧 **WATER_NOW** — Soil below minimum → water (adjusted for heat + climate risk)
7. ✅ **NO_ACTION** — Everything is fine

Every decision passes through the **Caring Invariants Firewall** — 5 rules Julia *never* breaks:

1. Always explain **why**
2. Always offer a **monitoring signal**
3. **Prefer reversible** actions when risk > 50%
4. **Ask for help** when uncertainty > 60%
5. **Respect user constraints** (time, water, tools)

## 📜 Philosophy

- **Offline-capable** — Works without constant internet
- **Human-on-the-loop** — Julia advises, you decide
- **Transparent** — Every decision is logged to the Bitácora
- **Accessible** — Built with affordable, available components
- **Caribbean-first** — Designed for Puerto Rico's tropical climate, hurricane season, and local crops

## 👨‍💻 Author

**Danny** — FAA Air Traffic Systems Specialist, Bayamón, Puerto Rico. 20+ years of signal processing experience, building Julia to honor his grandmother's love of gardening.

## 📄 License

MIT