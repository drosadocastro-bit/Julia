# JULIA — AI Crop Caretaker
## Project Specification Document
### Version 0.1 - Draft

---

> *"En honor a mi abuela Julia, que amaba las flores y le gustaba plantar"*
> 
> *In honor of my grandmother Julia, who loved flowers and enjoyed planting* 🌺

---

## 1. Executive Summary

**Julia** is an AI-powered crop caretaker system designed to help home gardeners successfully grow plants through intelligent monitoring, automated watering, and proactive health detection.

**The Problem:**
- Home gardeners struggle with inconsistent watering schedules
- Plants die from overwatering, underwatering, or undetected problems
- No easy way to know when something is wrong until it's too late
- Manual care is time-consuming and error-prone
- *"I planted a carrot and it just... disappeared"* 🥕👻

**The Solution:**
Julia combines soil sensors, weather forecasting, computer vision, and machine learning to:
- Monitor soil moisture, temperature, and humidity in real-time
- Make intelligent watering decisions based on multiple factors
- Detect plant health issues before they become critical
- Alert the gardener to pests, disease, or environmental problems
- Learn and adapt to each plant's specific needs

**Philosophy:**
- **Offline-capable** — Works without constant internet (like NIC)
- **Human-on-the-loop** — Julia advises, you decide
- **Transparent** — Every decision is explainable
- **Accessible** — Built with affordable, available components

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           JULIA SYSTEM ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    🌱 PLANTS (Basil, Peppers, Tomatoes, etc.)                               │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         SENSOR LAYER                                 │   │
│   │                                                                      │   │
│   │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐    │   │
│   │   │ HAOZEE   │   │ HAOZEE   │   │ HAOZEE   │   │   ARDUCAM    │    │   │
│   │   │ SENSOR 1 │   │ SENSOR 2 │   │ SENSOR 3 │   │   CAMERA     │    │   │
│   │   │          │   │          │   │          │   │              │    │   │
│   │   │ • Soil   │   │ • Soil   │   │ • Soil   │   │ • RGB Images │    │   │
│   │   │ • Temp   │   │ • Temp   │   │ • Temp   │   │ • Video      │    │   │
│   │   │ • Humid  │   │ • Humid  │   │ • Humid  │   │ • Time-lapse │    │   │
│   │   └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘    │   │
│   │        │              │              │                 │            │   │
│   │        └──────────────┴──────┬───────┴─────────────────┘            │   │
│   │                              │                                       │   │
│   └──────────────────────────────┼───────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      COMMUNICATION LAYER                             │   │
│   │                                                                      │   │
│   │   ┌────────────────┐              ┌─────────────────┐               │   │
│   │   │  ZIGBEE HUB    │              │   USB / CSI     │               │   │
│   │   │  (Coordinator) │              │   Connection    │               │   │
│   │   │                │              │                 │               │   │
│   │   │ • Zigbee2MQTT  │              │ • Direct to     │               │   │
│   │   │ • Mesh network │              │   Jetson        │               │   │
│   │   └───────┬────────┘              └────────┬────────┘               │   │
│   │           │                                │                         │   │
│   └───────────┼────────────────────────────────┼─────────────────────────┘   │
│               │                                │                             │
│               ▼                                ▼                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    JETSON ORIN NANO (40 TOPS)                        │   │
│   │                         "Julia's Brain"                              │   │
│   │                                                                      │   │
│   │   ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐ │   │
│   │   │    HOME       │ │    JULIA      │ │      VISION ENGINE        │ │   │
│   │   │   ASSISTANT   │ │    CORE       │ │                           │ │   │
│   │   │               │ │               │ │ ┌───────────────────────┐ │ │   │
│   │   │ • Zigbee2MQTT │ │ • Weather API │ │ │   Plant Health Model  │ │ │   │
│   │   │ • Sensor data │ │ • Regression  │ │ │   (YOLOv8 / Custom)   │ │ │   │
│   │   │ • Automations │ │   model       │ │ ├───────────────────────┤ │ │   │
│   │   │ • REST API    │ │ • Decision    │ │ │   Pest Detection      │ │ │   │
│   │   │               │ │   engine      │ │ │   (Iguana Watch 🦎)   │ │ │   │
│   │   │               │ │ • Learning    │ │ ├───────────────────────┤ │ │   │
│   │   │               │ │   module      │ │ │   Growth Tracking     │ │ │   │
│   │   │               │ │               │ │ │   (Time-lapse)        │ │ │   │
│   │   └───────┬───────┘ └───────┬───────┘ │ └───────────────────────┘ │ │   │
│   │           │                 │         └─────────────┬─────────────┘ │   │
│   │           └─────────────────┼───────────────────────┘               │   │
│   │                             │                                        │   │
│   │                             ▼                                        │   │
│   │                  ┌─────────────────────┐                            │   │
│   │                  │   DECISION ENGINE   │                            │   │
│   │                  │                     │                            │   │
│   │                  │ Inputs:             │                            │   │
│   │                  │ • Soil moisture     │                            │   │
│   │                  │ • Temperature       │                            │   │
│   │                  │ • Humidity          │                            │   │
│   │                  │ • Weather forecast  │                            │   │
│   │                  │ • Plant health      │                            │   │
│   │                  │ • Growth stage      │                            │   │
│   │                  │                     │                            │   │
│   │                  │ Outputs:            │                            │   │
│   │                  │ • Water YES/NO      │                            │   │
│   │                  │ • Water amount (ml) │                            │   │
│   │                  │ • Alerts            │                            │   │
│   │                  │ • Recommendations   │                            │   │
│   │                  └──────────┬──────────┘                            │   │
│   │                             │                                        │   │
│   └─────────────────────────────┼────────────────────────────────────────┘   │
│                                 │                                            │
│               ┌─────────────────┼─────────────────┐                         │
│               ▼                 ▼                 ▼                         │
│   ┌───────────────────┐ ┌─────────────┐ ┌───────────────────┐              │
│   │   THIRDREALITY    │ │  LOGGING &  │ │   DANNY'S APP     │              │
│   │   WATERING VALVE  │ │   DATABASE  │ │   (Notifications) │              │
│   │                   │ │             │ │                   │              │
│   │   💧 Automated    │ │ • Decisions │ │ • Status updates  │              │
│   │      watering     │ │ • Sensor    │ │ • Alerts          │              │
│   │                   │ │   history   │ │ • Manual override │              │
│   │                   │ │ • Photos    │ │ • Growth timeline │              │
│   └───────────────────┘ └─────────────┘ └───────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Sensor Layer** | Collect environmental data | Haozee Zigbee sensors, Arducam |
| **Communication Layer** | Connect sensors to brain | Zigbee mesh, USB/CSI |
| **Julia Brain** | Process data, make decisions | Jetson Orin Nano |
| **Home Assistant** | Device management, automations | Home Assistant OS |
| **Vision Engine** | Plant health, pest detection | NVIDIA TensorRT, YOLO |
| **Decision Engine** | Water scheduling logic | Python, scikit-learn |
| **Actuator Layer** | Execute watering | ThirdReality Zigbee valve |
| **User Interface** | Alerts, monitoring | Mobile app / Web dashboard |

---

## 3. Hardware Specifications

### 3.1 Bill of Materials

| Component | Model | Quantity | Unit Price | Total | Purpose |
|-----------|-------|----------|------------|-------|---------|
| **Compute** | NVIDIA Jetson Orin Nano (8GB) | 1 | $249.00 | $249.00 | Julia's brain |
| **Watering** | ThirdReality Zigbee Smart Watering Kit | 1 | $47.99 | $47.99 | Automated irrigation |
| **Sensors** | Haozee Zigbee Soil Moisture Sensor (3-pack) | 1 | $35.99 | $35.99 | Soil/temp/humidity |
| **Zigbee Hub** | Sonoff Zigbee 3.0 USB Dongle Plus | 1 | $29.99 | $29.99 | Zigbee coordinator |
| **Camera** | Arducam (existing) | 1 | $0.00 | $0.00 | Plant vision |
| **Storage** | MicroSD 64GB+ (for Jetson) | 1 | $12.00 | $12.00 | OS + data |
| **Power** | USB-C Power Supply (Jetson) | 1 | $15.00 | $15.00 | Power |
| **Enclosure** | Weatherproof enclosure (optional) | 1 | $20.00 | $20.00 | Outdoor protection |
| | | | **TOTAL** | **~$410** | |

*Note: Arducam already owned. Prices as of Feb 2026.*

### 3.2 Jetson Orin Nano Specifications

| Spec | Value |
|------|-------|
| **AI Performance** | 40 TOPS |
| **GPU** | 1024-core NVIDIA Ampere |
| **CPU** | 6-core Arm Cortex-A78AE (1.5 GHz) |
| **Memory** | 8GB 128-bit LPDDR5 |
| **Storage** | MicroSD / NVMe SSD |
| **Connectivity** | Gigabit Ethernet, USB 3.0, CSI camera |
| **Power** | 7W - 15W |
| **Size** | 100mm x 79mm x 21mm |

### 3.3 Sensor Specifications

**Haozee Soil Moisture Sensor:**

| Spec | Value |
|------|-------|
| **Protocol** | Zigbee 3.0 |
| **Measurements** | Soil moisture, temperature, humidity |
| **Moisture Range** | 0% - 100% |
| **Temperature Range** | -10°C to 50°C |
| **Battery** | 2x AAA (6-12 months) |
| **Wireless Range** | 30m (indoor), 100m (outdoor) |

**ThirdReality Watering Valve:**

| Spec | Value |
|------|-------|
| **Protocol** | Zigbee 3.0 |
| **Tubing Length** | 10 meters |
| **Flow Rate** | Adjustable via drippers |
| **Battery** | Battery powered |
| **Integration** | Home Assistant, SmartThings |

---

## 4. Software Architecture

### 4.1 Software Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                        │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ Julia Core   │  │ Julia Vision │  │ Julia Dashboard  │  │
│   │ (Python)     │  │ (TensorRT)   │  │ (Web/App)        │  │
│   └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     INTEGRATION LAYER                        │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ Home         │  │ Weather API  │  │ Notification     │  │
│   │ Assistant    │  │ (OpenWeather)│  │ Service          │  │
│   │ REST API     │  │              │  │ (Pushover/MQTT)  │  │
│   └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     PLATFORM LAYER                           │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ Home         │  │ Zigbee2MQTT  │  │ SQLite /         │  │
│   │ Assistant OS │  │              │  │ InfluxDB         │  │
│   └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     NVIDIA LAYER                             │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ JetPack SDK  │  │ TensorRT     │  │ CUDA             │  │
│   │              │  │              │  │                  │  │
│   └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     HARDWARE LAYER                           │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │              NVIDIA Jetson Orin Nano                  │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Julia Core Modules

```
julia/
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── decision_engine.py     # Watering decision logic
│   ├── weather_service.py     # Weather API integration
│   └── scheduler.py           # Task scheduling
│
├── sensors/
│   ├── __init__.py
│   ├── ha_client.py           # Home Assistant API client
│   ├── sensor_reader.py       # Read sensor data
│   └── data_validator.py      # Validate sensor readings
│
├── vision/
│   ├── __init__.py
│   ├── camera.py              # Arducam interface
│   ├── plant_health.py        # Plant health detection model
│   ├── pest_detector.py       # Pest/animal detection
│   └── growth_tracker.py      # Track plant growth over time
│
├── actuators/
│   ├── __init__.py
│   └── watering.py            # Control watering valve
│
├── models/
│   ├── water_regression.pkl   # Watering decision model
│   ├── plant_health.engine    # TensorRT plant health model
│   └── pest_detection.engine  # TensorRT pest detection model
│
├── data/
│   ├── plants.json            # Plant profiles and thresholds
│   ├── history.db             # SQLite decision/sensor history
│   └── images/                # Captured plant images
│
├── notifications/
│   ├── __init__.py
│   └── notifier.py            # Push notifications
│
├── api/
│   ├── __init__.py
│   └── dashboard.py           # Web dashboard API
│
├── tests/
│   ├── test_decision_engine.py
│   ├── test_sensors.py
│   └── test_vision.py
│
├── julia_main.py              # Main entry point
├── requirements.txt
└── README.md
```

### 4.3 Decision Engine

The heart of Julia — determines when and how much to water:

```python
# julia/core/decision_engine.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pickle

class WaterDecision(Enum):
    WATER_NOW = "water_now"
    SKIP_RAIN = "skip_rain_forecast"
    SKIP_WET = "skip_soil_wet"
    SKIP_RECENT = "skip_recent_watering"
    EMERGENCY = "emergency_watering"
    NO_ACTION = "no_action_needed"

@dataclass
class SensorData:
    soil_moisture: float      # 0-100%
    temperature: float        # Celsius
    humidity: float           # 0-100%
    sensor_id: str
    timestamp: str

@dataclass
class WeatherForecast:
    rain_probability_24h: float   # 0-100%
    rain_probability_48h: float   # 0-100%
    temp_high: float              # Celsius
    temp_low: float               # Celsius
    humidity: float               # 0-100%

@dataclass
class PlantHealth:
    status: str               # healthy, wilting, yellow_leaves, pest, disease
    confidence: float         # 0-1
    details: Optional[str]

@dataclass
class WateringResult:
    decision: WaterDecision
    water_amount_ml: int
    reason: str
    confidence: float

class JuliaDecisionEngine:
    """
    Julia's brain for watering decisions.
    
    Philosophy:
    - Better to underwater slightly than overwater
    - Trust sensors but verify with vision
    - Explain every decision
    - Learn from outcomes
    """
    
    def __init__(self, config_path: str = "data/plants.json"):
        self.plant_profiles = self._load_plant_profiles(config_path)
        self.model = self._load_regression_model()
        self.last_watering = {}
        
    def _load_plant_profiles(self, path: str) -> dict:
        """Load plant-specific thresholds and preferences."""
        # Default profiles
        return {
            "basil": {
                "min_moisture": 40,
                "max_moisture": 70,
                "optimal_moisture": 55,
                "water_amount_ml": 200,
                "min_hours_between_watering": 12,
                "drought_tolerant": False
            },
            "pepper": {
                "min_moisture": 35,
                "max_moisture": 65,
                "optimal_moisture": 50,
                "water_amount_ml": 250,
                "min_hours_between_watering": 18,
                "drought_tolerant": True
            },
            "tomato": {
                "min_moisture": 45,
                "max_moisture": 75,
                "optimal_moisture": 60,
                "water_amount_ml": 300,
                "min_hours_between_watering": 12,
                "drought_tolerant": False
            },
            "carrot": {
                "min_moisture": 45,
                "max_moisture": 75,
                "optimal_moisture": 60,
                "water_amount_ml": 150,
                "min_hours_between_watering": 24,
                "drought_tolerant": False,
                "notes": "Watch for iguanas 🦎"
            }
        }
    
    def _load_regression_model(self):
        """Load trained watering prediction model."""
        try:
            with open("models/water_regression.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None  # Fall back to rule-based
    
    def decide(
        self,
        plant_type: str,
        sensor_data: SensorData,
        weather: WeatherForecast,
        plant_health: Optional[PlantHealth] = None
    ) -> WateringResult:
        """
        Main decision function.
        
        Returns a WateringResult with decision, amount, and explanation.
        """
        profile = self.plant_profiles.get(plant_type, self.plant_profiles["basil"])
        
        # EMERGENCY: Plant is wilting and soil is dry
        if plant_health and plant_health.status == "wilting":
            if sensor_data.soil_moisture < profile["min_moisture"]:
                return WateringResult(
                    decision=WaterDecision.EMERGENCY,
                    water_amount_ml=int(profile["water_amount_ml"] * 1.5),
                    reason=f"🚨 Emergency: Plant wilting, soil at {sensor_data.soil_moisture}%",
                    confidence=0.95
                )
        
        # SKIP: Rain forecast
        if weather.rain_probability_24h > 60:
            return WateringResult(
                decision=WaterDecision.SKIP_RAIN,
                water_amount_ml=0,
                reason=f"🌧️ Rain likely ({weather.rain_probability_24h}% chance). Skipping.",
                confidence=0.85
            )
        
        # SKIP: Soil already wet
        if sensor_data.soil_moisture > profile["max_moisture"]:
            return WateringResult(
                decision=WaterDecision.SKIP_WET,
                water_amount_ml=0,
                reason=f"💧 Soil moisture at {sensor_data.soil_moisture}% (max: {profile['max_moisture']}%). Too wet.",
                confidence=0.90
            )
        
        # WATER: Soil is dry
        if sensor_data.soil_moisture < profile["min_moisture"]:
            # Adjust for temperature
            amount = profile["water_amount_ml"]
            if sensor_data.temperature > 30:
                amount = int(amount * 1.2)  # Hot day, more water
            
            return WateringResult(
                decision=WaterDecision.WATER_NOW,
                water_amount_ml=amount,
                reason=f"💧 Soil at {sensor_data.soil_moisture}% (min: {profile['min_moisture']}%). Watering {amount}ml.",
                confidence=0.88
            )
        
        # CHECK: Use regression model for edge cases
        if self.model:
            features = [
                sensor_data.soil_moisture,
                sensor_data.temperature,
                sensor_data.humidity,
                weather.rain_probability_24h,
                weather.temp_high
            ]
            prediction = self.model.predict([features])[0]
            
            if prediction > 0.5:
                return WateringResult(
                    decision=WaterDecision.WATER_NOW,
                    water_amount_ml=profile["water_amount_ml"],
                    reason=f"🤖 Model suggests watering (confidence: {prediction:.2f})",
                    confidence=prediction
                )
        
        # NO ACTION: Everything looks good
        return WateringResult(
            decision=WaterDecision.NO_ACTION,
            water_amount_ml=0,
            reason=f"✅ Soil at {sensor_data.soil_moisture}%. All good!",
            confidence=0.85
        )
```

### 4.4 Vision Engine

Plant health detection using Jetson's GPU:

```python
# julia/vision/plant_health.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import cv2

# TensorRT for optimized inference on Jetson
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

@dataclass
class HealthDetection:
    label: str          # healthy, yellow_leaves, wilting, pest, disease, overwatered
    confidence: float   # 0-1
    bbox: Tuple[int, int, int, int]  # x, y, w, h (if applicable)

class PlantHealthDetector:
    """
    Detect plant health issues using computer vision.
    
    Runs on Jetson Orin Nano GPU using TensorRT.
    """
    
    LABELS = [
        "healthy",
        "yellow_leaves",
        "wilting",
        "brown_spots",
        "pest_damage",
        "overwatered",
        "underwatered",
        "nutrient_deficiency"
    ]
    
    def __init__(self, model_path: str = "models/plant_health.engine"):
        self.engine = self._load_engine(model_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = (224, 224)  # Model input size
        
    def _load_engine(self, path: str):
        """Load TensorRT engine."""
        with open(path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            return runtime.deserialize_cuda_engine(f.read())
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for inference."""
        # Resize
        img = cv2.resize(image, self.input_shape)
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)
    
    def detect(self, image: np.ndarray) -> List[HealthDetection]:
        """
        Detect plant health issues in image.
        
        Args:
            image: BGR image from camera
            
        Returns:
            List of HealthDetection objects
        """
        # Preprocess
        input_data = self.preprocess(image)
        
        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        output = np.empty((1, len(self.LABELS)), dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        
        # Copy input to GPU
        cuda.memcpy_htod(d_input, input_data)
        
        # Run inference
        self.context.execute_v2([int(d_input), int(d_output)])
        
        # Copy output from GPU
        cuda.memcpy_dtoh(output, d_output)
        
        # Parse results
        detections = []
        probabilities = self._softmax(output[0])
        
        for i, prob in enumerate(probabilities):
            if prob > 0.1:  # Threshold
                detections.append(HealthDetection(
                    label=self.LABELS[i],
                    confidence=float(prob),
                    bbox=(0, 0, 0, 0)  # Classification, no bbox
                ))
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class PestDetector:
    """
    Detect pests and animals near plants.
    
    Especially important for detecting iguanas 🦎
    """
    
    PESTS = [
        "iguana",      # 🦎 Public enemy #1 in Puerto Rico
        "aphid",
        "caterpillar",
        "snail",
        "bird",
        "rat",
        "unknown_pest"
    ]
    
    def __init__(self, model_path: str = "models/pest_detection.engine"):
        # YOLOv8 TensorRT engine for object detection
        self.model_path = model_path
        # ... similar TensorRT setup
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect pests in image.
        
        Returns list of detections with bounding boxes.
        """
        # ... YOLOv8 inference
        pass
    
    def alert_if_pest(self, detections: List[dict]) -> str:
        """Generate alert message if pest detected."""
        for det in detections:
            if det["label"] == "iguana" and det["confidence"] > 0.7:
                return "🦎 IGUANA ALERT! An iguana was spotted near your plants!"
        return None
```

---

## 5. Data Flow

### 5.1 Sensor Data Collection

```
Every 15 minutes:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Haozee      │────▶│ Zigbee2MQTT │────▶│ Home        │
│ Sensor      │     │             │     │ Assistant   │
│             │     │ • Decode    │     │             │
│ • Moisture  │     │ • Publish   │     │ • Store     │
│ • Temp      │     │   to MQTT   │     │ • API       │
│ • Humidity  │     │             │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Julia Core  │
                                        │             │
                                        │ • Validate  │
                                        │ • Store     │
                                        │ • Process   │
                                        └─────────────┘
```

### 5.2 Watering Decision Flow

```
┌─────────────┐
│ Scheduled   │ (Every 6 hours or on-demand)
│ Check       │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ Get Sensor  │────▶│ Validate    │
│ Data        │     │ Readings    │
└─────────────┘     └──────┬──────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌─────────────┐                         ┌─────────────┐
│ Get Weather │                         │ Get Plant   │
│ Forecast    │                         │ Health      │
│             │                         │ (Vision)    │
└──────┬──────┘                         └──────┬──────┘
       │                                       │
       └───────────────┬───────────────────────┘
                       ▼
                ┌─────────────┐
                │ Decision    │
                │ Engine      │
                │             │
                │ Input:      │
                │ • Sensors   │
                │ • Weather   │
                │ • Health    │
                │ • History   │
                └──────┬──────┘
                       │
                       ▼
              ┌───────────────┐
              │ DECISION      │
              │               │
              │ • Water: Y/N  │
              │ • Amount: ml  │
              │ • Reason      │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌─────────────┐ ┌─────────┐ ┌─────────────┐
│ Execute     │ │ Log     │ │ Notify      │
│ Watering    │ │ Decision│ │ Danny       │
│ (if yes)    │ │         │ │             │
└─────────────┘ └─────────┘ └─────────────┘
```

---

## 6. Machine Learning Models

### 6.1 Watering Regression Model

**Purpose:** Predict optimal watering based on multiple factors.

**Type:** Gradient Boosting Regressor (scikit-learn)

**Features:**
| Feature | Description | Range |
|---------|-------------|-------|
| soil_moisture | Current soil moisture % | 0-100 |
| temperature | Ambient temperature °C | -10 to 50 |
| humidity | Air humidity % | 0-100 |
| rain_probability | Chance of rain next 24h | 0-100 |
| temp_forecast | Forecasted high temp | -10 to 50 |
| days_since_water | Days since last watering | 0-14 |
| plant_growth_stage | Seedling/Growing/Mature | 0-2 |

**Output:** Water amount (ml) or 0 for no watering

**Training Data Collection:**
```python
# Data structure for training
training_sample = {
    "timestamp": "2026-02-09T14:30:00",
    "features": {
        "soil_moisture": 35,
        "temperature": 28,
        "humidity": 65,
        "rain_probability": 20,
        "temp_forecast": 32,
        "days_since_water": 1,
        "growth_stage": 1
    },
    "action_taken": "water",
    "water_amount_ml": 200,
    "outcome_24h": {
        "plant_health": "healthy",
        "soil_moisture_after": 58
    }
}
```

### 6.2 Plant Health Classification Model

**Purpose:** Identify plant health issues from images.

**Type:** MobileNetV2 (fine-tuned) + TensorRT optimization

**Classes:**
- healthy
- yellow_leaves
- wilting
- brown_spots
- pest_damage
- overwatered
- underwatered
- nutrient_deficiency

**Training:**
- Dataset: PlantVillage + custom Puerto Rico plants
- Fine-tune on Jetson-friendly architecture
- Export to TensorRT for fast inference

### 6.3 Pest Detection Model

**Purpose:** Detect pests and animals near plants.

**Type:** YOLOv8n (nano) + TensorRT

**Classes:**
- iguana 🦎 (priority #1!)
- aphid
- caterpillar
- snail
- bird
- unknown_pest

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Basic sensor reading and manual watering triggers

- [ ] Set up Jetson Orin Nano with JetPack
- [ ] Install Home Assistant
- [ ] Configure Zigbee2MQTT with Sonoff dongle
- [ ] Pair Haozee sensors
- [ ] Pair ThirdReality watering valve
- [ ] Create basic Julia Core structure
- [ ] Implement Home Assistant API client
- [ ] Test sensor readings and watering control
- [ ] **Plant the basil! 🌿**

**Deliverable:** Can read sensors and trigger watering via script

### Phase 2: Smart Decisions (Weeks 3-4)
**Goal:** Weather-aware automated watering

- [ ] Integrate OpenWeather API
- [ ] Implement rule-based decision engine
- [ ] Add scheduling (check every 6 hours)
- [ ] Implement logging and history
- [ ] Create basic notification system
- [ ] Collect training data for ML model
- [ ] Test with basil plant

**Deliverable:** Julia waters automatically based on sensors + weather

### Phase 3: Vision (Weeks 5-6)
**Goal:** Plant health monitoring with camera

- [ ] Set up Arducam with Jetson
- [ ] Train/fine-tune plant health model
- [ ] Convert model to TensorRT
- [ ] Implement health detection pipeline
- [ ] Add health status to watering decisions
- [ ] Capture daily photos for growth tracking
- [ ] Test emergency watering on wilting detection

**Deliverable:** Julia can "see" plant health and respond

### Phase 4: Intelligence (Weeks 7-8)
**Goal:** Learning and optimization

- [ ] Train regression model on collected data
- [ ] Implement A/B testing (rule-based vs ML)
- [ ] Add pest detection (Iguana Watch 🦎)
- [ ] Create growth tracking time-lapse
- [ ] Implement feedback loop (did the plant improve?)
- [ ] Tune thresholds based on results

**Deliverable:** Julia learns and improves over time

### Phase 5: Polish (Weeks 9-10)
**Goal:** Dashboard and expansion

- [ ] Create web dashboard
- [ ] Add multiple plant support
- [ ] Create mobile-friendly interface
- [ ] Document everything
- [ ] **Add more plants! 🌶️🥕🍅**

**Deliverable:** Production-ready Julia system

---

## 8. Plant Profiles (Puerto Rico)

### 8.1 Supported Plants (Phase 1)

| Plant | Min Moisture | Max Moisture | Water (ml) | Notes |
|-------|-------------|--------------|------------|-------|
| **Basil** 🌿 | 40% | 70% | 200 | Easy starter, fast growth |
| **Pepper** 🌶️ | 35% | 65% | 250 | Heat tolerant, good for PR |
| **Tomato** 🍅 | 45% | 75% | 300 | Needs consistent moisture |
| **Carrot** 🥕 | 45% | 75% | 150 | BEWARE IGUANAS 🦎 |
| **Cilantro** 🌱 | 40% | 60% | 150 | Bolts in heat, partial shade |
| **Recao** 🌱 | 50% | 80% | 200 | Native to PR, shade tolerant |

### 8.2 Puerto Rico Climate Considerations

| Factor | Consideration |
|--------|---------------|
| **High humidity** | Reduce watering frequency vs mainland |
| **Year-round growing** | No winter dormancy |
| **Hurricane season** | Move plants indoors when storms approach |
| **Intense sun** | Some plants need afternoon shade |
| **Iguanas** 🦎 | Physical barriers, pest detection alerts |
| **Salt air** | Near coast, rinse leaves occasionally |

---

## 9. API Reference

### 9.1 Julia Core API

```python
# Get current plant status
julia.get_plant_status(plant_id: str) -> PlantStatus

# Trigger manual watering
julia.water_now(plant_id: str, amount_ml: int) -> WateringResult

# Get watering history
julia.get_history(plant_id: str, days: int = 7) -> List[WateringEvent]

# Get latest sensor data
julia.get_sensors(plant_id: str) -> SensorData

# Capture plant photo
julia.capture_image(plant_id: str) -> str  # Returns image path

# Check plant health
julia.check_health(plant_id: str) -> PlantHealth

# Get weather forecast
julia.get_weather() -> WeatherForecast
```

### 9.2 Home Assistant Integration

```yaml
# Home Assistant configuration.yaml

sensor:
  - platform: mqtt
    name: "Basil Soil Moisture"
    state_topic: "zigbee2mqtt/basil_sensor/soil_moisture"
    unit_of_measurement: "%"
    
  - platform: mqtt
    name: "Basil Temperature"
    state_topic: "zigbee2mqtt/basil_sensor/temperature"
    unit_of_measurement: "°C"

switch:
  - platform: mqtt
    name: "Basil Watering Valve"
    command_topic: "zigbee2mqtt/watering_valve/set"
    state_topic: "zigbee2mqtt/watering_valve/state"

automation:
  - alias: "Julia Watering Check"
    trigger:
      - platform: time_pattern
        hours: "/6"  # Every 6 hours
    action:
      - service: rest_command.julia_check_watering
```

---

## 10. Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Plant survival rate** | 100% | Plants alive after 30 days |
| **Watering accuracy** | >90% | Correct decisions vs manual review |
| **False positive rate** | <5% | Unnecessary watering events |
| **Health detection accuracy** | >85% | Correct health classifications |
| **Pest detection** | >90% | Iguanas caught on camera 🦎 |
| **System uptime** | >99% | Julia available and responding |
| **Response time** | <5s | Sensor read to decision |

---

## 11. Future Expansion

### 11.1 Version 2.0 Ideas

- **Multiple zones** — Different watering schedules per area
- **Nutrient management** — Automated fertilizer dosing
- **Greenhouse integration** — Climate control (fans, shade)
- **Voice interface** — "Julia, how's my basil?"
- **Community sharing** — Share plant profiles with other Julia users
- **Marketplace** — Plant profiles for Puerto Rico crops

### 11.2 ARIA Integration

Julia and ARIA (GTI AI) could share:
- Weather data
- Alert infrastructure
- Dashboard framework
- ML training pipeline

---

## 12. Acknowledgments

**Julia** is named in honor of **Abuela Julia**, who loved flowers and gardening. Her spirit of nurturing life inspires this project.

This project is built with the help of:
- **Claude (Anthropic)** — Architecture design, documentation
- **Nova** — Project ideation, refinement
- **Open source community** — Home Assistant, Zigbee2MQTT, NVIDIA

---

## Appendix A: Shopping List

### Immediate Purchase (Phase 1)

| Item | Link | Price |
|------|------|-------|
| NVIDIA Jetson Orin Nano 8GB | [NVIDIA Store](https://store.nvidia.com/) | $249 |
| ThirdReality Zigbee Watering | [Amazon](https://amazon.com) | $47.99 |
| Haozee Soil Sensors (3-pack) | [Amazon](https://amazon.com) | $35.99 |
| Sonoff Zigbee USB Dongle | [Amazon](https://amazon.com) | $29.99 |
| MicroSD 64GB | [Amazon](https://amazon.com) | $12.00 |
| **TOTAL** | | **~$375** |

### Already Owned

| Item | Status |
|------|--------|
| Arducam | ✅ Have it |
| CanaKit (Pi Zero W2) | ✅ Backup |

### Future (Phase 3+)

| Item | Price | Purpose |
|------|-------|---------|
| Additional sensors | $12/each | More plants |
| Weatherproof enclosure | $20-30 | Outdoor setup |
| Grow lights (optional) | $30-50 | Indoor growing |

---

## Appendix B: Puerto Rico Plant Calendar

| Month | Best to Plant | Harvest |
|-------|---------------|---------|
| Jan-Feb | Tomatoes, peppers | Lettuce |
| Mar-Apr | Basil, peppers | Tomatoes |
| May-Jun | Heat-tolerant crops | Peppers |
| Jul-Aug | Monsoon prep | Basil |
| Sep-Oct | Hurricane watch ⚠️ | — |
| Nov-Dec | Cool crops | Peppers, tomatoes |

---

*Document Version: 0.1 Draft*
*Author: Danny & Claude*
*Date: February 2026*

---

**Julia: Because every plant deserves a chance to thrive.** 🌱💚

*En memoria de Abuela Julia* 🌺
