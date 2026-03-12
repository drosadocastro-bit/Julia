
import pytest
from unittest.mock import patch, MagicMock
from julia.core.weather import WeatherService

@pytest.fixture
def weather_service():
    return WeatherService()

def test_fallback_initialization(weather_service):
    """Test that service initialises with fallback data."""
    data = weather_service.current_weather
    assert "temperature" in data
    assert "rain_probability" in data
    assert data["source"] == "Simulation"

@patch("requests.get")
def test_fetch_real_weather_success(mock_get, weather_service):
    """Test successful API response parsing."""
    # Mock Open-Meteo response
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "current": {
            "temperature_2m": 30.5,
            "relative_humidity_2m": 55,
            "rain": 0,
            "precipitation": 0
        },
        "daily": {
            "precipitation_probability_max": [15]
        }
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    # Force update
    # We need to mock _fetch_real_weather to act normally, or just call update
    # But weather_service.update() calls _fetch_real_weather which calls requests.get
    
    data = weather_service.update()
    
    assert data["temperature"] == 30.5
    assert data["humidity"] == 55
    assert data["rain_probability"] == 15.0
    assert data["is_raining"] == False
    assert data["source"] == "Open-Meteo"

@patch("requests.get")
def test_api_failure_fallback(mock_get, weather_service):
    """Test fallback to simulation when API fails."""
    mock_get.side_effect = Exception("API Timeout")
    
    # Set initial state
    weather_service.current_weather = {
        "temperature": 25.0,
        "humidity": 50.0,
        "rain_probability": 0.0,
        "is_raining": False,
        "source": "Simulation"
    }

    data = weather_service.update()
    
    assert data["source"] == "Simulation"
    # Should wander slightly from 25.0
    assert 24.0 <= data["temperature"] <= 26.0
