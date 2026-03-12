"""Tests for Julia's LLM Brain module."""

import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from julia.core.llm_brain import JuliaBrain, JuliaContext


class TestContextBuilder:
    """Test context building from database."""
    
    def test_empty_context_without_db(self):
        brain = JuliaBrain(db=None)
        context = brain.build_context()
        assert context.plants == []
        assert context.weather == {}
        assert context.mistakes == []
    
    def test_context_format_includes_plants(self):
        brain = JuliaBrain(db=None)
        context = JuliaContext(
            plants=[{
                "plant": "basil",
                "soil_moisture": 45.0,
                "temperature": 28.0,
                "humidity": 65.0,
                "timestamp": "2026-02-18T10:00:00"
            }],
            weather={"temperature": 30, "rain_probability_24h": 20},
            recent_decisions=[],
            recent_waterings=[],
            mistakes=[]
        )
        formatted = brain.format_context_for_prompt(context)
        assert "Basil" in formatted
        assert "45.0%" in formatted
    
    def test_context_format_includes_mistakes(self):
        brain = JuliaBrain(db=None)
        context = JuliaContext(
            plants=[],
            weather={},
            recent_decisions=[],
            recent_waterings=[],
            mistakes=[{
                "timestamp": "2026-02-17",
                "plant_id": "basil",
                "action": "skip",
                "soil_moisture": 42,
                "outcome_moisture": 18,
                "outcome_health": "wilting"
            }]
        )
        formatted = brain.format_context_for_prompt(context)
        assert "SHOULD HAVE WATERED" in formatted
        assert "wilting" in formatted


class TestChatMethod:
    """Test the chat interface."""
    
    @patch("requests.post")
    def test_successful_chat(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{
                "message": {"content": "Water your basil now! 🌱"}
            }]
        }
        
        brain = JuliaBrain(db=None)
        response = brain.chat("Should I water my basil?")
        assert "basil" in response.lower()
    
    @patch("requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = Exception("Connection refused")
        
        brain = JuliaBrain(db=None)
        response = brain.chat("Hello")
        assert "Something went wrong" in response
    
    def test_conversation_history_maintained(self):
        brain = JuliaBrain(db=None)
        brain.conversation_history.append(
            {"role": "user", "content": "test"}
        )
        brain.conversation_history.append(
            {"role": "assistant", "content": "response"}
        )
        assert len(brain.conversation_history) == 2
    
    def test_clear_history(self):
        brain = JuliaBrain(db=None)
        brain.conversation_history = [{"role": "user", "content": "test"}]
        brain.clear_history()
        assert brain.conversation_history == []


class TestDailyBriefing:
    """Test proactive features."""
    
    @patch("requests.post")
    def test_morning_briefing(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{
                "message": {"content": "Good morning! Your garden looks great."}
            }]
        }
        
        brain = JuliaBrain(db=None)
        briefing = brain.generate_daily_briefing()
        assert len(briefing) > 0
