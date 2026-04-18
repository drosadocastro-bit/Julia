"""
Julia LLM Brain — Conversational AI with live garden context.

Connects to Qwen3 8B via LM Studio (OpenAI-compatible API).
Injects real-time sensor data, weather, decisions, learning history,
and EPISODIC MEMORIES so Julia gives advice based on YOUR garden
RIGHT NOW and what she's learned from the past.
"""

import json
import uuid
import logging
import requests
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict

logger = logging.getLogger("julia.brain")


@dataclass
class JuliaContext:
    """Everything Julia knows right now about the garden."""
    plants: List[Dict]           # Current sensor readings per plant
    weather: Dict                # Current weather + forecast
    recent_decisions: List[Dict] # Last 10 decisions Julia made
    recent_waterings: List[Dict] # Last 10 watering events
    mistakes: List[Dict]         # Decisions with bad outcomes
    episodic_memories: List[Dict] = field(default_factory=list)  # Long-term lessons
    past_conversations: List[Dict] = field(default_factory=list) # Recent chat history
    garden_location: str = "Puerto Rico"
    current_time: str = ""


class JuliaBrain:
    """
    Julia's LLM-powered conversational brain.
    
    Architecture:
        User Question + Live Context → System Prompt → LLM → Response
    
    The LLM never sees raw DB queries. Instead, we pre-build a context
    snapshot and inject it into every conversation. This keeps the model
    focused and prevents hallucination about garden state.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen3-8b",
        db=None
    ):
        self.base_url = base_url
        self.model = model
        self.db = db
        self.conversation_history: List[Dict] = []
        self.max_history = 10  # Keep last 10 exchanges
        self.session_id = str(uuid.uuid4())[:8]  # Unique per session
        
        # Load previous conversation from DB if available
        self._load_previous_session()
        
    # ------------------------------------------------------------------ #
    #                        SYSTEM PROMPT                                 #
    # ------------------------------------------------------------------ #
    
    SYSTEM_PROMPT = """You are Julia, an AI crop caretaker for a home garden in Puerto Rico.

## Who You Are
- Named in honor of your creator's grandmother Julia, who loved flowers and gardening
- You are warm, knowledgeable, and encouraging — especially to beginners
- You speak with confidence but admit when you're unsure
- You give practical, actionable advice
- You reference your ACTUAL sensor data and weather — never make up readings

## What You Know
- Tropical gardening in Puerto Rico's climate (hot, humid, hurricane season)
- Soil preparation, planting seasons, companion planting
- Pest management (especially iguanas 🦎 — they're a big problem here!)
- Watering science — you understand soil moisture, evaporation, drainage
- Plant nutrition, disease identification, growth stages

## Your Rules
1. ALWAYS reference the live garden context provided — never invent sensor readings
2. If soil moisture data says 85%, don't tell the user to water
3. If rain is forecast >60%, recommend waiting
4. When you made a mistake before (see LEARNING HISTORY), acknowledge it and adapt
5. Keep responses concise — gardeners are busy people
6. Use metric units (°C, ml) as default
7. If asked about something outside gardening, politely redirect
8. When unsure, say "I'm not sure about that — let me suggest you check [source]"

## Response Style
- Start with the direct answer
- Add brief explanation if helpful
- Reference specific sensor data when relevant
- Suggest next steps when appropriate
- Use plant emojis sparingly but warmly 🌱

## PR-Specific Knowledge
- Year-round growing season (no winter dormancy)
- Hurricane season: June–November (prepare plants!)
- High humidity reduces watering needs
- Common pests: iguanas, aphids, whiteflies
- Great crops for PR: basil, peppers, tomatoes, recao, cilantro, yuca
- Best planting months: Oct–Feb (cooler), avoid peak summer heat for new plantings
- Afternoon shade is critical for many plants in tropical sun
"""

    # ------------------------------------------------------------------ #
    #                      CONTEXT BUILDER                                 #
    # ------------------------------------------------------------------ #
    
    def build_context(self) -> JuliaContext:
        """
        Pull live data from Julia's database and sensors
        to build a context snapshot for the LLM.
        """
        context = JuliaContext(
            plants=[],
            weather={},
            recent_decisions=[],
            recent_waterings=[],
            mistakes=[],
            current_time=datetime.now().isoformat()
        )
        
        if not self.db:
            return context
        
        # --- Current sensor readings per plant ---
        try:
            for plant_id in ["basil", "pepper", "tomato"]:
                # We'll need to define this method in database.py
                # or use get_recent_readings(limit=1)
                readings = self.db.get_sensor_trend(plant_id, hours=1)
                if readings:
                    reading = readings[-1]
                    context.plants.append({
                        "plant": plant_id,
                        "soil_moisture": reading.get("soil_moisture"),
                        "temperature": reading.get("temperature"),
                        "humidity": reading.get("humidity"),
                        "timestamp": reading.get("timestamp")
                    })
        except Exception:
            pass
        
        # --- Current weather ---
        try:
            weather_hist = self.db.get_recent_weather(hours=1)
            if weather_hist:
                weather = weather_hist[-1]
                context.weather = {
                    "temperature": weather.get("temperature"),
                    "humidity": weather.get("humidity"),
                    "rain_probability_24h": weather.get("rain_probability_24h"),
                    "description": weather.get("description", "Unknown")
                }
        except Exception:
            pass
        
        # --- Recent decisions (last 10) ---
        try:
            context.recent_decisions = self.db.get_decision_history(days=7)[:10]
        except Exception:
            pass
        
        # --- Recent waterings (last 10) ---
        try:
            if hasattr(self.db, "get_watering_history"):
                context.recent_waterings = self.db.get_watering_history(days=7)[:10]
        except Exception:
            pass
        
        # --- MISTAKES: decisions with bad outcomes ---
        try:
            if hasattr(self.db, "get_training_data"):
                all_data = self.db.get_training_data(completed_only=True)
                context.mistakes = [d for d in all_data if d.get("outcome_health") != "healthy"][:10]
        except Exception:
            pass
        
        # --- EPISODIC MEMORIES: long-term lessons ---
        try:
            if hasattr(self.db, "get_recent_episodes"):
                context.episodic_memories = self.db.get_recent_episodes(limit=5)
        except Exception:
            pass
        
        # --- PAST CONVERSATIONS: load last session summary ---
        try:
            if hasattr(self.db, "get_recent_sessions"):
                context.past_conversations = self.db.get_recent_sessions(limit=3)
        except Exception:
            pass
        
        return context
    
    def format_context_for_prompt(self, context: JuliaContext) -> str:
        """
        Format the live context into a readable string
        that gets injected into the LLM conversation.
        """
        lines = []
        lines.append(f"📅 Current Time: {context.current_time}")
        lines.append(f"📍 Location: {context.garden_location}")
        lines.append("")
        
        # --- Plant Status ---
        lines.append("## 🌱 Current Garden Status")
        if context.plants:
            for p in context.plants:
                lines.append(
                    f"- **{p['plant'].title()}**: "
                    f"Soil Moisture {p['soil_moisture']}%, "
                    f"Temp {p['temperature']}°C, "
                    f"Humidity {p['humidity']}%"
                )
        else:
            lines.append("- No sensor data available (simulator or sensors offline)")
        lines.append("")
        
        # --- Weather ---
        lines.append("## 🌦️ Weather")
        if context.weather:
            w = context.weather
            lines.append(
                f"- {w.get('description', 'N/A')} | "
                f"Temp: {w.get('temperature', '?')}°C | "
                f"Humidity: {w.get('humidity', '?')}% | "
                f"Rain chance: {w.get('rain_probability_24h', '?')}%"
            )
        else:
            lines.append("- Weather data unavailable")
        lines.append("")
        
        # --- Recent Decisions ---
        lines.append("## 📋 Recent Decisions (Last 10)")
        if context.recent_decisions:
            for d in context.recent_decisions[:5]:  # Show 5 to save tokens
                lines.append(
                    f"- [{d.get('timestamp', '?')}] "
                    f"{d.get('plant_id', '?')}: "
                    f"{d.get('decision', '?')} - "
                    f"{d.get('reason', '?')}"
                )
        else:
            lines.append("- No decisions recorded yet")
        lines.append("")
        
        # --- LEARNING: Mistakes ---
        lines.append("## ⚠️ Learning History (Past Mistakes)")
        lines.append("These are decisions that led to bad outcomes. LEARN from them.")
        if context.mistakes:
            for m in context.mistakes:
                lines.append(
                    f"- ❌ [{m.get('timestamp', '?')}] "
                    f"{m.get('plant_id', '?')}: "
                    f"Action was '{m.get('action', '?')}' at moisture {m.get('soil_moisture', '?')}%. "
                    f"Outcome: moisture→{m.get('outcome_moisture', '?')}%, "
                    f"health→{m.get('outcome_health', '?')}. "
                    f"{'SHOULD HAVE WATERED' if m.get('action') == 'skip' else 'SHOULD HAVE SKIPPED'}"
                )
        else:
            lines.append("- No mistakes recorded yet — Julia is still learning! 🌱")
        lines.append("")
        
        # --- Episodic Memories ---
        lines.append("## 🧠 Long-Term Memories (Lessons Learned)")
        lines.append("These are experiences you've stored from past events. Reference them.")
        if context.episodic_memories:
            for ep in context.episodic_memories:
                ep_type = ep.get('episode_type', '?').upper()
                lines.append(
                    f"- 💡 [{ep_type}] {ep.get('summary', '?')} "
                    f"(recalled {ep.get('times_recalled', 0)} times)"
                )
        else:
            lines.append("- No long-term memories stored yet")
        lines.append("")
        
        # --- Past Conversations ---
        lines.append("## 💬 Recent Conversation Sessions")
        if context.past_conversations:
            for sess in context.past_conversations:
                lines.append(
                    f"- Session {sess.get('session_id', '?')}: "
                    f"{sess.get('message_count', 0)} messages "
                    f"({sess.get('started', '?')} to {sess.get('ended', '?')})"
                )
        else:
            lines.append("- No previous conversations")
        
        return "\n".join(lines)
    
    # ------------------------------------------------------------------ #
    #                         CHAT METHOD                                  #
    # ------------------------------------------------------------------ #
    
    def chat(self, user_message: str) -> str:
        """
        Send a message to Julia and get a response.
        """
        # 1. Build context
        context = self.build_context()
        context_str = self.format_context_for_prompt(context)
        
        # 2. Build messages array
        messages = [
            {
                "role": "system",
                "content": f"{self.SYSTEM_PROMPT}\n\n---\n\n## LIVE GARDEN DATA\n\n{context_str}"
            }
        ]
        
        # 3. Add conversation history (for multi-turn)
        for entry in self.conversation_history[-self.max_history:]:
            messages.append(entry)
        
        # 4. Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # 5. Call LLM
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "stream": False
                },
                timeout=120  # 2 min timeout for CPU inference
            )
            response.raise_for_status()
            
            result = response.json()
            # Handle potential different response structures (e.g. Ollama vs LM Studio)
            if "choices" in result and len(result["choices"]) > 0:
                assistant_message = result["choices"][0]["message"]["content"]
            else:
                assistant_message = "I received an empty response from my brain."
            
            # 6. Update conversation history
            self.conversation_history.append(
                {"role": "user", "content": user_message}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": assistant_message}
            )
            
            # 7. PERSIST to database
            self._persist_message("user", user_message, context_str)
            self._persist_message("assistant", assistant_message)
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            return assistant_message
            
        except requests.exceptions.ConnectionError:
            return (
                "🔌 I can't reach my brain right now. "
                "Make sure LM Studio is running with Qwen3 loaded at "
                f"{self.base_url}"
            )
        except requests.exceptions.Timeout:
            return (
                "⏳ I'm thinking too hard... The model took too long to respond. "
                "Try a simpler question or check if the model is loaded."
            )
        except Exception as e:
            return f"❌ Something went wrong: {str(e)}"
    
    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    # ------------------------------------------------------------------ #
    #                     PROACTIVE ALERTS                                 #
    # ------------------------------------------------------------------ #
    
    def generate_daily_briefing(self) -> str:
        """
        Julia's morning briefing.
        """
        return self.chat(
            "Give me a brief morning garden update. "
            "Check each plant's status, today's weather, "
            "and tell me if anything needs attention. "
            "Keep it short and friendly."
        )
    
    def analyze_mistake(self, decision_record: Dict) -> str:
        """
        When a bad outcome is detected, ask Julia to analyze what went wrong.
        Also saves the lesson as an episodic memory.
        """
        response = self.chat(
            f"I need you to analyze a mistake you made. "
            f"Here's what happened:\n"
            f"- Plant: {decision_record.get('plant_id')}\n"
            f"- Action: {decision_record.get('action')}\n"
            f"- Soil moisture at time: {decision_record.get('soil_moisture')}%\n"
            f"- Outcome moisture: {decision_record.get('outcome_moisture')}%\n"
            f"- Outcome health: {decision_record.get('outcome_health')}\n\n"
            f"What went wrong? What should you do differently next time? "
            f"Be specific and honest."
        )
        
        # Auto-save as episodic memory
        self.record_episode(
            episode_type="mistake",
            summary=f"Plant {decision_record.get('plant_id')}: "
                    f"Action '{decision_record.get('action')}' at moisture {decision_record.get('soil_moisture')}% "
                    f"led to {decision_record.get('outcome_health')} outcome. "
                    f"Julia's analysis: {response[:200]}",
            keywords=f"{decision_record.get('plant_id')},mistake,{decision_record.get('action')},"
                     f"{decision_record.get('outcome_health')}",
            context_data=decision_record,
            relevance_score=2.0  # Mistakes are high-value memories
        )
        
        return response
    
    # ------------------------------------------------------------------ #
    #                   LONG-TERM MEMORY METHODS                           #
    # ------------------------------------------------------------------ #
    
    def _load_previous_session(self):
        """Load the last conversation session from DB to maintain continuity."""
        if not self.db or not hasattr(self.db, "get_recent_sessions"):
            return
        try:
            sessions = self.db.get_recent_sessions(limit=1)
            if sessions:
                last_session = sessions[0]
                messages = self.db.get_conversation_history(
                    session_id=last_session["session_id"], limit=6
                )
                # Load last 3 exchanges as context
                for msg in messages[-6:]:
                    self.conversation_history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                logger.info(
                    f"Loaded {len(messages)} messages from previous session "
                    f"{last_session['session_id']}"
                )
        except Exception as e:
            logger.debug(f"Could not load previous session: {e}")
    
    def _persist_message(self, role: str, content: str, context_summary: str = ""):
        """Save a message to the database for long-term memory."""
        if not self.db or not hasattr(self.db, "save_conversation_message"):
            return
        try:
            self.db.save_conversation_message(
                session_id=self.session_id,
                role=role,
                content=content,
                context_summary=context_summary[:500] if context_summary else ""
            )
        except Exception as e:
            logger.debug(f"Failed to persist message: {e}")
    
    def record_episode(
        self, episode_type: str, summary: str, keywords: str,
        context_data: Optional[Dict] = None, relevance_score: float = 1.0
    ):
        """
        Manually record an episodic memory (lesson learned).
        
        Args:
            episode_type: 'mistake', 'success', 'observation', 'seasonal'
            summary: Natural language description of the lesson
            keywords: Comma-separated keywords for retrieval
            context_data: Optional contextual data
            relevance_score: Importance (default 1.0, mistakes=2.0)
        """
        if not self.db or not hasattr(self.db, "save_episode"):
            logger.warning("Cannot save episode — no database connection.")
            return
        
        self.db.save_episode(
            episode_type=episode_type,
            summary=summary,
            keywords=keywords,
            context_data=context_data,
            relevance_score=relevance_score
        )
    
    def recall_relevant_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Search episodic memory for lessons relevant to a query.
        Used internally to enhance LLM context.
        """
        if not self.db or not hasattr(self.db, "search_episodes"):
            return []
        try:
            return self.db.search_episodes(query=query, limit=limit)
        except Exception:
            return []
