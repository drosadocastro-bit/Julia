"""
julia_main.py — Main entry point for the Julia AI Crop Caretaker.

Usage:
    python -m julia.julia_main run          # Start scheduler (continuous)
    python -m julia.julia_main check        # Single watering check cycle
    python -m julia.julia_main check --dry-run  # Simulate with mock data
    python -m julia.julia_main status       # Show current sensor status
    python -m julia.julia_main water basil 200  # Manual water: plant_id amount_ml
    python -m julia.julia_main history      # Show watering history
    python -m julia.julia_main export       # Export ML training data CSV

Julia: Because every plant deserves a chance to thrive. 🌱💚
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

# Fix Windows console encoding for emoji output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

from julia.core.config import JuliaConfig
from julia.core.decision_engine import (
    JuliaDecisionEngine,
    WaterDecision,
    WeatherForecast,
)
from julia.core.weather_service import WeatherService
from julia.core.scheduler import JuliaScheduler
from julia.sensors.ha_client import HomeAssistantClient
from julia.sensors.sensor_reader import SensorReader, SensorData
from julia.sensors.data_validator import DataValidator
from julia.actuators.watering import WateringController
from julia.notifications.notifier import Notifier, AlertLevel
from julia.data.database import JuliaDatabase
from julia.data.ml_collector import MLCollector
from julia.core.llm_brain import JuliaBrain
from julia.core.llm_config import get_llm_config


# ======================================================================
# Logging Setup
# ======================================================================

def setup_logging(level: str = "INFO"):
    """Configure structured logging for Julia."""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


logger = logging.getLogger("julia.main")


# ======================================================================
# Julia Application
# ======================================================================

class JuliaApp:
    """
    Main application class — wires all modules together.

    This is the orchestrator that connects sensors, decisions,
    actuators, notifications, database, and ML collection.
    """

    BANNER = """
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║   🌱  JULIA — AI Crop Caretaker                      ║
    ║                                                       ║
    ║   En honor a Abuela Julia 🌺                         ║
    ║   "Every plant deserves a chance to thrive"           ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(self, config: Optional[JuliaConfig] = None):
        self.config = config or JuliaConfig()
        setup_logging(self.config.log_level)

        # Phase 2: SQLite database + ML collector
        self.db = JuliaDatabase()
        self.ml_collector = MLCollector(self.db)

        # Initialize modules (now with DB integration)
        self.ha = HomeAssistantClient(
            url=self.config.home_assistant.url,
            token=self.config.home_assistant.token,
            timeout=self.config.home_assistant.timeout,
        )
        self.sensors = SensorReader(self.config, self.ha)
        self.validator = DataValidator()
        self.decision_engine = JuliaDecisionEngine(self.config, db=self.db)
        self.weather = WeatherService(self.config, db=self.db)
        self.watering = WateringController(self.config, self.ha)
        self.notifier = Notifier(self.config)
        self.scheduler = JuliaScheduler(self.config)

        # Register decision callback for DB logging + ML collection
        self.decision_engine.add_decision_callback(self._on_decision)

        # Phase 6: LLM Brain (lazy loaded in commands usually, but we can init here)
        self.llm_config = get_llm_config()
        self.brain = JuliaBrain(
            base_url=self.llm_config.base_url,
            model=self.llm_config.model,
            db=self.db
        )

    # ------------------------------------------------------------------
    # Decision Callback (Phase 2)
    # ------------------------------------------------------------------

    def _on_decision(self, plant_id, result, sensor_data, weather):
        """Callback fired after every decision — logs to DB and collects ML data."""
        try:
            # Log decision to database
            self.db.log_decision(
                plant_id=plant_id,
                decision=result.decision.value,
                water_amount_ml=result.water_amount_ml,
                reason=result.reason,
                confidence=result.confidence,
                soil_moisture=sensor_data.soil_moisture,
                temperature=sensor_data.temperature,
                humidity=sensor_data.humidity,
                rain_probability=weather.rain_probability_24h,
                weather_available=weather.is_available,
            )

            # Collect ML training sample
            hours = self.db.get_hours_since_watering(plant_id)
            days_since = (hours / 24) if hours is not None else 7.0
            self.ml_collector.record_decision(
                plant_id=plant_id,
                soil_moisture=sensor_data.soil_moisture,
                temperature=sensor_data.temperature,
                humidity=sensor_data.humidity,
                rain_probability=weather.rain_probability_24h,
                temp_forecast=weather.temp_high,
                days_since_water=days_since,
                growth_stage=1,  # Default to "growing" until Phase 3 vision
                action="water" if result.should_water() else "skip",
                water_amount_ml=result.water_amount_ml,
            )
        except Exception as e:
            logger.warning(f"Decision callback error: {e}")

    # ------------------------------------------------------------------
    # Core Pipeline
    # ------------------------------------------------------------------

    def watering_check_cycle(self):
        """
        Full watering decision cycle.

        1. Read sensors for all plants
        2. Validate readings + log to DB
        3. Fetch weather forecast
        4. Run decision engine (callbacks auto-log to DB + ML)
        5. Execute watering if needed
        6. Send notifications
        """
        logger.info("=" * 60)
        logger.info("🌱 Starting watering check cycle...")
        logger.info("=" * 60)

        # Step 1: Read sensors
        readings = self.sensors.read_all()
        if not readings:
            logger.warning("No sensor readings available!")
            self.notifier.warning("No sensor data available — check HA connection")
            return

        # Step 2: Validate + log to DB
        validations = self.validator.validate_batch(readings)
        for plant_id, data in readings.items():
            self.db.log_sensor_reading(
                plant_id=plant_id,
                moisture=data.soil_moisture,
                temperature=data.temperature,
                humidity=data.humidity,
                is_valid=data.is_valid,
                warnings=data.warnings,
            )
        for plant_id, result in validations.items():
            if result.has_errors:
                self.notifier.notify_sensor_failure(plant_id, result.errors)

        # Step 3: Get weather (auto-logs to DB via WeatherService)
        forecast = self.weather.get_forecast()

        # Step 4: Make decisions (callbacks auto-log to DB + collect ML data)
        decisions = self.decision_engine.decide_all(readings, forecast)

        # Step 5 & 6: Execute and notify
        for plant_id, result in decisions.items():
            logger.info(f"Decision for {plant_id}: {result.decision.value} — {result.reason}")

            if result.decision == WaterDecision.EMERGENCY:
                self.notifier.notify_emergency(plant_id, result.reason)

            if result.should_water():
                event = self.watering.water(
                    plant_id=plant_id,
                    amount_ml=result.water_amount_ml,
                    reason=result.reason,
                )
                if event.success:
                    self.decision_engine.record_watering(plant_id)
                    self.db.log_watering_event(
                        plant_id=plant_id,
                        amount_ml=result.water_amount_ml,
                        reason=result.reason,
                        decision_type=result.decision.value,
                        duration_seconds=event.duration_seconds,
                        moisture_before=readings[plant_id].soil_moisture,
                    )
                    self.notifier.notify_decision(plant_id, result.reason, True)
                else:
                    self.notifier.warning(
                        f"Watering failed for {plant_id}!",
                        title="Julia — Watering Error"
                    )
            else:
                self.notifier.notify_decision(plant_id, result.reason, False)

        logger.info("✅ Watering check cycle complete.")

    def sensor_read_cycle(self):
        """Quick sensor read and log to DB (no decisions)."""
        readings = self.sensors.read_all()
        for plant_id, data in readings.items():
            self.db.log_sensor_reading(
                plant_id=plant_id,
                moisture=data.soil_moisture,
                temperature=data.temperature,
                humidity=data.humidity,
                is_valid=data.is_valid,
                warnings=data.warnings,
            )
            logger.info(
                f"📊 {plant_id}: moisture={data.soil_moisture}%, "
                f"temp={data.temperature}°C, humidity={data.humidity}%"
            )

    def outcome_check_cycle(self):
        """Check 24h outcomes for ML training data."""
        readings = self.sensors.read_all()
        current = {
            pid: {"soil_moisture": d.soil_moisture}
            for pid, d in readings.items()
        }
        recorded = self.ml_collector.check_outcomes(current)
        if recorded:
            logger.info(f"📈 Recorded {recorded} ML outcomes")

    # ------------------------------------------------------------------
    # CLI Commands
    # ------------------------------------------------------------------

    def cmd_run(self):
        """Start Julia in continuous scheduler mode."""
        print(self.BANNER)
        logger.info(f"Config: {self.config}")

        # Run one check immediately on startup
        logger.info("Running initial watering check...")
        self.watering_check_cycle()

        # Then start scheduler
        self.scheduler.setup(
            watering_check=self.watering_check_cycle,
            sensor_read=self.sensor_read_cycle,
            outcome_check=self.outcome_check_cycle,
        )
        self.scheduler.start()

    def cmd_check(self, dry_run: bool = False):
        """Run a single watering check cycle."""
        print(self.BANNER)

        if dry_run:
            self.config.dry_run = True
            logger.info("🏜️ DRY RUN MODE — no actual watering will occur.")

            # Use mock data if HA is unreachable
            if not self.sensors.is_ha_connected():
                logger.info("HA not connected — using mock sensor data for dry run.")
                self._mock_check_cycle()
                return

        self.watering_check_cycle()

    def cmd_status(self):
        """Show current system status."""
        print(self.BANNER)
        print(f"  Config: {self.config}\n")

        # HA connection
        ha_ok = self.sensors.is_ha_connected()
        ha_status = "✅ Connected" if ha_ok else "❌ Unreachable"
        print(f"  Home Assistant: {ha_status}")
        print(f"    URL: {self.config.home_assistant.url}\n")

        # Plant profiles
        print(f"  Plant Profiles ({len(self.config.plant_profiles)}):")
        for pid, profile in self.config.plant_profiles.items():
            print(f"    {profile.emoji} {profile.name} — moisture: {profile.min_moisture}-{profile.max_moisture}%")
        print()

        # Sensor mappings
        print(f"  Sensor Mappings ({len(self.config.sensor_mappings)}):")
        for pid, mapping in self.config.sensor_mappings.items():
            print(f"    {pid}: {mapping.sensor_entity_id}")
        print()

        # Weather
        print(f"  Weather: {self.config.weather.location_name}")
        print(f"    API Key: {'✅ Set' if self.config.weather.api_key else '❌ Not set'}")
        print()

        # Database stats
        ml_stats = self.ml_collector.get_stats()
        print(f"  📊 Database: {self.db.db_path}")
        print(f"    ML samples: {ml_stats['total_samples']} ({ml_stats['completed']} completed)")
        print()

        # Live readings (if HA is up)
        if ha_ok:
            print("  📊 Live Sensor Readings:")
            readings = self.sensors.read_all()
            for pid, data in readings.items():
                status = "✅" if data.is_valid else "⚠️"
                print(
                    f"    {status} {pid}: moisture={data.soil_moisture}%, "
                    f"temp={data.temperature}°C, humidity={data.humidity}%"
                )
        print()

    def cmd_water(self, plant_id: str, amount_ml: int):
        """Manually water a specific plant."""
        print(self.BANNER)
        logger.info(f"Manual watering: {plant_id} — {amount_ml}ml")

        profile = self.config.get_profile(plant_id)
        event = self.watering.water(
            plant_id=plant_id,
            amount_ml=amount_ml,
            reason=f"Manual watering requested by user",
        )

        if event.success:
            self.decision_engine.record_watering(plant_id)
            self.db.log_watering_event(
                plant_id=plant_id,
                amount_ml=amount_ml,
                reason="Manual watering requested by user",
                decision_type="manual",
                duration_seconds=event.duration_seconds,
            )
            print(f"\n  ✅ Watered {profile.emoji} {profile.name}: {amount_ml}ml over {event.duration_seconds:.0f}s")
        else:
            print(f"\n  ❌ Watering failed! Check HA connection.")

    def cmd_history(self, plant_id: Optional[str] = None, days: int = 7):
        """Show watering history and ML stats."""
        print(self.BANNER)
        target = plant_id or "all plants"
        print(f"  📜 Watering History ({target}, last {days} days):")
        print("  " + "─" * 55)

        history = self.db.get_watering_history(plant_id=plant_id, days=days)
        if not history:
            print("    No watering events found.")
        else:
            for event in history:
                ts = event.get('timestamp', 'unknown')[:16]
                pid = event.get('plant_id', '?')
                ml = event.get('amount_ml', 0)
                reason = event.get('decision_type', '')
                print(f"    {ts}  {pid:<10s} {ml:>4d}ml  ({reason})")

        print("  " + "─" * 55)

        # Daily stats
        stats = self.db.get_daily_stats(days=days)
        if stats:
            print(f"\n  📊 Daily Summary:")
            for day in stats:
                print(
                    f"    {day['date']}  {day['total_waterings']} waterings, "
                    f"{day['total_water_ml'] or 0}ml total"
                )

        # ML stats
        ml_stats = self.ml_collector.get_stats()
        print(f"\n  🧠 ML Training Data:")
        print(f"    Total samples: {ml_stats['total_samples']}")
        print(f"    Completed (with outcomes): {ml_stats['completed']}")
        print(f"    Pending outcomes: {ml_stats['pending']}")
        print()

    def cmd_export(self):
        """Export ML training data as CSV."""
        print(self.BANNER)
        filepath = self.ml_collector.export_csv()
        stats = self.ml_collector.get_stats()
        print(f"  📤 ML Training Data Export")
        print(f"    File: {filepath}")
        print(f"    Samples: {stats['completed']} completed, {stats['pending']} pending")
        print()

    def cmd_chat(self):
        """Interactive chat with Julia."""
        print(self.BANNER)
        print("  🌱 Chat with Julia about your garden!")
        print(f"  🧠 Model: {self.llm_config.model} @ {self.llm_config.base_url}")
        print("  Type 'quit' to exit, 'clear' to reset conversation\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 See you in the garden! 🌱")
                break
            
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 See you in the garden! 🌱")
                break
            if user_input.lower() == "clear":
                self.brain.clear_history()
                print("🔄 Conversation cleared!\n")
                continue
            
            print("\nJulia: ", end="", flush=True)
            response = self.brain.chat(user_input)
            print(response)
            print()

    def cmd_briefing(self):
        """Get Julia's morning garden briefing."""
        print(self.BANNER)
        print("  ☀️ Julia's Garden Briefing")
        print("  " + "─" * 40)
        print("  Thinking...", end="\r")
        response = self.brain.generate_daily_briefing()
        print(f"\n{response}\n")

    # ------------------------------------------------------------------
    # Mock Data (for dry runs without HA)
    # ------------------------------------------------------------------

    def _mock_check_cycle(self):
        """Run a decision cycle with mock sensor data for testing."""
        logger.info("Using mock sensor data...")

        mock_readings = {
            "basil": SensorData(
                soil_moisture=35.0, temperature=29.0, humidity=72.0,
                sensor_id="basil", timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            "pepper": SensorData(
                soil_moisture=55.0, temperature=31.0, humidity=68.0,
                sensor_id="pepper", timestamp=datetime.now(timezone.utc).isoformat(),
            ),
            "tomato": SensorData(
                soil_moisture=80.0, temperature=28.0, humidity=75.0,
                sensor_id="tomato", timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        }

        forecast = WeatherForecast(
            rain_probability_24h=25, rain_probability_48h=40,
            temp_high=32, temp_low=24, humidity=70,
            description="Partly cloudy", is_available=True,
        )

        decisions = self.decision_engine.decide_all(mock_readings, forecast)

        print("\n  📋 Mock Decision Results:")
        print("  " + "─" * 50)
        for plant_id, result in decisions.items():
            icon = "💧" if result.should_water() else "⏸️"
            print(f"    {icon} {result.reason}")
        print("  " + "─" * 50)
        print()


# ======================================================================
# CLI Entry Point
# ======================================================================

def main():
    """CLI argument parsing and dispatch."""
    parser = argparse.ArgumentParser(
        description="Julia — AI Crop Caretaker 🌱",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Julia: Because every plant deserves a chance to thrive. 🌱💚\nEn memoria de Abuela Julia 🌺",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Start continuous scheduler")

    # check
    check_parser = subparsers.add_parser("check", help="Run single watering check")
    check_parser.add_argument("--dry-run", action="store_true", help="Simulate without watering")

    # status
    status_parser = subparsers.add_parser("status", help="Show system status")

    # water
    water_parser = subparsers.add_parser("water", help="Manual watering")
    water_parser.add_argument("plant_id", help="Plant ID (e.g., basil)")
    water_parser.add_argument("amount_ml", type=int, help="Water amount in ml")

    # history (Phase 2)
    history_parser = subparsers.add_parser("history", help="Show watering history")
    history_parser.add_argument("--plant", help="Filter by plant ID")
    history_parser.add_argument("--days", type=int, default=7, help="Days of history (default: 7)")

    # export (Phase 2)
    export_parser = subparsers.add_parser("export", help="Export ML training data as CSV")

    # chat (Phase 6)
    chat_parser = subparsers.add_parser("chat", help="Chat with Julia")
    
    # briefing (Phase 6)
    briefing_parser = subparsers.add_parser("briefing", help="Get garden briefing")

    # Common args
    parser.add_argument("--config", help="Path to plants.json config")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create app
    config = JuliaConfig(config_path=args.config) if args.config else JuliaConfig()
    app = JuliaApp(config)

    # Dispatch
    if args.command == "run":
        app.cmd_run()
    elif args.command == "check":
        app.cmd_check(dry_run=args.dry_run)
    elif args.command == "status":
        app.cmd_status()
    elif args.command == "water":
        app.cmd_water(args.plant_id, args.amount_ml)
    elif args.command == "history":
        app.cmd_history(plant_id=args.plant, days=args.days)
    elif args.command == "export":
        app.cmd_export()
    elif args.command == "chat":
        app.cmd_chat()
    elif args.command == "briefing":
        app.cmd_briefing()


if __name__ == "__main__":
    main()
