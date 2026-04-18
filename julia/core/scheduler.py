"""
julia.core.scheduler — Task scheduler for Julia.

Runs periodic checks:
- Sensor reading (every 15 min)
- Watering decision cycle (every 6 hours)
- Vision health check (every 12 hours) — Phase 3

Uses APScheduler for reliable background task execution.
"""

import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Callable, Optional

from julia.core.config import JuliaConfig

logger = logging.getLogger("julia.scheduler")


class JuliaScheduler:
    """
    Manages periodic task execution for Julia.

    Wraps APScheduler with Julia-specific job setup.
    Handles graceful shutdown on SIGINT/SIGTERM.
    """

    def __init__(self, config: JuliaConfig):
        self.config = config
        self._scheduler = None
        self._running = False

    def _create_scheduler(self):
        """Create the APScheduler instance."""
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            from apscheduler.triggers.interval import IntervalTrigger

            self._scheduler = BlockingScheduler()
            return True
        except ImportError:
            logger.error(
                "APScheduler not installed. Install it with: pip install apscheduler"
            )
            return False

    def setup(
        self,
        watering_check: Callable,
        sensor_read: Optional[Callable] = None,
        vision_check: Optional[Callable] = None,
        outcome_check: Optional[Callable] = None,
    ):
        """
        Set up scheduled jobs.

        Args:
            watering_check: Function to run for watering decisions
            sensor_read: Optional function for periodic sensor reads
            vision_check: Optional function for vision health checks (Phase 3)
            outcome_check: Optional function for ML outcome recording (Phase 2)
        """
        if not self._create_scheduler():
            return

        from apscheduler.triggers.interval import IntervalTrigger

        # Main watering check — every N hours
        self._scheduler.add_job(
            watering_check,
            trigger=IntervalTrigger(hours=self.config.schedule.check_interval_hours),
            id="watering_check",
            name=f"Watering Check (every {self.config.schedule.check_interval_hours}h)",
            max_instances=1,
            replace_existing=True,
        )
        logger.info(
            f"📅 Scheduled watering check every {self.config.schedule.check_interval_hours}h"
        )

        # Sensor read — every N minutes (if provided)
        if sensor_read:
            self._scheduler.add_job(
                sensor_read,
                trigger=IntervalTrigger(minutes=self.config.schedule.sensor_read_interval_minutes),
                id="sensor_read",
                name=f"Sensor Read (every {self.config.schedule.sensor_read_interval_minutes}min)",
                max_instances=1,
                replace_existing=True,
            )
            logger.info(
                f"📅 Scheduled sensor read every {self.config.schedule.sensor_read_interval_minutes}min"
            )

        # Vision check — every N hours (Phase 3, if provided)
        if vision_check:
            self._scheduler.add_job(
                vision_check,
                trigger=IntervalTrigger(hours=self.config.schedule.vision_check_interval_hours),
                id="vision_check",
                name=f"Vision Check (every {self.config.schedule.vision_check_interval_hours}h)",
                max_instances=1,
                replace_existing=True,
            )
            logger.info(
                f"📅 Scheduled vision check every {self.config.schedule.vision_check_interval_hours}h"
            )

        # ML outcome check — every 6 hours (Phase 2)
        if outcome_check:
            self._scheduler.add_job(
                outcome_check,
                trigger=IntervalTrigger(hours=6),
                id="outcome_check",
                name="ML Outcome Check (every 6h)",
                max_instances=1,
                replace_existing=True,
            )
            logger.info("📅 Scheduled ML outcome check every 6h")

    def start(self):
        """Start the scheduler (blocking)."""
        if not self._scheduler:
            logger.error("Scheduler not set up. Call setup() first.")
            return

        # Graceful shutdown
        def _signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received. Stopping Julia...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        self._running = True
        logger.info("🌱 Julia scheduler started. Press Ctrl+C to stop.")

        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Julia scheduler stopped.")
        finally:
            self._running = False

    def stop(self):
        """Stop the scheduler."""
        if self._scheduler and self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped.")

    def run_once(self, job_func: Callable):
        """Run a job immediately (for manual/CLI triggers)."""
        logger.info(f"Running {job_func.__name__} immediately...")
        try:
            job_func()
        except Exception as e:
            logger.error(f"Job {job_func.__name__} failed: {e}", exc_info=True)
