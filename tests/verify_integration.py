import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from julia.simulator.sim_engine import SimulationEngine, SimulationState

def verify_integration():
    print("Initializing SimulationEngine...")
    sim = SimulationEngine()
    
    print("Starting simulation...")
    sim.start()
    
    # Tick simulation (e.g. 10 times)
    # We force multiple ticks to trigger weather updates and decision cycles
    # sim.speed is 1.0 (1 hour per second).
    # Decision cycle is every 6 sim-hours.
    # To trigger it quickly, we can manually advance elapsed_hours or just call tick repeatedly?
    # tick() uses real time delta.
    
    # Let's force tick logic manually for testing speed
    sim.speed = 3600 # 1 hour per second -> 1 hour per 1/3600 second? No.
    # speed = sim-hours per real-second.
    # If we want 6 hours to pass instantly, we can just cheat or spin.
    
    # Cheat: manually invoke _run_decision_cycle
    print("Forcing weather update...")
    sim._update_weather(1.0)
    print(f"Weather: {sim.weather}")
    
    print("Forcing decision cycle...")
    sim._run_decision_cycle()
    
    print("Checking events...")
    for event in sim.events:
        print(f"EVENT: {event.message}")
        
    print("Checking database...")
    readings = sim.db.get_recent_readings("basil", 5)
    print(f"Recent readings for Basil: {len(readings)}")
    for r in readings:
        print(r)
        
    decisions = sim.db.get_recent_decisions(5)
    print(f"Recent decisions: {len(decisions)}")
    for d in decisions:
        print(d)

    assert len(readings) > 0, "No readings logged!"
    assert len(decisions) > 0, "No decisions logged!"
    
    print("VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    verify_integration()
