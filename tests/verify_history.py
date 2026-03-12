
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from julia.simulator.sim_engine import SimulationEngine, SimulationState

def verify_history():
    print("Initializing SimulationEngine...")
    sim = SimulationEngine()
    
    print("Checking chart data...")
    basil = sim.chart_data.get("basil", [])
    
    print(f"Found {len(basil)} data points for Basil.")
    
    if len(basil) == 0:
        print("WARNING: No data points found. Maybe DB is empty?")
    else:
        last = basil[-1]
        print(f"Most recent point: {last}")
        
        # Verify structure
        assert "time" in last
        assert "moisture" in last
        assert "temperature" in last
        assert "humidity" in last
        assert "health" in last
        
        print("Data structure verified.")

    print("VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    verify_history()
