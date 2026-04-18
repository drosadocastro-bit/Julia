"""
julia.simulator.server — Optimized HTTP server for the simulation dashboard.
"""

import json
import sys
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from julia.simulator.sim_engine import SimulationEngine

# Global simulation instance
sim = SimulationEngine()

class SimulatorHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the simulation dashboard."""
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.send_error(405, "API calls must be POST")
            return

        # Redirect legacy /dashboard paths to root
        if parsed.path in ["/dashboard", "/dashboard.html"]:
            self.send_response(301)
            self.send_header("Location", "/")
            self.end_headers()
            return
        
        # Default behavior: Serve static files from directory
        # This is more robust than manual implementation
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        
        # API Routes
        if parsed.path == "/api/state":
            self._serve_json(sim.get_state())
        elif parsed.path == "/api/start":
            sim.start()
            self._serve_json({"status": "running"})
        elif parsed.path == "/api/pause":
            sim.pause()
            self._serve_json({"status": "paused"})
        elif parsed.path == "/api/stop":
            sim.stop()
            self._serve_json({"status": "stopped"})
        elif parsed.path == "/api/tick":
            sim.tick()
            self._serve_json(sim.get_state())
        elif parsed.path == "/api/speed":
            body = self._read_body()
            sim.set_speed(body.get("speed", 1.0))
            self._serve_json({"speed": sim.speed})
        elif parsed.path == "/api/water":
            body = self._read_body()
            sim.manual_water(body.get("plant_id"), body.get("amount_ml", 200))
            self._serve_json({"status": "watered"})
        elif parsed.path == "/api/camera/latest":
            frame_bytes = sim.get_latest_image()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", len(frame_bytes))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(frame_bytes)
        elif parsed.path == "/api/chat":
            body = self._read_body()
            user_message = body.get("message", "")
            if not user_message:
                self.send_error(400, "No message provided")
                return
            
            response = sim.llm_brain.chat(user_message)
            self._serve_json({
                "response": response,
                "timestamp": "now" # Simplified
            })
        elif parsed.path == "/api/chat/clear":
            sim.llm_brain.clear_history()
            self._serve_json({"status": "cleared"})
        else:
            self.send_error(404, "Not Found")

    def _serve_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0: return {}
        try:
            return json.loads(self.rfile.read(length))
        except:
            return {}

    def log_message(self, format, *args):
        pass

def main():
    port = int(os.environ.get("JULIA_SIM_PORT", 8787))
    host = "0.0.0.0"
    
    # Change current directory to where index.html is
    os.chdir(Path(__file__).parent)

    print(f"JULIA Simulation Server active on http://{host}:{port}")
    server = HTTPServer((host, port), SimulatorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == "__main__":
    main()
