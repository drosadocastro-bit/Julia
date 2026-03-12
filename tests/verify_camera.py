
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock cv2 if not present to avoid ImportError in test if opencv not installed yet
try:
    import cv2
except ImportError:
    print("OpenCV not installed, skipping test logic that requires it, but we can test the fallback structure.")

from julia.vision.camera import CameraService

def verify_camera():
    print("Initializing CameraService...")
    cam = CameraService()
    
    print("Capturing frame...")
    frame_bytes = cam.capture()
    
    print(f"Captured {len(frame_bytes)} bytes.")
    
    # Check for JPEG header
    if frame_bytes.startswith(b'\xff\xd8'):
        print("Valid JPEG header detected.")
    else:
        print("ERROR: Invalid image format (not JPEG).")
        exit(1)

    print("VERIFICATION SUCCESSFUL")

if __name__ == "__main__":
    verify_camera()
