
import cv2
import time
import numpy as np
from datetime import datetime

class CameraService:
    """
    Handles camera interaction.
    If a real camera is not available, it generates a synthetic image.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self._last_frame = None
        self._last_capture_time = 0
        
        # Try to initialize camera
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print(f"Warning: Camera index {camera_index} not available. Using mock mode.")
                self.cap = None
        except Exception as e:
            print(f"Error initializing camera: {e}. Using mock mode.")
            self.cap = None

    def capture(self) -> bytes:
        """Capture a frame and return it as JPEG bytes."""
        frame = None
        
        # 1. Try real camera
        if self.cap and self.cap.isOpened():
            ret, read_frame = self.cap.read()
            if ret:
                frame = read_frame
        
        # 2. Fallback to mock if needed
        if frame is None:
            frame = self._generate_mock_frame()

        # 3. Add timestamp overlay
        cv2.putText(
            frame, 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        # 4. Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def _generate_mock_frame(self):
        """Generate a placeholder image (green noise)."""
        # Create a 640x480 black image
        img = np.zeros((480, 640, 3), np.uint8)
        
        # Add some "green" noise to simulate plants
        noise = np.random.randint(0, 50, (480, 640), dtype=np.uint8)
        img[:, :, 1] = noise + 50 # Green channel base
        
        # Draw a "Plant" circle
        cv2.circle(img, (320, 240), 100, (0, 200, 0), -1)
        
        # Add text
        cv2.putText(img, "SIMULATION MODE", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img

    def release(self):
        if self.cap:
            self.cap.release()
