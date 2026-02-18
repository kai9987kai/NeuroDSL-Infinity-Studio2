import psutil
import torch
import cv2
import threading
import time
import numpy as np
from typing import Dict, Any, Optional

class SensorNexus:
    """Monitors system telemetry and provides it as a data source for models."""
    
    @staticmethod
    def get_system_telemetry() -> Dict[str, float]:
        """Collects current CPU, GPU (simulated), and RAM usage."""
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        # Simulate GPU if torch.cuda is not used or available
        gpu_usage = 0.0
        if torch.cuda.is_available():
            # This is a guestimate based on simple usage
            gpu_usage = 10.0 # Placeholder for actual NVML query if needed
            
        return {
            "cpu_percent": cpu_usage,
            "ram_percent": ram_usage,
            "gpu_percent": gpu_usage,
            "disk_usage": psutil.disk_usage('/').percent
        }

    @staticmethod
    def telemetry_to_tensor() -> torch.Tensor:
        """Converts telemetry data to a torch tensor for architectural input."""
        data = SensorNexus.get_system_telemetry()
        values = [data["cpu_percent"], data["ram_percent"], data["gpu_percent"], data["disk_usage"]]
        return torch.tensor(values, dtype=torch.float32)

class VisionStream:
    """Handles camera integration and live video processing."""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.is_running = False
        self.last_frame = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.is_running:
            return
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Camera {self.camera_id} not available.")
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.last_frame = frame
            time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.last_frame

    def get_tensor(self, size=(224, 224)) -> Optional[torch.Tensor]:
        """Converts the current frame to a normalized torch tensor [1, 3, H, W]."""
        frame = self.get_frame()
        if frame is None:
            return None
        
        resized = cv2.resize(frame, size)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)
