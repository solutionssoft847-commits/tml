import cv2
import numpy as np
import threading
import queue
import time
import os
from typing import Optional, Dict, List, Tuple
import re

# Try to import ffmpegcv
try:
    import ffmpegcv
    # extra check: sometimes import works but init fails if ffmpeg exe missing
    FFMPEG_AVAILABLE = True
except (ImportError, RuntimeError, Exception) as e:
    FFMPEG_AVAILABLE = False
    print(f"ffmpegcv not available ({e}), falling back to cv2")

class QualityFilter:
    """Check image quality (blur detection + brightness)"""
    def __init__(self, threshold: float = 80.0, min_brightness: float = 30.0):
        self.threshold = threshold
        self.min_brightness = min_brightness
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate Laplacian Variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def check_quality(self, image: np.ndarray) -> Tuple[bool, str, float]:
        """
        Returns: (is_good, reason, score)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        brightness = np.mean(gray)
        if brightness < self.min_brightness:
            return False, "DARK", brightness
            
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < self.threshold:
            return False, "BLURRY", sharpness
            
        return True, "GOOD", sharpness

class MotionGate:
    """
    detect -> wait for stability -> trigger
    State Machine:
    0: IDLE
    1: MOTION_DETECTED (Resets timer)
    2: STABLE (Waiting for duration)
    3: TRIGGER_READY
    """
    def __init__(self, stability_duration: float = 0.5, threshold: int = 25):
        self.stability_duration = stability_duration
        self.pixel_threshold = threshold
        
        self.prev_frame = None
        self.last_motion_time = 0
        self.state = "IDLE" 
        self.motion_score = 0.0
    
    def process(self, frame: np.ndarray) -> str:
        """
        Returns: 'IDLE', 'MOVING', 'STABILIZING', 'READY'
        """
        now = time.time()
        
        # 1. Resize for fast processing (128x128)
        small = cv2.resize(frame, (128, 128))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return "IDLE"
            
        # 2. Frame Delta
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pixels = cv2.countNonZero(thresh)
        
        self.motion_score = motion_pixels
        self.prev_frame = gray
        
        # 3. State Machine
        if motion_pixels > self.pixel_threshold:
            self.state = "MOVING"
            self.last_motion_time = now
            return "MOVING"
        else:
            # No current motion
            elapsed = now - self.last_motion_time
            
            if self.state == "MOVING":
                self.state = "STABILIZING"
            
            if elapsed >= self.stability_duration:
                if self.state != "READY":
                    self.state = "READY"
                    return "READY" # Trigger transition
                return "IDLE" # Already triggered or just idle
            else:
                return "STABILIZING"

class StreamLoader:
    """Non-blocking frame loader with Circular Buffer (Size 1)"""
    def __init__(self, source, is_ffmpeg=False, resize=(640, 640)):
        self.source = source
        self.is_ffmpeg = is_ffmpeg and FFMPEG_AVAILABLE
        self.resize_dim = resize
        
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.last_read_time = 0
        
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        print(f" Camera Thread Started (Source: {self.source}, FFmpeg: {self.is_ffmpeg})")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _update(self):
        """Thread loop"""
        if self.is_ffmpeg:
            cap = ffmpegcv.VideoCapture(self.source)
        else:
            cap = cv2.VideoCapture(self.source)
            # Optimize CV2 buffer
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print(f" Failed to open camera: {self.source}")
            self.running = False
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(" Stream end ")
                break
                
            # Pre-process immediately
            if self.resize_dim:
                frame = cv2.resize(frame, self.resize_dim)
            
            with self.lock:
                self.latest_frame = frame
                self.last_read_time = time.time()
            
            # Reduce CPU usage slightly
            time.sleep(0.001)
            
        cap.release()
        print(" Camera Thread Stopped")

    def read(self):
        with self.lock:
            return self.latest_frame if self.latest_frame is not None else None


class CameraManager:
    def __init__(self, is_cloud: bool = False, fps: int = 30, buffer_size: int = 1):
        self.is_cloud = is_cloud
        self.stream_loader = None
        self.current_id = None
        self.camera_type = None
        
        # Tools
        self.motion_gate = MotionGate(stability_duration=0.7) # 700ms default
        self.quality_filter = QualityFilter(threshold=100.0)
        
        # Output
        self.current_status = "IDLE" # For UI
        
        print(f" CameraManager initialized (cloud={is_cloud})")

    async def get_available_cameras(self) -> List[Dict]:
        """Get list of available cameras"""
        cameras = []
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        
        if not self.is_cloud:
            for i in range(2): # Check first 2 only for speed
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    is_active = (self.current_id == i)
                    cameras.append({
                        "id": i,
                        "name": f"USB Camera {i}" + (" (Active)" if is_active else ""),
                        "type": "usb"
                    })
                    cap.release()
        
        # IP Camera placeholder
        cameras.append({"id": "ip", "name": "IP Camera (Add New)", "type": "ip"})
        return cameras

    async def select_camera(self, camera_id) -> bool:
        """Switch camera"""
        self.release_camera()
        
        source = None
        if isinstance(camera_id, int):
            source = camera_id
            self.camera_type = 'usb'
        elif isinstance(camera_id, str) and (camera_id.startswith('rtsp') or camera_id.startswith('http')):
            source = camera_id
            self.camera_type = 'ip'
        elif str(camera_id).isdigit():
             source = int(camera_id) # Handle string int
             self.camera_type = 'usb'
        else:
             print(f"Invalid camera ID: {camera_id}")
             return False

        # Init StreamLoader
        # Prefer ffmpeg for IP/RTSP, cv2 for USB usually better/simpler
        use_ffmpeg = (self.camera_type == 'ip')
        
        self.stream_loader = StreamLoader(source, is_ffmpeg=use_ffmpeg, resize=(640, 640))
        self.stream_loader.start()
        
        self.current_id = camera_id
        return True

    def release_camera(self):
        if self.stream_loader:
            self.stream_loader.stop()
            self.stream_loader = None
        self.current_id = None
        print(" Camera released")

    def get_frame(self):
        """
        Get latest frame AND process motion/quality logic
        Returns: (frame, is_high_quality_capture)
        """
        if not self.stream_loader:
            return None, False
            
        frame = self.stream_loader.read()
        if frame is None:
            return None, False
            
        # Motion Gating (DISABLED for Live Inspection Removal)
        # motion_state = self.motion_gate.process(frame)
        # self.current_status = motion_state
        
        should_capture = False
        
        # if motion_state == "READY":
        #     # Motion stopped -> check quality
        #     is_good, reason, score = self.quality_filter.check_quality(frame)
        #     if is_good:
        #         should_capture = True
        #         print(f" [CAPTURE] Motion Stable ({self.motion_gate.stability_duration}s) & Sharp (Score: {score:.1f})")
        #     else:
        #         print(f" [SKIP] Stable but {reason} (Score: {score:.1f})")
        #         
        return frame, should_capture

    def test_connection(self, camera_id) -> bool:
        """Test if a camera is reachable without switching to it"""
        try:
            # Force CV2 for quick test
            if isinstance(camera_id, str) and camera_id.isdigit():
                camera_id = int(camera_id)
            
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        except Exception as e:
            print(f"Connection test error: {e}")
            return False

    def is_active(self) -> bool:
        """Check if camera is active"""
        return self.stream_loader is not None and self.stream_loader.running
    
    def get_info(self) -> Dict:
        """Get camera info"""
        if not self.is_active():
            return {
                "active": False,
                "id": None,
                "type": None
            }
        
        info = {
            "active": True,
            "id": self.current_id,
            "type": self.camera_type,
            "consecutive_errors": 0 # StreamLoader handles errors internally/silently for now
        }
        
        return info