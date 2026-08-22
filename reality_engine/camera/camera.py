import cv2
import threading
import time
import sys
import numpy as np
from typing import Tuple, Optional

class ThreadedCamera:
    """Threaded webcam capture with macOS support, retry logic, and synthetic fallback."""
    
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720, target_fps: int = 30):
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.lock = threading.Lock()
        
        self.stream = self._init_camera(src)
        self.is_synthetic = (self.stream is None)
        
        if not self.is_synthetic:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.stream.set(cv2.CAP_PROP_FPS, self.target_fps)
            (self.grabbed, self.frame) = self.stream.read()
            
    def _init_camera(self, primary_src: int):
        # On macOS, try to use AVFoundation specifically
        is_mac = sys.platform == 'darwin'
        indices_to_try = [primary_src, 0, 1] if primary_src != 0 else [0, 1]
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY] if is_mac else [cv2.CAP_ANY]
        
        for idx in indices_to_try:
            for backend in backends:
                print(f"Attempting to open camera {idx} with backend {backend}...")
                cap = cv2.VideoCapture(idx, backend)
                if cap is not None and cap.isOpened():
                    # Read a test frame to ensure it actually works
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Successfully initialized camera {idx} with backend {backend}.")
                        return cap
                    else:
                        cap.release()
                elif cap is not None:
                    cap.release()
                    
        print("WARNING: Could not initialize any physical camera. Using synthetic fallback.")
        return None

    def start(self):
        """Start the thread to read frames from the video stream."""
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def _update(self):
        """Keep looping indefinitely until the thread is stopped."""
        frame_time = 1.0 / self.target_fps
        
        while not self.stopped:
            start_t = time.time()
            
            if self.is_synthetic:
                frame = self._generate_synthetic_frame()
                with self.lock:
                    self.grabbed = True
                    self.frame = frame
            else:
                grabbed, frame = self.stream.read()
                with self.lock:
                    self.grabbed = grabbed
                    if grabbed:
                        self.frame = frame
                        
                if not grabbed:
                    time.sleep(0.1)
                    continue
                    
            # Cap the framerate for synthetic or fast streams
            elapsed = time.time() - start_t
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    def _generate_synthetic_frame(self):
        """Generates a test pattern if no physical camera is available."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Moving gradient background
        t = time.time()
        c = int((np.sin(t) + 1) * 30) + 20
        frame[:, :] = (c, c, int(c*1.5))
        
        # Simulated "hand" (a moving circle)
        cx = int(self.width / 2 + np.sin(t * 2) * 200)
        cy = int(self.height / 2 + np.cos(t * 2) * 100)
        cv2.circle(frame, (cx, cy), 50, (200, 150, 100), -1)
        
        # Warning text
        text = "Camera permission required: System Settings -> Privacy & Security -> Camera"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "SYNTHETIC FALLBACK MODE", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return frame
                
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return the most recently read frame."""
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None
            
    def stop(self):
        """Indicate that the thread should be stopped."""
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if not self.is_synthetic and self.stream is not None:
            self.stream.release()
