import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DetectedObject:
    bbox: Tuple[int, int, int, int] # x, y, w, h
    class_name: str
    confidence: float
    center: Tuple[int, int]

class ObjectDetector:
    def __init__(self):
        # We use the nano model for speed
        self.model = YOLO('yolov8n.pt')
        self.frame_skip = 5 # Run every 5 frames
        self.frame_count = 0
        self.last_objects: List[DetectedObject] = []
        
    def process(self, frame: np.ndarray) -> List[DetectedObject]:
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return self.last_objects
            
        results = self.model(frame, verbose=False)
        
        objects = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf > 0.5:
                    w = x2 - x1
                    h = y2 - y1
                    center = (x1 + w//2, y1 + h//2)
                    objects.append(DetectedObject(
                        bbox=(x1, y1, w, h),
                        class_name=result.names[cls],
                        confidence=conf,
                        center=center
                    ))
                    
        self.last_objects = objects
        return objects
