import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Tuple, Dict, Optional

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from reality_engine.config import Config
from reality_engine.core.state import AppState, HandData
from reality_engine.vision.model_utils import ensure_model

class HandTracker:
    def __init__(self):
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        model_path = ensure_model("hand_landmarker.task", model_url)
        
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=Config.MAX_NUM_HANDS,
            min_hand_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=Config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Connections for rendering skeleton
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        
        # For smoothing and velocity calculation
        # Map hand id (handedness + index) to previous landmarks and timestamps
        self.prev_landmarks: Dict[str, np.ndarray] = {}
        self.prev_times: Dict[str, float] = {}
        self.ema_alpha = 0.5  # Smoothing factor (1.0 = no smoothing, 0.0 = completely frozen)
        
    def process(self, state: AppState):
        """Process the current frame to extract hand landmarks and update state."""
        frame = state.frame
        if frame is None:
            return
            
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.detector.detect(mp_image)
        
        state.hands = []
        current_hand_ids = set()
        
        if results.hand_landmarks and results.handedness:
            for idx, (hand_landmarks, handedness_info) in enumerate(zip(results.hand_landmarks, results.handedness)):
                # MediaPipe handedness is flipped because we flip the image, 
                # but since we process on the flipped image, it might be correct or reversed.
                # Usually we want the visual left to be Left.
                label = handedness_info[0].category_name
                hand_id = f"{label}_{idx}"
                current_hand_ids.add(hand_id)
                
                # Extract normalized coordinates to a numpy array
                landmarks_normalized = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                
                # Apply EMA Smoothing
                if hand_id in self.prev_landmarks:
                    smoothed_landmarks = self.ema_alpha * landmarks_normalized + (1 - self.ema_alpha) * self.prev_landmarks[hand_id]
                else:
                    smoothed_landmarks = landmarks_normalized
                
                self.prev_landmarks[hand_id] = smoothed_landmarks
                
                # Convert to pixel coordinates
                pixel_landmarks = np.zeros((21, 2), dtype=np.int32)
                for i in range(21):
                    pixel_landmarks[i] = [
                        int(smoothed_landmarks[i][0] * w),
                        int(smoothed_landmarks[i][1] * h)
                    ]
                
                # Calculate palm center (average of wrist, index base, pinky base)
                wrist = pixel_landmarks[0]
                index_mcp = pixel_landmarks[5]
                pinky_mcp = pixel_landmarks[17]
                palm_center = (
                    int((wrist[0] + index_mcp[0] + pinky_mcp[0]) / 3),
                    int((wrist[1] + index_mcp[1] + pinky_mcp[1]) / 3)
                )
                
                # Calculate velocity (based on palm center)
                velocity = (0.0, 0.0)
                current_time = state.timestamp
                
                # For velocity, we need previous palm center. Let's just track palm center specifically or compute it from prev_landmarks.
                if hand_id in self.prev_times:
                    dt = current_time - self.prev_times[hand_id]
                    if dt > 0:
                        prev_norm = self.prev_landmarks[hand_id]
                        prev_wrist = prev_norm[0]
                        prev_index = prev_norm[5]
                        prev_pinky = prev_norm[17]
                        prev_palm_x = (prev_wrist[0] + prev_index[0] + prev_pinky[0]) / 3 * w
                        prev_palm_y = (prev_wrist[1] + prev_index[1] + prev_pinky[1]) / 3 * h
                        
                        vx = (palm_center[0] - prev_palm_x) / dt
                        vy = (palm_center[1] - prev_palm_y) / dt
                        velocity = (vx, vy)
                
                self.prev_times[hand_id] = current_time
                
                # Determine pinch distance for later modules
                thumb_tip = pixel_landmarks[4]
                index_tip = pixel_landmarks[8]
                pinch_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
                
                # Placeholder for gesture booleans, to be populated by gesture engine
                hand_data = HandData(
                    landmarks=smoothed_landmarks,
                    pixel_landmarks=pixel_landmarks,
                    handedness=label,
                    palm_center=palm_center,
                    pinch_distance=pinch_dist,
                    is_pinching=False,
                    is_fist=False,
                    is_open_palm=False,
                    is_pointing=False,
                    velocity=velocity
                )
                
                state.hands.append(hand_data)
                
                # Render if debug mode
                if state.debug_mode:
                    # Draw connections
                    for connection in self.connections:
                        start_idx = connection.start
                        end_idx = connection.end
                        pt1 = tuple(pixel_landmarks[start_idx])
                        pt2 = tuple(pixel_landmarks[end_idx])
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                        
                    # Draw landmarks
                    for pt in pixel_landmarks:
                        cv2.circle(frame, tuple(pt), 4, (0, 0, 255), -1)
                    
                    # Draw Palm Center
                    cv2.circle(frame, palm_center, 6, (255, 0, 0), -1)
                    
                    # Draw velocity vector
                    end_pt = (
                        int(palm_center[0] + velocity[0] * 0.1),
                        int(palm_center[1] + velocity[1] * 0.1)
                    )
                    cv2.arrowedLine(frame, palm_center, end_pt, (0, 255, 255), 2, tipLength=0.2)
                    
        # Cleanup lost hands
        lost_hands = set(self.prev_landmarks.keys()) - current_hand_ids
        for h_id in lost_hands:
            del self.prev_landmarks[h_id]
            del self.prev_times[h_id]
