import numpy as np
from collections import deque
from reality_engine.core.state import HandData, AppState
from reality_engine.gestures.base import BaseGestureDetector

class PinchDetector(BaseGestureDetector):
    def detect(self, hand: HandData, state: AppState) -> bool:
        # Distance between thumb tip (4) and index tip (8)
        # Normalize by palm size (distance from wrist to middle finger MCP)
        wrist = hand.pixel_landmarks[0]
        middle_mcp = hand.pixel_landmarks[9]
        palm_size = np.linalg.norm(np.array(wrist) - np.array(middle_mcp))
        
        if palm_size == 0:
            return False
            
        pinch_dist = hand.pinch_distance
        normalized_dist = pinch_dist / palm_size
        return normalized_dist < 0.4

class FistDetector(BaseGestureDetector):
    def detect(self, hand: HandData, state: AppState) -> bool:
        wrist = hand.pixel_landmarks[0]
        middle_mcp = hand.pixel_landmarks[9]
        palm_size = np.linalg.norm(np.array(wrist) - np.array(middle_mcp))
        
        if palm_size == 0:
            return False
            
        fingertips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
        for tip_id in fingertips:
            tip_to_wrist = np.linalg.norm(np.array(hand.pixel_landmarks[tip_id]) - np.array(wrist))
            if tip_to_wrist > palm_size * 1.5:
                return False
        return True

class OpenPalmDetector(BaseGestureDetector):
    def detect(self, hand: HandData, state: AppState) -> bool:
        # Fingers should be extended, distance from wrist to tips > wrist to MCPs
        wrist = hand.pixel_landmarks[0]
        fingertips = [8, 12, 16, 20]
        mcps = [5, 9, 13, 17]
        
        for tip_id, mcp_id in zip(fingertips, mcps):
            tip_dist = np.linalg.norm(np.array(hand.pixel_landmarks[tip_id]) - np.array(wrist))
            mcp_dist = np.linalg.norm(np.array(hand.pixel_landmarks[mcp_id]) - np.array(wrist))
            if tip_dist < mcp_dist * 1.5:
                return False
        return True

class PointDetector(BaseGestureDetector):
    def detect(self, hand: HandData, state: AppState) -> bool:
        wrist = hand.pixel_landmarks[0]
        index_tip = hand.pixel_landmarks[8]
        index_mcp = hand.pixel_landmarks[5]
        
        # Index extended
        if np.linalg.norm(np.array(index_tip) - np.array(wrist)) < np.linalg.norm(np.array(index_mcp) - np.array(wrist)) * 1.5:
            return False
            
        # Others curled
        other_tips = [12, 16, 20]
        other_mcps = [9, 13, 17]
        for tip_id, mcp_id in zip(other_tips, other_mcps):
            tip_dist = np.linalg.norm(np.array(hand.pixel_landmarks[tip_id]) - np.array(wrist))
            mcp_dist = np.linalg.norm(np.array(hand.pixel_landmarks[mcp_id]) - np.array(wrist))
            if tip_dist > mcp_dist * 1.5:
                return False
                
        return True

class CircleDetector(BaseGestureDetector):
    def __init__(self):
        self.history = {} # hand_id -> deque of points
        self.max_history = 30
        
    def detect(self, hand: HandData, state: AppState) -> bool:
        # Need a unique hand id, but state.hands doesn't carry it directly. 
        # For simplicity, use handedness and index (assume 1 per hand).
        # We can just match by handedness for single user.
        h_id = hand.handedness
        
        if h_id not in self.history:
            self.history[h_id] = deque(maxlen=self.max_history)
            
        # Use index tip for circle drawing
        point = hand.pixel_landmarks[8]
        self.history[h_id].append(point)
        
        if len(self.history[h_id]) < 20:
            return False
            
        # Calculate bounding box of history
        points = np.array(self.history[h_id])
        min_p = np.min(points, axis=0)
        max_p = np.max(points, axis=0)
        
        width = max_p[0] - min_p[0]
        height = max_p[1] - min_p[1]
        
        # Check if it's roughly circular/square bounding box and large enough
        if width < 50 or height < 50:
            return False
            
        aspect_ratio = width / max(1, height)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
            
        # Check start/end points are close
        dist = np.linalg.norm(points[0] - points[-1])
        if dist > min(width, height) * 0.5:
            return False
            
        # If detected, clear history to prevent immediate re-trigger
        self.history[h_id].clear()
        return True
