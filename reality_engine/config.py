import cv2

class Config:
    # Camera Settings
    CAMERA_ID = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    TARGET_FPS = 30
    
    # Hand Tracking
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Gestures
    DEBOUNCE_TIME = 0.2  # seconds
    
    # Rendering
    DEBUG_FONT = cv2.FONT_HERSHEY_SIMPLEX
    DEBUG_FONT_SCALE = 0.6
    DEBUG_COLOR = (0, 255, 0)
