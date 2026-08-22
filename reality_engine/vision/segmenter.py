import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from reality_engine.core.state import AppState
from reality_engine.vision.model_utils import ensure_model

class HumanSegmenter:
    def __init__(self):
        model_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
        model_path = ensure_model("selfie_segmenter.tflite", model_url)
        
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=False,
            output_confidence_masks=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        
    def process(self, state: AppState):
        if state.frame is None:
            return
            
        frame_rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.segmenter.segment(mp_image)
        
        if results.confidence_masks and len(results.confidence_masks) > 0:
            # Person is usually category 1, background is 0.
            # But the documentation for selfie segmenter indicates confidence masks are generated for each category,
            # but wait, let's just grab mask 1 if it exists (mask 0 is background, mask 1 is person).
            
            # The model output two categories, background at index 0 and person at index 1.
            if len(results.confidence_masks) > 1:
                person_mask = results.confidence_masks[1].numpy_view()
            else:
                # Fallback if only one mask returned
                person_mask = results.confidence_masks[0].numpy_view()
                
            # The mask is float32 [0.0, 1.0]
            # Smooth the mask for softer edges
            mask_uint8 = (person_mask * 255).astype(np.uint8)
            mask_smoothed = cv2.bilateralFilter(mask_uint8, 9, 75, 75)
            
            state.human_mask = mask_smoothed.astype(np.float32) / 255.0
        else:
            state.human_mask = None
