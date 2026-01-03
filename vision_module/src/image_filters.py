import cv2
import numpy as np

def apply_fog_filter(frame, enabled=False):
    if not enabled:
        return frame

    fog = np.full(frame.shape, 200, dtype=np.uint8)
    return cv2.addWeighted(frame, 0.7, fog, 0.3, 0)
