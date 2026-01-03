import cv2
import numpy as np


def simulate_fog(frame, intensity=0.6):
    fog_layer = np.full(frame.shape, 255, dtype=np.uint8)
    return cv2.addWeighted(frame, 1 - intensity, fog_layer, intensity, 0)


def dehaze(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def brighten(frame):
    return cv2.convertScaleAbs(frame, alpha=1.3, beta=30)


def reduce_noise(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)


def enhance_frame(frame, weather):
    if weather == "FOG":
        return dehaze(frame)
    if weather == "NIGHT":
        return brighten(frame)
    if weather == "RAIN":
        return reduce_noise(frame)
    return frame
