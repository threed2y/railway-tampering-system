import cv2
import numpy as np


class WeatherEngine:
    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        contrast = gray.std()
        edges = cv2.Canny(gray, 50, 150).mean()

        if brightness < 55:
            return "NIGHT"
        if contrast < 35 and edges < 20:
            return "FOG"
        if edges > 80:
            return "RAIN"
        return "CLEAR"
