import cv2
import numpy as np


class TrackZone:
    def __init__(self, roi_polygon):
        self.roi = np.array(roi_polygon, dtype=np.int32)

    def intrusion_from_box(self, box):
        x1, y1, x2, y2 = box

        foot_points = [((x1 + x2) // 2, y2), (x1 + 10, y2), (x2 - 10, y2)]

        return any(cv2.pointPolygonTest(self.roi, p, False) >= 0 for p in foot_points)

    def detect_tampering(self, intrusions):
        return len(intrusions) >= 2
