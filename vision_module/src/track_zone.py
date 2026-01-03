import cv2
import numpy as np


class TrackZone:
    """
    Railway track ROI + intrusion + tampering logic
    """

    def __init__(self, roi_polygon):
        self.roi = np.array(roi_polygon, dtype=np.int32)

    def point_inside(self, point):
        return cv2.pointPolygonTest(self.roi, point, False) >= 0

    def intrusion_from_box(self, box):
        """
        Uses bottom-center of bounding box
        """
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        return self.point_inside((cx, cy))

    def detect_tampering(self, intrusions):
        """
        Simple, explainable heuristic:
        - Multiple objects on track
        - Persistent obstruction
        """
        return len(intrusions) >= 2
