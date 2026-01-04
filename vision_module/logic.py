import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, box
from shapely.validation import make_valid


class RailRakshakSystem:
    def __init__(
        self, track_model_path="track_model.pt", object_model_path="yolov8n.pt"
    ):
        print("🚀 Loading RailRakshak Brains...")
        self.track_model = YOLO(track_model_path)
        self.object_model = YOLO(object_model_path)
        # 0=person, 15-19=animals
        self.target_classes = [0, 15, 16, 17, 18, 19]

    def process_frame(self, frame):
        alert_status = "SAFE"
        overlay = frame.copy()

        # --- PHASE 1: Track Detection ---
        track_results = self.track_model(frame, verbose=False)
        track_polygon = None

        if track_results[0].masks is not None:
            masks = track_results[0].masks.xy
            if len(masks) > 0:
                track_points = max(masks, key=len)

                # Geometry Fix
                raw_poly = Polygon(track_points)
                track_polygon = raw_poly.buffer(0)
                if not track_polygon.is_valid:
                    track_polygon = make_valid(track_polygon)

                # Draw Track (Green)
                pts = np.array(track_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 255, 0))

        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # --- PHASE 2: Object Detection ---
        obj_results = self.object_model(
            frame, classes=self.target_classes, verbose=False
        )

        # 🟢 CRITICAL FIX: Count the objects
        object_count = len(obj_results[0].boxes)

        for box_data in obj_results[0].boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            label = self.object_model.names[int(box_data.cls[0])]
            obj_poly = box(x1, y1, x2, y2)

            # --- PHASE 3: Intersection Logic ---
            danger_level = "SAFE"
            color = (0, 255, 0)

            if track_polygon is not None and not track_polygon.is_empty:
                try:
                    if track_polygon.intersects(obj_poly):
                        intersection_area = track_polygon.intersection(obj_poly).area
                        obj_area = obj_poly.area
                        if obj_area > 0:
                            overlap_ratio = intersection_area / obj_area

                            if overlap_ratio > 0.15:
                                danger_level = "CRITICAL"
                                color = (0, 0, 255)  # Red
                                alert_status = f"DANGER: {label.upper()} ON TRACK!"
                            elif overlap_ratio > 0.01:
                                danger_level = "WARNING"
                                color = (0, 255, 255)  # Yellow
                                if "DANGER" not in alert_status:
                                    alert_status = "WARNING: Object Near Track"
                except Exception:
                    pass  # Skip math errors

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # 🟢 CRITICAL FIX: Return 3 values
        return frame, alert_status, object_count
