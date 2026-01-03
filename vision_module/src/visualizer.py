import cv2
import numpy as np


class Visualizer:
    def __init__(self, roi_polygon):
        self.roi = np.array(roi_polygon, dtype=np.int32)

        # Indian Context Color Coding
        self.COLORS = {
            0: (0, 0, 255),  # Person: Red
            19: (0, 140, 255),  # Cow: Orange
            20: (0, 0, 139),  # Elephant: Dark Red
            16: (0, 255, 255),  # Dog: Yellow
        }
        self.LABELS = {0: "HUMAN", 19: "CATTLE", 20: "ELEPHANT", 16: "ANIMAL"}

    def draw_hud(self, frame, status, weather, detections, last_boxes, object_states):
        overlay = frame.copy()

        # 1. Draw Track Zone (Green = Safe, Red = Danger)
        zone_color = (
            (0, 255, 0)
            if "CRITICAL" not in status and "TAMPERING" not in status
            else (0, 0, 255)
        )
        cv2.polylines(frame, [self.roi], True, zone_color, 2)

        # Fill zone slightly transparent for high-tech look
        cv2.fillPoly(overlay, [self.roi], zone_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # 2. Draw Bounding Boxes
        for track_id, cls, conf in detections:
            if track_id in last_boxes:
                # Retrieve coordinates safely
                box_data = last_boxes[track_id]
                x1, y1, x2, y2 = box_data[:4]

                state = object_states.get(track_id, "OBSERVED")

                # Dynamic Color based on Class & State
                color = self.COLORS.get(cls, (255, 255, 255))
                if state == "CRITICAL":
                    color = (0, 0, 255)  # Force Red for Critical

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw Label Background
                label = f"{self.LABELS.get(cls, 'OBJ')} | {state}"
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + 200, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        # 3. Status Banner (Top)
        self._draw_status_bar(frame, status, weather)

        # 4. Critical Flash (Red Border)
        if "CRITICAL" in status or "TAMPERING" in status:
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10
            )

        return frame

    def _draw_status_bar(self, frame, status, weather):
        # Black background bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)

        # Status Text Color
        color = (0, 255, 0)  # Green default
        if "CRITICAL" in status:
            color = (0, 0, 255)
        elif "INTRUSION" in status:
            color = (0, 165, 255)
        elif "TAMPERING" in status:
            color = (255, 0, 0)  # Blue for sabotage

        cv2.putText(
            frame,
            f"STATUS: {status}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        # Weather Text
        cv2.putText(
            frame,
            f"WX MODE: {weather}",
            (frame.shape[1] - 280, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
