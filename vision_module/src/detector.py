import time
import csv
from collections import defaultdict


class VisionDetector:
    def __init__(self, model, config, log_path):
        self.model = model
        self.config = config
        self.log_path = log_path
        self.track_hits = defaultdict(int)
        self.last_logged = {}
        self.last_boxes = {}

        self._init_logs()

    def _init_logs(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "camera_id", "track_id", "event", "confidence", "state"]
            )

    def process_frame(self, frame):
        detections = []

        results = self.model.track(
            frame, conf=self.config["conf"], persist=True, verbose=False
        )

        if not results or results[0].boxes.id is None:
            return detections

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls not in self.config["target_classes"]:
                continue

            track_id = int(box.id[0])
            conf = float(box.conf[0])
            risk = self.config["risk_map"].get(cls, "OBSTRUCTION")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            self.last_boxes[track_id] = (x1, y1, x2, y2)
            self.track_hits[track_id] += 1

            if self.track_hits[track_id] >= self.config["alert_frames"]:
                detections.append((track_id, risk, conf))

        return detections

    def log_event(self, camera_id, track_id, event, conf, state):
        key = (track_id, state)
        now = time.time()

        if now - self.last_logged.get(key, 0) < self.config["cooldown"]:
            return

        self.last_logged[key] = now

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    time.strftime("%H:%M:%S"),
                    camera_id,
                    track_id,
                    event,
                    f"{conf:.2f}",
                    state,
                ]
            )
