import time
import csv
from collections import defaultdict

class VisionDetector:
    def __init__(self, model, config, log_path):
        self.model = model
        self.config = config
        self.log_path = log_path
        self.track_memory = defaultdict(int)
        self.last_alert_time = {}

        self._init_logs()

    def _init_logs(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "camera_id",
                "track_id",
                "risk_type",
                "confidence"
            ])

    def process_frame(self, frame, camera_id="CAM_01"):
        alerts = []

        results = self.model.track(
            frame,
            conf=self.config["conf"],
            persist=True,
            verbose=False
        )

        if not results or results[0].boxes.id is None:
            return frame, alerts

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls not in self.config["target_classes"]:
                continue

            track_id = int(box.id[0])
            conf = float(box.conf[0])
            self.track_memory[track_id] += 1

            if self.track_memory[track_id] < self.config["alert_frames"]:
                continue

            risk = self.config["risk_map"].get(cls, "OBSTRUCTION")
            now = time.time()

            if now - self.last_alert_time.get(track_id, 0) < self.config["cooldown"]:
                continue

            self.last_alert_time[track_id] = now
            alerts.append((track_id, risk, conf))

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    time.strftime("%H:%M:%S"),
                    camera_id,
                    track_id,
                    risk,
                    f"{conf:.2f}"
                ])

        return frame, alerts
