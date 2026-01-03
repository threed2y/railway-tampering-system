from collections import defaultdict


class VisionDetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # Store box AND class_id: {track_id: (x1, y1, x2, y2, class_id)}
        self.last_boxes = {}
        self.track_hits = defaultdict(int)

    def process_frame(self, frame):
        detections = []

        results = self.model.track(
            frame, conf=self.config["conf"], persist=True, verbose=False
        )

        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in self.config["target_classes"]:
                continue

            track_id = int(box.id[0]) if box.id is not None else -1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # UPDATED: We now store 'cls' (Class ID) at index 4
            self.last_boxes[track_id] = (x1, y1, x2, y2, cls)
            self.track_hits[track_id] += 1

            if self.track_hits[track_id] >= self.config["alert_frames"]:
                # UPDATED: We now return 'cls' so the visualizer knows the color
                detections.append((track_id, cls, conf))

        return detections
