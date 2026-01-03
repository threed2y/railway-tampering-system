import cv2
import argparse
from ultralytics import YOLO

from src.detector import VisionDetector
from src.monitor import ThreatMonitor
from src.track_zone import TrackZone
from src.image_filters import apply_fog_filter

CONFIG = {
    "conf": 0.5,
    "alert_frames": 5,
    "cooldown": 5,
    "target_classes": [0, 3, 16, 19, 20],
    "risk_map": {
        0: "TRESPASSER",
        3: "VEHICLE",
        16: "ANIMAL",
        19: "CATTLE",
        20: "WILDLIFE",
    },
}

# 🔺 Fixed demo-friendly track ROI (tunable)
TRACK_ROI = [(200, 380), (440, 380), (360, 120), (280, 120)]


def main(args):
    model = YOLO("models/yolov8n.pt")

    detector = VisionDetector(
        model=model, config=CONFIG, log_path="logs/vision_alerts.csv"
    )

    monitor = ThreatMonitor()
    track_zone = TrackZone(TRACK_ROI)

    cap = cv2.VideoCapture(args.source)
    frame_id = 0

    print("🚆 RailRakshak | Track Monitoring Started")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % args.skip != 0:
            continue

        frame = cv2.resize(frame, (640, 384))
        frame = apply_fog_filter(frame, args.fog)

        detections = detector.process_frame(frame)

        intrusions = []
        states = []

        for track_id, risk, conf in detections:
            box = detector.last_boxes.get(track_id)
            if not box:
                continue

            if track_zone.intrusion_from_box(box):
                state = monitor.update(track_id)
                states.append(state)
                intrusions.append(track_id)

                detector.log_event(
                    camera_id="CAM_01",
                    track_id=track_id,
                    event="TRACK_INTRUSION",
                    conf=conf,
                    state=state,
                )

        monitor.cleanup()

        tampering = track_zone.detect_tampering(intrusions)

        if tampering:
            status = "🚨 TRACK TAMPERING"
        elif "CRITICAL" in states:
            status = "🚨 CRITICAL INTRUSION"
        elif "THREAT" in states:
            status = "⚠ INTRUSION MONITORING"
        else:
            status = "🟢 TRACK CLEAR"

        print(f"SYSTEM STATUS: {status}")

        if not args.headless:
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if "🚨" in status else (0, 255, 0),
                2,
            )

            cv2.polylines(frame, [track_zone.roi], True, (255, 0, 255), 2)
            cv2.imshow("RailRakshak | Active Track Monitoring", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Monitoring Stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--fog", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()

    main(args)
