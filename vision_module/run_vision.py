import cv2
import argparse
from ultralytics import YOLO

from src.detector import VisionDetector
from src.image_filters import apply_fog_filter

CONFIG = {
    "conf": 0.5,
    "alert_frames": 5,
    "cooldown": 5,
    "target_classes": [0, 3, 16, 19, 20],
    "risk_map": {
        0: "TRESPASSER",
        3: "VEHICLE_ON_TRACK",
        16: "ANIMAL_HAZARD",
        19: "CATTLE_COLLISION",
        20: "CRITICAL_WILDLIFE"
    }
}

def main(args):
    model = YOLO("models/yolov8n.pt")
    detector = VisionDetector(
        model=model,
        config=CONFIG,
        log_path="logs/vision_alerts.csv"
    )

    cap = cv2.VideoCapture(args.source)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % args.skip != 0:
            continue

        frame = cv2.resize(frame, (640, 384))
        frame = apply_fog_filter(frame, args.fog)

        frame, alerts = detector.process_frame(frame)

        for _, risk, _ in alerts:
            print(f"⚠ ALERT: {risk}")

        if not args.headless:
            cv2.imshow("Railway Vision Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0)
    parser.add_argument("--fog", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()

    main(args)
