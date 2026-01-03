import cv2
import argparse
import sys
import os
from ultralytics import YOLO

# Ensure src/ is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from detector import VisionDetector
from monitor import ThreatMonitor
from track_zone import TrackZone
from weather_engine import WeatherEngine
from image_filters import simulate_fog, enhance_frame
from visualizer import Visualizer  # <--- NEW IMPORT

CONFIG = {
    "conf": 0.35,  # elephant‑safe
    "alert_frames": 2,  # fast escalation
    "target_classes": [0, 16, 19, 20],  # PERSON, DOG, COW, ELEPHANT
}

# The "Danger Zone" Polygon
TRACK_ROI = [(180, 380), (460, 380), (380, 120), (260, 120)]


def resize_safe(frame, max_width=960):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def weather_confidence(weather):
    return {"CLEAR": 0.45, "FOG": 0.35, "RAIN": 0.35, "NIGHT": 0.30}.get(weather, 0.40)


def main(args):
    model = YOLO("models/yolov8n.pt")

    detector = VisionDetector(model, CONFIG)
    monitor = ThreatMonitor()
    track_zone = TrackZone(TRACK_ROI)
    weather_engine = WeatherEngine()
    visualizer = Visualizer(TRACK_ROI)  # <--- NEW: Initialize Visualizer

    cap = cv2.VideoCapture(args.source)

    print("🚆 RailRakshak | FINAL VERIFIED MONITORING STARTED")
    print("⌨️  Press 'Q' to Exit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_safe(frame)

        if args.simulate_fog:
            frame = simulate_fog(frame)

        # 1. Weather & Enhancement
        weather = weather_engine.analyze(frame)
        CONFIG["conf"] = weather_confidence(weather)
        enhanced_frame = enhance_frame(frame, weather)

        # 2. Process Detection
        detections = detector.process_frame(enhanced_frame)

        intrusions = []
        object_states = {}  # Store states to pass to Visualizer

        # 3. Logic Loop
        # NOTE: Ensure your detector.py returns (track_id, cls, conf)
        # If using old detector, this might error. Make sure detector.py is updated!
        for track_id, cls, conf in detections:
            # We need the box coordinates for the zone check
            box_data = detector.last_boxes.get(track_id)

            if box_data:
                # box_data is now (x1, y1, x2, y2, cls)
                coords = box_data[:4]

                if track_zone.intrusion_from_box(coords):
                    # Indian Context: Elephants (20) are INSTANT CRITICAL
                    if cls == 20:
                        state = "CRITICAL"
                    else:
                        state = monitor.update(track_id)

                    object_states[track_id] = state
                    intrusions.append(track_id)

        monitor.cleanup()

        # 4. Status Determination
        if track_zone.detect_tampering(intrusions):
            status = "🚨 TRACK TAMPERING"
        elif "CRITICAL" in object_states.values():
            status = "🚨 CRITICAL INTRUSION"
        elif "THREAT" in object_states.values():
            status = "⚠ INTRUSION MONITORING"
        elif intrusions:
            status = "👀 OBJECT NEAR TRACK"
        else:
            status = "🟢 TRACK CLEAR"

        print(f"SYSTEM STATUS: {status} | WEATHER: {weather}")

        # 5. VISUALIZATION (Draw the HUD)
        final_view = visualizer.draw_hud(
            enhanced_frame,
            status,
            weather,
            detections,
            detector.last_boxes,
            object_states,
        )

        cv2.imshow("RailRakshak Operator View", final_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Monitoring Stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--simulate_fog", action="store_true")
    args = parser.parse_args()

    main(args)
