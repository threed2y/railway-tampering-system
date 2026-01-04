import streamlit as st
import cv2
import time
import os
import base64
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="RailRakshak AI",
    page_icon="🚄",
    layout="wide",
)

# ---------------- 2. CSS STYLING ----------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; }
    .status-pill {
        padding: 15px; border-radius: 8px; text-align: center;
        font-size: 24px; font-weight: bold; margin-bottom: 20px;
        letter-spacing: 2px; border: 2px solid rgba(255,255,255,0.1);
    }
    .safe { background-color: #00C853; color: white; box-shadow: 0 0 15px #00C853; }
    .warning { background-color: #FFD600; color: black; box-shadow: 0 0 15px #FFD600; }
    .danger { 
        background-color: #D50000; color: white; 
        box-shadow: 0 0 20px #D50000; animation: pulse 0.8s infinite; 
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(0.98); }
        100% { opacity: 1; transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- 3. SESSION STATE ----------------
if "alert_history" not in st.session_state: st.session_state["alert_history"] = []
if "last_alarm" not in st.session_state: st.session_state["last_alarm"] = None
if "recording" not in st.session_state: st.session_state["recording"] = {}
if "writers" not in st.session_state: st.session_state["writers"] = {}

# ---------------- 4. THE BRAIN (LOGIC) ----------------
class RailRakshakSystem:
    def __init__(self, track_model_path, object_model_path):
        self.track_model = YOLO(track_model_path)
        self.object_model = YOLO(object_model_path)
        # 0=Person, 15-21=Animals (Elephant, Bear, Cow, etc.)
        self.target_classes = [0, 15, 16, 17, 18, 19, 20, 21]

    def process_frame(self, frame):
        alert_status = "SAFE"
        overlay = frame.copy()
        
        # --- 1. Track Segmentation ---
        track_results = self.track_model(frame, verbose=False)
        track_polygon = None
        
        if track_results[0].masks is not None:
            masks = track_results[0].masks.xy
            if len(masks) > 0:
                track_points = max(masks, key=len)
                raw_poly = Polygon(track_points)
                # Buffer 30 expands the track zone
                track_polygon = raw_poly.buffer(30) 
                if not track_polygon.is_valid:
                    track_polygon = make_valid(track_polygon)

                pts = np.array(track_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

        # --- 2. Object Detection ---
        obj_results = self.object_model(frame, classes=self.target_classes, verbose=False)
        object_count = len(obj_results[0].boxes)
        
        for box_data in obj_results[0].boxes:
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            cls_id = int(box_data.cls[0])
            label = self.object_model.names[cls_id]
            obj_poly = box(x1, y1, x2, y2)
            
            # --- 3. Intersection Logic ---
            danger_level = "SAFE"
            color = (0, 255, 0) 
            
            if track_polygon is not None and not track_polygon.is_empty:
                try:
                    if track_polygon.intersects(obj_poly):
                        intersection_area = track_polygon.intersection(obj_poly).area
                        obj_area = obj_poly.area
                        
                        if obj_area > 0:
                            overlap_ratio = intersection_area / obj_area
                            
                            # Sensitivity: 1% for Humans, 10% for Animals
                            threshold = 0.10 
                            if cls_id == 0: threshold = 0.01 
                            
                            if overlap_ratio > threshold:
                                danger_level = "CRITICAL"
                                color = (0, 0, 255)
                                alert_status = f"DANGER: {label.upper()} ON TRACK!"
                            elif overlap_ratio > 0.005: 
                                danger_level = "WARNING"
                                color = (0, 255, 255)
                                if "DANGER" not in alert_status:
                                    alert_status = "WARNING: Object Near Track"
                except: pass 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, alert_status, object_count

# ---------------- 5. HELPER FUNCTIONS (THE HUNTER) ----------------
@st.cache_resource
def find_file_recursive(filename):
    """
    Searches for a file recursively starting from the current directory.
    This ensures we find the file even if folders are messy.
    """
    # 1. Fast Check (Common Locations)
    common_paths = [
        filename,
        os.path.join("assets", filename),
        os.path.join("vision_module", "assets", filename),
        os.path.join("data", "assets", filename),
        os.path.join("..", "assets", filename)
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p

    # 2. Deep Search (Walk through all folders)
    # We walk from the current working directory "."
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
            
    return None

@st.cache_resource
def load_system():
    # Use the hunter to find the model
    track_path = find_file_recursive("track_model.pt")
    
    if not track_path:
        st.error("❌ CRITICAL: 'track_model.pt' not found anywhere in the project folder.")
        st.stop()
        
    return RailRakshakSystem(track_path, "yolov8m.pt")

def play_alarm_sound(level):
    if st.session_state["last_alarm"] == level: return 
    st.session_state["last_alarm"] = level
    
    filename = "danger.mp3" if level == "DANGER" else "warning.mp3"
    
    # Use the hunter to find the audio
    audio_path = find_file_recursive(filename)
    
    if audio_path:
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            audio_html = f"""
                <audio autoplay="true" style="display:none;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.toast(f"Audio Error: {e}")
    else:
        st.toast(f"Audio file missing: {filename}")

# ---------------- 6. MAIN APP ----------------
def main():
    with st.sidebar:
        st.title("🚄 Control Panel")
        conf_threshold = st.slider("Sensitivity", 0.1, 1.0, 0.30) 
        if st.button("🗑️ Clear Logs"): st.session_state["alert_history"] = []
        st.markdown("---")
        run_system = st.toggle("✅ ENABLE SURVEILLANCE", value=False)

    st.title("🚄 RailRakshak Command Center")
    status_placeholder = st.empty()
    status_placeholder.markdown('<div class="status-pill safe">SYSTEM ONLINE</div>', unsafe_allow_html=True)

    # 🎥 FILES
    TARGET_FILENAMES = { "Track Cam A": "test.mp4", "Track Cam B": "Test2.mp4" }
    FINAL_SOURCES = {}
    webcam_counter = 0

    for name, filename in TARGET_FILENAMES.items():
        # Use recursive hunter for videos too!
        found_path = find_file_recursive(filename)
        if found_path:
            FINAL_SOURCES[name] = found_path
        else:
            FINAL_SOURCES[f"{name} (Live)"] = webcam_counter
            st.toast(f"⚠️ {filename} missing. Using Webcam {webcam_counter}", icon="📷")
            webcam_counter += 1

    cols = st.columns(len(FINAL_SOURCES))
    video_slots = {}
    for col, name in zip(cols, FINAL_SOURCES):
        with col:
            st.subheader(f"📹 {name}")
            video_slots[name] = st.empty()

    st.markdown("---")
    st.subheader("📝 Incident Log")
    log_placeholder = st.empty()

    system = load_system()
    system.object_model.conf = conf_threshold
    
    caps = {}
    valid_feeds = []
    for name, src in FINAL_SOURCES.items():
        cap = cv2.VideoCapture(src)
        if cap.isOpened():
            caps[name] = cap
            valid_feeds.append(name)

    # ---------------- LOOP ----------------
    if run_system:
        os.makedirs("recordings", exist_ok=True)
        while run_system:
            global_status = "SAFE" 
            active_feeds = False

            for cam_name in valid_feeds:
                cap = caps[cam_name]
                ret, frame = cap.read()
                
                if not ret and isinstance(FINAL_SOURCES[cam_name], str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                if ret:
                    active_feeds = True
                    processed_frame, status, count = system.process_frame(frame)
                    video_slots[cam_name].image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), width="stretch")
                    
                    if "DANGER" in status:
                        global_status = "DANGER"
                        if cam_name not in st.session_state["recording"]:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            path = f"recordings/{cam_name.replace(' ','_')}_{ts}.mp4"
                            h, w, _ = processed_frame.shape
                            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
                            st.session_state["recording"][cam_name] = path
                            st.session_state["writers"][cam_name] = writer
                        st.session_state["writers"][cam_name].write(processed_frame)
                        log_msg = f"{time.strftime('%H:%M:%S')} | {cam_name} | {status}"
                        if not st.session_state["alert_history"] or st.session_state["alert_history"][0] != log_msg:
                            st.session_state["alert_history"].insert(0, log_msg)
                    else:
                        if cam_name in st.session_state["recording"]:
                            st.session_state["writers"][cam_name].release()
                            del st.session_state["writers"][cam_name]
                            del st.session_state["recording"][cam_name]

            if global_status == "DANGER":
                status_placeholder.markdown(f'<div class="status-pill danger">🚨 CRITICAL THREAT DETECTED</div>', unsafe_allow_html=True)
                play_alarm_sound("DANGER")
            elif global_status == "WARNING":
                status_placeholder.markdown(f'<div class="status-pill warning">⚠️ CAUTION: OBJECT NEAR TRACK</div>', unsafe_allow_html=True)
                play_alarm_sound("WARNING")
            else:
                status_placeholder.markdown('<div class="status-pill safe">✅ SYSTEM ONLINE - TRACK CLEAR</div>', unsafe_allow_html=True)
                st.session_state["last_alarm"] = None

            with log_placeholder.container(height=300):
                for log in st.session_state["alert_history"][:20]:
                    color = "red" if "DANGER" in log else "orange" if "WARNING" in log else "green"
                    st.markdown(f":{color}[{log}]")

            if not active_feeds: break
            time.sleep(0.01)

    if not run_system:
        for cap in caps.values(): cap.release()
        for w in st.session_state["writers"].values(): w.release()

if __name__ == "__main__":
    main()