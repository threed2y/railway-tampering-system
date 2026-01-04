# 🚄 RailRakshak: Autonomous Railway Surveillance System

> **AI-Powered Threat Detection for Railway Safety | Hackathon 2026**

## 🚨 The Problem
Railway safety is compromised by unauthorized track access, vandalism, and wildlife collisions. Traditional manual monitoring is slow and error-prone.

## 💡 The Solution
**RailRakshak** is a real-time Computer Vision system that uses **Drone/CCTV footage** to:
1.  **Segment Tracks:** Automatically detects the railway track (Green Zone) using a custom YOLOv8-Seg model.
2.  **Detect Threats:** Identifies Humans, Elephants, and Cattle.
3.  **Calculate Danger:** Uses geometric logic to trigger alerts ONLY if the object is physically ON or dangerously CLOSE to the tracks.
4.  **Auto-Record:** Automatically saves video clips of incidents for legal evidence.

---

## 🛠️ Tech Stack
* **AI Models:** YOLOv8-Seg (Custom Trained), YOLOv8m (Object Detection)
* **Logic:** Shapely (Geometric Intersection)
* **Interface:** Streamlit (Real-time Dashboard)
* **Language:** Python 3.10+

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/RailRakshak.git](https://github.com/YOUR_USERNAME/RailRakshak.git)
cd RailRakshak
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Add Missing Assets (Important!)

Due to file size limits, some assets are not on GitHub.

    Step A: Ensure track_model.pt is in the vision_module/ folder.

    Step B: Place your test videos in vision_module/data/samples/.

4. Launch the Dashboard
```bash
streamlit run vision_module/app.py
```
📸 Features

    ✅ Multi-Cam Support: Toggle between Track Cam A and B.

    ✅ Smart Sensitivity: High sensitivity for Humans (Vandalism), optimized for Large animals.

    ✅ Silent Alarm: Visual and Audio alerts for the operator.

    ✅ Evidence Locker: All threats are recorded to /recordings.

Team RAT

Built with ❤️
