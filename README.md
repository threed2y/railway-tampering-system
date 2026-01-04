# 🚄 RailRakshak: AI-Powered Autonomous Railway Surveillance

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-yellow)

> **"Eyes on the Track, Safety on the Rack."**
> An intelligent, real-time Computer Vision system designed to prevent railway accidents caused by vandalism, unauthorized human access, and wildlife collisions.

---

## 🚨 The Problem
Railway safety in India is critical. Manual patrolling is inefficient against:
1.  **Vandalism:** Miscreants tampering with tracks or placing obstacles.
2.  **Wildlife Collisions:** Elephants and cattle wandering onto tracks, causing derailments and loss of life.
3.  **Human Encroachment:** Unauthorized walking on tracks in blind spots.

## 💡 The Solution: RailRakshak
RailRakshak is a **smart surveillance node** that can be deployed on CCTVs or Drones. It acts as a "Third Eye" for the pilot/station master.

### Key Capabilities:
* **🟢 Dynamic Track Segmentation:** Uses a custom trained **YOLOv8-Seg** model to understand exactly where the "Safe Zone" (Track) is.
* **🐘 Multi-Class Threat Detection:** Identifies Humans (Vandalism), Elephants, Bears, and Cattle.
* **📐 Geometric Danger Logic:** It doesn't just "see" objects; it calculates if they are **physically intersecting** with the track's danger zone (with a safety buffer).
* **🔊 Instant Alerts:** Triggers visual alarms and audio warnings (Siren/Voice) immediately.
* **📹 Black Box Recording:** Automatically starts recording video evidence the moment a threat is detected.

---

## 🎥 Demo
![RailRakshak Demo](preview/demo_preview.png)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/RailRakshak.git](https://github.com/YOUR_USERNAME/RailRakshak.git)
cd RailRakshak
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. 📥 Download External Assets (CRITICAL)
Due to GitHub's file size limits, the trained AI models and sample footage are hosted externally. 👉 CLICK HERE TO DOWNLOAD ASSETS (Google Drive)
4. Organize the Files
```bash
After downloading the assets, ensure your folder looks exactly like this structure:
Plaintext

railway-tampering-system/
│
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
├── track_model.pt            # 🧠 The Custom AI Model (Place here!)
│
└── Preview/                  # skip this
│
└── vision_module/
    ├── app.py                # 🚀 Main Application Code
    │
    ├── assets/
    │   ├── danger.mp3        # 🔊 Alarm Sound
    │   └── warning.mp3       # 🔊 Warning Sound
    │
    └── data/
        └── samples/
            ├── test.mp4      # 📹 Test Video 1
            └── Test2.mp4     # 📹 Test Video 2

    Note: The system uses a "Smart Hunter" algorithm, so as long as track_model.pt and the videos are somewhere in the project folder, the code will find them!
```
5. Run the System
```bash
streamlit run vision_module/app.py
```
🧠 How It Works (The Math)

    Segmentation: The system predicts a polygon mask for the railway track.

    Buffering: We apply a Buffer(+30px) to this polygon to create a "Danger Zone" that extends slightly beyond the rails.

    Intersection over Union (IoU):

        If a Human (Class 0) overlaps the Danger Zone by just 1%, it triggers a CRITICAL ALERT (High sensitivity for vandalism).

        If an Elephant (Class 20) overlaps by 10%, it triggers.

    Feedback Loop: If the status is "DANGER", the system locks the frame, writes it to the disk (/recordings), and plays the audio alert via Base64 injection.

🔮 Future Roadmap

    GPS Integration: To send the exact coordinates of the threat to the nearest station.

    Night Vision: Training the model on IR (Infrared) footage for 24/7 operation.

    Speed Estimation: Calculating the time-to-collision for approaching trains.

🏆 Team

    Developer: Ethan Hunt

    Role: AI & Machine learning

    Event:

    Team RAT

Built with sheer will.
