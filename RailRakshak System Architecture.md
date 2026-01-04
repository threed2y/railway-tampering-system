🏛️ RailRakshak System Architecture
This architecture follows a Sensors $\rightarrow$ Intelligence $\rightarrow$ Actuators pipeline.
1. High-Level ASCII Diagram
This is perfect for text-based documentation or README.md.

Plaintext


┌─────────────────┐      ┌───────────────────────────────┐      ┌─────────────────┐
│  INPUT SENSORS  │      │       PROCESSING CORE         │      │  OUTPUT ACTION  │
│                 │      │        (The "Brain")          │      │                 │
├─────────────────┤      ├───────────────────────────────┤      ├─────────────────┤
│                 │      │  1. FRAME EXTRACTION          │      │                 │
│  📹 CCTV Cam A  │ ──►  │     (OpenCV VideoCapture)     │      │  🖥️ DASHBOARD   │
│                 │      │             ⬇                 │      │  (Streamlit UI) │
│  🚁 Drone Feed  │ ──►  │  2. PARALLEL AI INFERENCE     │ ──►  │                 │
│                 │      │     ├─ YOLOv8-Seg (Track)     │      │  🔊 AUDIO ALERT │
│  💾 Video File  │ ──►  │     └─ YOLOv8m (Threats)      │ ──►  │  (Base64 MP3)   │
│                 │      │             ⬇                 │      │                 │
└─────────────────┘      │  3. GEOMETRIC LOGIC LAYER     │      │  💾 BLACK BOX   │
                         │     (Shapely Intersection)    │ ──►  │  (Video Record) │
                         │     - Buffer Zone (+30px)     │      │                 │
                         │     - IoU Calculation         │      │                 │
                         └───────────────────────────────┘      └─────────────────┘


2. Technical Flow Breakdown
1. Input Layer (Data Acquisition)
Sources: The system accepts RTSP streams (CCTV), USB Cameras, or pre-recorded MP4 files (Simulated Drone footage).
Preprocessing: OpenCV (cv2) reads frames at 30 FPS and resizes them for model compatibility (640x640).
2. The Intelligence Layer (Dual-Model Engine)
Track Segmentation Model (track_model.pt):
Task: Uses YOLOv8-Seg to generate a binary mask of the railway track.
Post-Processing: Converts the mask into a Shapely Polygon and applies a Buffer of 30 pixels to account for objects standing near the edge (Danger Zone).
Object Detection Model (yolov8m.pt):
Task: Detects specific classes: 0 (Person), 19 (Cow), 20 (Elephant), 21 (Bear).
Output: Returns Bounding Boxes (x1, y1, x2, y2) for every object.
3. Logic Layer (The Decision Maker)
This is where the "Magic" happens. It performs a geometric intersection test:

$$Intersection = Area(Track Polygon \cap Object Box)$$
Adaptive Sensitivity:
If Object == Human: Alert triggers at 1% overlap (Immediate zero-tolerance for vandalism).
If Object == Animal: Alert triggers at 10% overlap (Reduces false positives for animals grazing nearby).
4. Output Layer (User Interface)
Visual: Bounding boxes turn RED (Critical) or YELLOW (Warning) on the live video.
Audio: A Base64 encoded MP3 is injected into the browser to play a siren instantly.
Evidence: The system initializes a cv2.VideoWriter to save the incident to the /recordings folder for legal verification.
3. Professional Diagram (Mermaid)
If your documentation platform supports Mermaid (like GitHub), use this code block:

Code snippet


graph TD
    subgraph INPUT ["📷 Input Sources"]
        CamA[CCTV Camera A]
        CamB[Drone Feed B]
        File[Test Video File]
    end

    subgraph AI ["🧠 Vision Engine"]
        Frame[Frame Extraction]
        YOLO_Seg[YOLOv8-Seg<br/>(Track Segmentation)]
        YOLO_Obj[YOLOv8m<br/>(Object Detection)]
    end

    subgraph LOGIC ["📐 Geometric Logic"]
        Poly[Polygon Creation + Buffer(30px)]
        IoU[Calculate Intersection]
        Rules{Threat Level?}
    end

    subgraph ACTION ["🚨 Response System"]
        UI[Streamlit Dashboard]
        Sound[Audio Alert]
        Rec[Record Evidence]
    end

    %% Connections
    CamA --> Frame
    CamB --> Frame
    File --> Frame
    
    Frame --> YOLO_Seg
    Frame --> YOLO_Obj
    
    YOLO_Seg --> Poly
    YOLO_Obj --> IoU
    Poly --> IoU
    
    IoU --> Rules
    
    Rules -- "> 1% (Human)" --> Critical[CRITICAL ALERT]
    Rules -- "> 10% (Animal)" --> Critical
    Rules -- "Near Track" --> Warning[WARNING]
    
    Critical --> UI
    Critical --> Sound
    Critical --> Rec
    
    Warning --> UI
    Warning --> Sound
    
    style Critical fill:#ff0000,stroke:#333,stroke-width:2px,color:#fff
    style AI fill:#f9f,stroke:#333,stroke-width:2px
    style LOGIC fill:#ccf,stroke:#333,stroke-width:2px


