from ultralytics import YOLO
import torch
import os


def train():
    # 1. Setup Device (Auto-detect GPU or CPU)
    # Checks if CUDA is available for NVIDIA GPUs
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training RailRakshak on: {device.upper()}")

    # 2. Load the Base Model (Nano Segmentation)
    # 'yolov8n-seg.pt' is the fastest model.
    model = YOLO("yolov8n-seg.pt")

    # 3. Define the Config Path
    # This gets the absolute path to your data.yaml file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, "data", "railway_seg", "data.yaml")

    print(f"📄 Loading config from: {yaml_path}")

    # 4. Start Training
    model.train(
        data=yaml_path,
        epochs=30,  # 30 epochs is a good balance for a demo
        imgsz=640,  # Standard image resolution
        batch=16,  # Batch size (reduce to 8 if you get "Out of Memory" errors)
        name="rail_rakshak_seg",  # The output folder name
        device=device,
        patience=5,  # Stop early if accuracy stops improving
        save=True,  # Save the best model weights
        workers=4,  # CPU threads for data loading
    )

    # 5. Validation
    print("📊 Validating Model...")
    metrics = model.val()

    # 6. Final Output
    print(
        f"✅ DONE! Best weights saved at: runs/segment/rail_rakshak_seg/weights/best.pt"
    )


if __name__ == "__main__":
    train()
