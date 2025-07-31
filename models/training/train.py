import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model

if not os.path.exists("*.yaml"):
    raise FileNotFoundError("No YAML configuration files found in the current directory.")

results = model.train(
    data = os.getenv('MODEL_YAML_PATH'),
    epochs = 100,
    imgsz = 640,
    patience = 10,
    batch = -1,
    save = True,
    project = "berries",
    name = "yolov8n_berries",
    exist_ok = True,
    cache = False
)
