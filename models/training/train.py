import os
from pathlib import Path
from ultralytics import YOLO


def validate_yaml_path():
    """Validate the YAML configuration path"""
    yaml_path = os.getenv('MODEL_YAML_PATH')
    if not yaml_path:
        raise ValueError("MODEL_YAML_PATH environment variable not set")
    
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    if yaml_file.suffix.lower() != '.yaml':
        raise ValueError("File must have .yaml extension")
    
    return str(yaml_file.resolve())


try:
    model = YOLO("yolov8n.pt")  # Load a pretrained YOLOv8n model
    
    # Validate YAML configuration
    yaml_path = validate_yaml_path()
    
    # Create weights directory if it doesn't exist
    weights_dir = Path("../weights")
    weights_dir.mkdir(exist_ok=True)
    
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        patience=10,
        batch=-1,
        save=True,
        project=str(weights_dir),
        name="yolov8n_berries",
        exist_ok=True,
        cache=False
    )
    
except Exception as e:
    print(f"Training failed: {str(e)}")
    raise