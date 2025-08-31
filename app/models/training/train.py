import os
import sys
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


def get_model_type():
    """Get model type from environment variable"""
    model_type = os.getenv('MODEL_TYPE', 'yolov8n')
    # Convert from format like YOLOv8_m to yolov8m
    model_name = model_type.lower().replace('_', '')
    if not model_name.endswith('.pt'):
        model_name += '.pt'
    return model_name


def load_model(model_filename):
    """Load YOLO model, trying local first, then download"""
    # Try local base_models first
    local_model_path = Path("/models/base_models") / model_filename
    
    if local_model_path.exists():
        print(f"Using local base model: {local_model_path}", flush=True)
        try:
            model = YOLO(str(local_model_path))
            print(f"Successfully loaded local model: {local_model_path}",
                  flush=True)
            return model
        except Exception as local_error:
            print(f"Failed to load local model: {local_error}", flush=True)
            print("Falling back to download...", flush=True)
    else:
        print(f"Local model not found: {local_model_path}", flush=True)
        print("Attempting to download model...", flush=True)
    
    # If local model failed or doesn't exist, try downloading
    try:
        print(f"Downloading model: {model_filename}", flush=True)
        model = YOLO(model_filename)  # This will trigger download
        print(f"Successfully downloaded model: {model_filename}",
              flush=True)
        return model
    except Exception as download_error:
        raise Exception(f"Could not load or download model {model_filename}: "
                        f"{download_error}")


def get_training_params():
    """Get training parameters from environment variables"""
    return {
        'epochs': int(os.getenv('EPOCHS', '100')),
        'imgsz': int(os.getenv('IMAGE_SIZE', '640')),
        'batch': int(os.getenv('BATCH_SIZE', '-1')),  # -1 for auto batch size
        'patience': int(os.getenv('PATIENCE', '10')),
        'lr0': float(os.getenv('LEARNING_RATE', '0.01')),
        'save': os.getenv('SAVE', 'True').lower() == 'true',
        'cache': os.getenv('CACHE', 'False').lower() == 'true',
        'project': os.getenv('PROJECT_DIR', '../weights'),
        'name': os.getenv('MODEL_NAME', 'yolo_training'),
        'exist_ok': True
    }


try:
    # Get configuration
    model_filename = get_model_type()
    yaml_path = validate_yaml_path()
    params = get_training_params()
    
    print(f"Starting training with model: {model_filename}", flush=True)
    print(f"Using dataset config: {yaml_path}", flush=True)
    print(f"Training parameters: {params}", flush=True)
    
    # Load model (try local first, then download)
    model = load_model(model_filename)
    
    # Create weights directory if it doesn't exist
    weights_dir = Path(params['project'])
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    results = model.train(
        data=yaml_path,
        epochs=params['epochs'],
        imgsz=params['imgsz'],
        batch=params['batch'],
        lr0=params['lr0'],
        patience=params['patience'],
        save=params['save'],
        project=str(weights_dir),
        name=params['name'],
        exist_ok=params['exist_ok'],
        cache=params['cache']
    )
    
    print("Training completed successfully!", flush=True)
    print(f"Results: {results}", flush=True)
    
except Exception as e:
    print(f"Training failed: {str(e)}", flush=True)
    sys.exit(1)