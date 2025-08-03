import os
from pathlib import Path

class InferenceService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print("Current path:", Path.cwd(), flush=True)
            print("Files in current directory:", os.listdir(Path.cwd()), flush=True) # TODO: Need to mount the model in the container
            model_path = Path("/app/api/best.pt") # TODO: Update so the path is dynamic
            self._model = YOLO(str(model_path))
            print(f"Model loaded successfully from {model_path}", flush=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def warmup(self):
        """Warm up the model with a dummy prediction"""
        try:
            # Create a small dummy image or use existing test image
            import torch
            dummy_tensor = torch.zeros((640, 640, 3))  # Dummy image
            temp_path = "/tmp/dummy_warmup.jpg"
            
            # Save dummy image temporarily
            from PIL import Image
            import numpy as np
            dummy_img = Image.fromarray(np.uint8(dummy_tensor.numpy()))
            dummy_img.save(temp_path)
            
            # Run warmup prediction
            self._model.predict(temp_path, verbose=False)
            
            # Cleanup
            os.remove(temp_path)
            print("Model warmup completed", flush=True)
        except Exception as e:
            print(f"Model warmup failed (non-critical): {e}", flush=True)

    def predict(self, image_path, output_path):
        print(f"Running prediction on {image_path}", flush=True)
        print(f"Receiving from image path: {image_path}", flush=True)
        print(f"Saving results to {output_path}", flush=True)
        results = self._model(image_path)
        for result in results:
            result.save(filename=output_path)
