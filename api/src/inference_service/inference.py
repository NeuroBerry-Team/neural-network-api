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
            root_dir = Path(__file__).resolve().parent.parent.parent.parent
            model_path = root_dir / "weights" / "best.pt" # Should be /app/weights/{model_name}.pt
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
        
        metadata = {
            'detection_count': 0,
            'detections': [],
            'image_info': {
                'original_size': None,
                'processed_size': None
            }
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                metadata['detection_count'] = len(boxes)
                metadata['image_info']['original_size'] = boxes.orig_shape  # (height, width)
                
                # Extract detection details
                for i in range(len(boxes)):
                    detection = {
                        'bbox': {
                            'x1': float(boxes.xyxy[i][0]),
                            'y1': float(boxes.xyxy[i][1]), 
                            'x2': float(boxes.xyxy[i][2]),
                            'y2': float(boxes.xyxy[i][3])
                        },
                        'bbox_normalized': {
                            'x1': float(boxes.xyxyn[i][0]),
                            'y1': float(boxes.xyxyn[i][1]),
                            'x2': float(boxes.xyxyn[i][2]),
                            'y2': float(boxes.xyxyn[i][3])
                        },
                        'center_point': {
                            'x': float(boxes.xywh[i][0]),
                            'y': float(boxes.xywh[i][1]),
                            'width': float(boxes.xywh[i][2]),
                            'height': float(boxes.xywh[i][3])
                        },
                        'confidence': float(boxes.conf[i]),
                        'class_id': int(boxes.cls[i]),
                        'area': float(boxes.xywh[i][2] * boxes.xywh[i][3])
                    }
                    metadata['detections'].append(detection)
            
            result.save(filename=output_path)
        return metadata
