import os
from pathlib import Path


class InferenceService:
    _instance = None
    _model = None
    _current_model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self, model_name="default"):
        try:
            from ultralytics import YOLO

            # Determine model path based on model name
            if model_name == "default":
                model_path = Path("/models/best.pt")
                if not model_path.exists():
                    model_path = Path("/models/weights/best.pt")
            else:
                model_path = Path(f"/models/weights/{model_name}.pt")
                
            if not model_path.exists():
                raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")

            self._model = YOLO(str(model_path))
            self._current_model_name = model_name
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    def _ensure_model_loaded(self, model_name="default"):
        """Ensure the correct model is loaded, reload if necessary"""
        if self._current_model_name != model_name or self._model is None:
            self._load_model(model_name)

    def warmup(self):
        """Warm up the model with a dummy prediction"""
        try:
            # Ensure default model is loaded for warmup
            self._ensure_model_loaded("default")
            
            import torch
            from PIL import Image
            import numpy as np

            dummy_tensor = torch.zeros((640, 640, 3))
            temp_path = "/tmp/dummy_warmup.jpg"

            dummy_img = Image.fromarray(np.uint8(dummy_tensor.numpy()))
            dummy_img.save(temp_path)

            self._model.predict(temp_path, verbose=False)
            os.remove(temp_path)
        except Exception:
            pass

    def predict(self, image_path, output_path, model_name="default"):
        # Load the appropriate model if different from current
        self._ensure_model_loaded(model_name)
        
        results = self._model.predict(
            source=image_path,
            iou=0.25,  # softer NMS
            agnostic_nms=True,
            augment=True,
        )

        metadata = {
            "detection_count": 0,
            "detections": [],
            "image_info": {"original_size": None, "processed_size": None},
        }

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                metadata["image_info"]["original_size"] = boxes.orig_shape
                metadata["image_info"]["processed_size"] = result.orig_shape

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])

                    # Get both pixel and normalized coordinates from YOLO
                    pixel_coords = boxes.xyxy[i]  # Pixel coordinates
                    normalized_coords = boxes.xyxyn[i]  # Normalized coordinates (0-1)

                    detection = {
                        "bbox": {
                            "x1": float(pixel_coords[0]),
                            "y1": float(pixel_coords[1]),
                            "x2": float(pixel_coords[2]),
                            "y2": float(pixel_coords[3]),
                        },
                        "bbox_normalized": {
                            "x1": float(normalized_coords[0]),
                            "y1": float(normalized_coords[1]),
                            "x2": float(normalized_coords[2]),
                            "y2": float(normalized_coords[3]),
                        },
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": self._model.names[class_id],
                    }
                    metadata["detections"].append(detection)

                metadata["detection_count"] = len(metadata["detections"])

            self._draw_custom_visualization(
                image_path, output_path, metadata["detections"]
            )

        return metadata

    def _draw_custom_visualization(
        self, input_image_path, output_image_path, detections
    ):
        """Draw bounding boxes and labels using the same data as metadata"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(input_image_path)
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size

            font_size = max(12, int(img_height * 0.025))  # Scale by 2.5% image height

            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
            ]  # Could add more if needed

            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"Successfully loaded font: {font_path}", flush=True)
                    break
                except (OSError, IOError) as e:
                    print(f"Failed to load font {font_path}: {e}", flush=True)
                    continue

            class_colors = {
                2: (153, 27, 27),  # C5 DarkRed #991B1B
                1: (239, 68, 68),  # C4 BrightRed #EF4444
                4: (245, 158, 11),  # C3 Orange #F59E0B
                3: (34, 197, 94),  # C2 Green #22C55E
                0: (107, 114, 128),  # C1 Boton (gray) #6B7280
            }

            class_labels = {
                2: "C5 DarkRed",
                1: "C4 BrightRed",
                4: "C3 Orange (Red dot)",
                3: "C2 Green",
                0: "C1 Boton",
            }

            for detection in detections:
                bbox = detection["bbox"]
                confidence = detection["confidence"]
                class_id = detection["class_id"]

                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

                color = class_colors.get(class_id, (99, 102, 241))
                class_name = class_labels.get(class_id, f"{class_id}")
                label_text = f"{class_name} {int(confidence * 100)}%"

                line_width = max(2, int(font_size * 0.15))  # Magic number 0.15
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]  # X2 - X1
                text_height = text_bbox[3] - text_bbox[1]  # Y2 - Y1

                label_padding = max(2, int(font_size * 0.5))  # Magic number 0.5
                total_label_width = text_width + (label_padding * 2)
                total_label_height = text_height + (label_padding * 2)

                if y1 - total_label_height >= 0:
                    label_x, label_y = x1, y1 - total_label_height
                else:
                    label_x, label_y = x1, y1

                label_x = max(0, min(label_x, img_width - total_label_width))
                label_y = max(0, min(label_y, img_height - total_label_height))
                draw.rectangle(
                    [
                        label_x,
                        label_y,
                        label_x + total_label_width,
                        label_y + total_label_height,
                    ],
                    fill=color,
                )
                draw.text(
                    (label_x + label_padding, label_y + label_padding),
                    label_text,
                    fill="white",
                    font=font,
                )

            img.save(output_image_path, quality=95)

        except Exception as e:
            print(f"Error creating custom visualization: {e}", flush=True)
            for result in self._model(input_image_path):
                result.save(filename=output_image_path)
                break
