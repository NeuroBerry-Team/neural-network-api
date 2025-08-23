import os
import json
import re
from pathlib import Path

def prepare_metadata(temp_dir, metadata_object_path, metadata):
    metadata_path = temp_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    original_size = metadata['image_info']['original_size']
    basic_metadata = {
        'detection-count': str(metadata['detection_count']),
        'original-width': (str(original_size[1]) if original_size else '0'),
        'original-height': (str(original_size[0]) if original_size else '0'),
        'detailed-metadata': metadata_object_path
    }
    return metadata_path, basic_metadata

def cleanup_temp_files(file_paths, temp_dir):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
    if temp_dir.exists() and temp_dir.is_dir():
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

def validate_request_data(request_data):
    if not request_data:
        raise ValueError("Request data is required")
    object_path = request_data.get('imgObjectKey')
    if not object_path:
        raise ValueError("imgObjectKey is required")
    if '..' in object_path or object_path.startswith('/'):
        raise ValueError("Invalid object path")
    if not re.match(r'^[a-zA-Z0-9\-_./]+$', object_path):
        raise ValueError("Object path contains invalid characters")
    return object_path

def setup_temp_directory(object_path, temp_images_dir):
    base_dir_id = object_path.split('/')[0]
    if not re.match(r'^[a-zA-Z0-9\-_]+$', base_dir_id):
        raise ValueError("Invalid directory name")
    filename = "original_img.jpg"
    temp_dir = (temp_images_dir / base_dir_id).resolve()
    if not str(temp_dir).startswith(str(temp_images_dir.resolve())):
        raise ValueError("Invalid temporary directory path")
    os.makedirs(temp_dir, exist_ok=True)
    download_path = (temp_dir / filename).resolve()
    return temp_dir, download_path, base_dir_id
