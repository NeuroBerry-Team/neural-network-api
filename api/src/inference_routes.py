import os
import json
import re
from pathlib import Path
from flask import Blueprint, request, jsonify
from .cloud_services.minio_connection import getMinioClient
from .inference_service.inference import InferenceService

from .security.decorators import check_auth

# load .env file (if exists) to environment
from dotenv import load_dotenv
load_dotenv()

# Get the path of the parent directory where this file is located (src)
SRC_DIR = Path(__file__).resolve().parent

# Temporary folder to save downloaded images
TEMP_IMAGES_DIR = SRC_DIR / "temp_images"
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)


def execute_inference(image_path, output_path):
    """Execute inference on the provided image."""
    print("Executing inference...", flush=True)
    service = InferenceService()  # Singleton
    boxes = service.predict(image_path, output_path)
    return boxes


def prepare_metadata(temp_dir, metadata_object_path, metadata):
    """Prepare and save metadata to a JSON file."""
    try:
        # temp_dir is already a Path object
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Fix the metadata format - original_size is a tuple (height, width)
        original_size = metadata['image_info']['original_size']
        basic_metadata = {
            'detection-count': str(metadata['detection_count']),
            # width is index 1
            'original-width': (str(original_size[1])
                               if original_size else '0'),
            # height is index 0
            'original-height': (str(original_size[0])
                                if original_size else '0'),
            'detailed-metadata': metadata_object_path
        }
    except Exception as e:
        print(f"Error saving metadata: {e}", flush=True)
        raise e  # Re-raise the exception instead of returning jsonify
    return metadata_path, basic_metadata


def download_image_from_minio(bucket_name, object_path, download_path):
    """Download image from MinIO storage."""
    print(f"Connecting to MinIO to download image: "
          f"{bucket_name}/{object_path}", flush=True)
    minio_client = getMinioClient()

    try:
        print(f"Downloading image from MinIO: "
              f"{bucket_name}/{object_path}", flush=True)
        minio_client.fget_object(bucket_name, str(object_path),
                                 str(download_path))
        print(f"Image downloaded to: {download_path.parent}", flush=True)
    except Exception as e:
        print(f"Error downloading image {object_path} from MinIO: {e}",
              flush=True)
        raise Exception(f"Error downloading image {object_path} from MinIO")


def upload_results_to_minio(bucket_name, result_image_path, metadata_path,
                            result_object_path, metadata_object_path,
                            basic_metadata):
    """Upload inference results and metadata to MinIO storage."""
    minio_client = getMinioClient()

    print(f"Uploading result image to MinIO: "
          f"{bucket_name}/{result_object_path}", flush=True)
    try:
        minio_client.fput_object(bucket_name, result_object_path,
                                 str(result_image_path),
                                 metadata=basic_metadata)
        print(f"Result image uploaded to: "
              f"{bucket_name}/{result_object_path}", flush=True)
        minio_client.fput_object(bucket_name, metadata_object_path,
                                 str(metadata_path))
    except Exception as upload_error:
        print(f"Error uploading result to MinIO: {str(upload_error)}",
              flush=True)
        raise Exception(f"Error uploading result to MinIO: "
                        f"{str(upload_error)}")


def cleanup_temp_files(file_paths, temp_dir):
    """Clean up temporary files and directory."""
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

        if temp_dir.exists() and temp_dir.is_dir():
            os.rmdir(temp_dir)

        print("Temporary files cleaned up successfully", flush=True)
    except Exception as cleanup_error:
        print(f"Warning: Error during cleanup: {str(cleanup_error)}",
              flush=True)


def validate_request_data(request_data):
    """Validate the incoming request data."""
    if not request_data:
        raise ValueError("Request data is required")

    object_path = request_data.get('imgObjectKey')
    if not object_path:
        raise ValueError("imgObjectKey is required")
    
    # Sanitize object path to prevent directory traversal
    if '..' in object_path or object_path.startswith('/'):
        raise ValueError("Invalid object path")
    
    # Allow only alphanumeric, hyphens, underscores, dots, and forward slashes
    if not re.match(r'^[a-zA-Z0-9\-_./]+$', object_path):
        raise ValueError("Object path contains invalid characters")

    return object_path


def setup_temp_directory(object_path):
    """Set up temporary directory structure for processing."""
    base_dir_id = object_path.split('/')[0]
    
    # Sanitize directory name
    if not re.match(r'^[a-zA-Z0-9\-_]+$', base_dir_id):
        raise ValueError("Invalid directory name")
    
    filename = "original_img.jpg"
    temp_dir = (TEMP_IMAGES_DIR / base_dir_id).resolve()
    
    # Ensure temp_dir is within TEMP_IMAGES_DIR
    if not str(temp_dir).startswith(str(TEMP_IMAGES_DIR.resolve())):
        raise ValueError("Invalid temporary directory path")

    os.makedirs(temp_dir, exist_ok=True)
    download_path = (temp_dir / filename).resolve()

    return temp_dir, download_path, base_dir_id


"""
------------------------
Below this the blueprint and the endpoints are declared
------------------------
"""
# Setup blueprint
inference = Blueprint('inference', __name__, url_prefix='/')


# Setup inference endpoint
@inference.route('/inferencia', methods=['POST'])
@check_auth
def run_inference():
    """
    Endpoint to execute inference on an image.
    """
    try:
        # Validate request data
        request_data = request.json
        object_path = validate_request_data(request_data)

        # Get environment variables
        bucket_name = os.environ.get('S3_BUCKET_INFERENCES_RESULTS')
        live_url = os.environ.get('S3_LIVE_BASE_URL')

        # Setup temporary directory and paths
        temp_dir, download_path, base_dir_id = setup_temp_directory(
            object_path)

        # Download image from MinIO
        download_image_from_minio(bucket_name, object_path, download_path)

        # Execute inference
        result_image_path = temp_dir / "inference_result.jpg"
        try:
            metadata = execute_inference(download_path, result_image_path)
        except Exception as exc:
            print(f"Error executing inference {object_path}: {str(exc)}",
                  flush=True)
            error_msg = f"Error executing inference {object_path}: {str(exc)}"
            return jsonify({"error": error_msg}), 500

        # Verify result image exists
        if not result_image_path.exists():
            print(f"Error: Result image not found at: {result_image_path}",
                  flush=True)
            return jsonify({"error": "Result image not found"}), 500

        # Prepare paths for MinIO upload
        result_object_path = f"{base_dir_id}/inference_result.jpg"
        metadata_object_path = f"{base_dir_id}/metadata.json"

        # Prepare and save metadata
        metadata_path, basic_metadata = prepare_metadata(
            temp_dir, metadata_object_path, metadata)

        # Upload results to MinIO
        upload_results_to_minio(
            bucket_name, result_image_path, metadata_path,
            result_object_path, metadata_object_path, basic_metadata
        )

        # Clean up temporary files
        temp_files = [download_path, result_image_path, metadata_path]
        cleanup_temp_files(temp_files, temp_dir)

        # Return response with URLs
        base_url = f"{live_url + bucket_name}"
        response_data = {
            "generatedImgUrl": f"{base_url}/{result_object_path}",
            "metadataUrl": f"{base_url}/{metadata_object_path}"
        }
        return jsonify(response_data), 200

    except ValueError as ve:
        print(f"Validation error: {str(ve)}", flush=True)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        obj_path = object_path if 'object_path' in locals() else 'unknown'
        print(f"Error generating inference for {obj_path}: {str(e)}",
              flush=True)
        return jsonify({"error": f"Error generating inference: {str(e)}"}), 500
