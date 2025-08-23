from fastapi import APIRouter, Depends, HTTPException, status, Request
from app.core.security import check_auth
from app.services.minio_connection import get_minio_client
from app.services.inference import InferenceService
from app.services.utils import prepare_metadata, cleanup_temp_files, validate_request_data, setup_temp_directory
import os
from pathlib import Path

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

from fastapi import Depends

@router.post("/inferencia")
async def run_inference(request: Request, auth=Depends(check_auth)):
    try:
        request_data = await request.json()
        object_path = validate_request_data(request_data)
        bucket_name = os.environ.get('S3_BUCKET_INFERENCES_RESULTS')
        live_url = os.environ.get('S3_LIVE_BASE_URL')
        TEMP_IMAGES_DIR = Path(os.environ.get('TEMP_IMAGES_DIR', '/tmp/temp_images'))
        temp_dir, download_path, base_dir_id = setup_temp_directory(object_path, TEMP_IMAGES_DIR)
        minio_client = get_minio_client()
        # Download image from MinIO
        minio_client.fget_object(bucket_name, str(object_path), str(download_path))
        # Execute inference
        result_image_path = temp_dir / "inference_result.jpg"
        service = InferenceService()
        try:
            metadata = service.predict(download_path, result_image_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error executing inference {object_path}: {str(exc)}")
        if not result_image_path.exists():
            raise HTTPException(status_code=500, detail="Result image not found")
        result_object_path = f"{base_dir_id}/inference_result.jpg"
        metadata_object_path = f"{base_dir_id}/metadata.json"
        metadata_path, basic_metadata = prepare_metadata(temp_dir, metadata_object_path, metadata)
        # Upload results to MinIO
        minio_client.fput_object(bucket_name, result_object_path, str(result_image_path), metadata=basic_metadata)
        minio_client.fput_object(bucket_name, metadata_object_path, str(metadata_path))
        # Clean up temporary files
        temp_files = [download_path, result_image_path, metadata_path]
        cleanup_temp_files(temp_files, temp_dir)
        base_url = f"{live_url}{bucket_name}"
        response_data = {
            "generatedImgUrl": f"{base_url}/{result_object_path}",
            "metadataUrl": f"{base_url}/{metadata_object_path}"
        }
        return response_data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        obj_path = locals().get('object_path', 'unknown')
        raise HTTPException(status_code=500, detail=f"Error generating inference for {obj_path}: {str(e)}")