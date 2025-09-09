from fastapi import APIRouter, Depends, HTTPException, Request
from app.core.security import check_auth
from app.services.minio_connection import get_minio_client
from app.services.inference import InferenceService
from app.services.training import TrainingService
from app.services.dataset import DatasetService
from app.services.model_manager import ModelManager
from app.services.utils import (
    prepare_metadata,
    cleanup_temp_files,
    validate_request_data,
    setup_temp_directory,
)
import os
from pathlib import Path
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/inferencia")
async def run_inference(request: Request, auth=Depends(check_auth)):
    try:
        request_data = await request.json()
        object_path = validate_request_data(request_data)
        
        # Get optional model name, default to "default" (best.pt)
        model_name = request_data.get("modelName", "default")
        
        bucket_name = os.environ.get("S3_BUCKET_INFERENCES_RESULTS")
        live_url = os.environ.get("S3_LIVE_BASE_URL")
        temp_images_dir = Path(os.environ.get("TEMP_IMAGES_DIR", "/tmp/temp_images"))
        temp_dir, download_path, base_dir_id = setup_temp_directory(
            object_path, temp_images_dir
        )
        minio_client = get_minio_client()
        # Download image from MinIO
        minio_client.fget_object(bucket_name, str(object_path), str(download_path))
        
        # Execute inference with specified model
        result_image_path = temp_dir / "inference_result.jpg"
        service = InferenceService()
        try:
            metadata = service.predict(download_path, result_image_path, model_name=model_name)
        except Exception as exc:
            error_msg = f"Error executing inference {object_path} with model {model_name}: {str(exc)}"
            raise HTTPException(status_code=500, detail=error_msg)
        
        if not result_image_path.exists():
            raise HTTPException(status_code=500, detail="Result image not found")
        result_object_path = f"{base_dir_id}/inference_result.jpg"
        metadata_object_path = f"{base_dir_id}/metadata.json"
        metadata_path, basic_metadata = prepare_metadata(
            temp_dir, metadata_object_path, metadata
        )
        # Upload results to MinIO
        minio_client.fput_object(
            bucket_name,
            result_object_path,
            str(result_image_path),
            metadata=basic_metadata,
        )
        minio_client.fput_object(bucket_name, metadata_object_path, str(metadata_path))
        # Clean up temporary files
        temp_files = [download_path, result_image_path, metadata_path]
        cleanup_temp_files(temp_files, temp_dir)
        base_url = f"{live_url}{bucket_name}"
        response_data = {
            "generatedImgUrl": f"{base_url}/{result_object_path}",
            "metadataUrl": f"{base_url}/{metadata_object_path}",
        }
        return response_data
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        obj_path = locals().get("object_path", "unknown")
        model_name = locals().get("model_name", "unknown")
        error_msg = f"Error generating inference for {obj_path} with model {model_name}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


# Training Job Management Endpoints
@router.post("/training")
async def create_training_job(request: Request, auth=Depends(check_auth)):
    """Create a new training job"""
    try:
        training_data = await request.json()

        # Validate required fields
        required_fields = ["modelName", "datasetId", "datasetPath", "trainingParams"]
        for field in required_fields:
            if field not in training_data:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        training_service = TrainingService()
        job_id = training_service.create_job(training_data)

        # Start training in background
        training_service.start_training_async(job_id)

        return {
            "success": True,
            "jobId": job_id,
            "message": "Training job created and started successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error creating training job: {str(e)}"
        print(f"Training job creation error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/training")
async def list_training_jobs(auth=Depends(check_auth)):
    """List all training jobs with their current status"""
    try:
        training_service = TrainingService()
        jobs = training_service.get_all_jobs()

        return {
            "success": True,
            "jobs": [job.to_dict() for job in jobs.values()],
            "totalJobs": len(jobs),
        }

    except Exception as e:
        error_msg = f"Error listing training jobs: {str(e)}"
        print(f"Training jobs listing error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/training/{job_id}")
async def get_training_job_status(job_id: str, auth=Depends(check_auth)):
    """Get detailed status of a specific training job"""
    try:
        training_service = TrainingService()
        job = training_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        return {"success": True, "job": job.to_dict()}

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error getting training job status: {str(e)}"
        print(f"Training job status error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/training/{job_id}/progress")
async def get_training_progress(job_id: str, auth=Depends(check_auth)):
    """Get real-time progress of a training job"""
    try:
        training_service = TrainingService()
        job = training_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")

        # Calculate estimated time remaining
        estimated_remaining = "Unknown"
        if job.progress.get("percentage", 0) > 0:
            elapsed_time = (
                datetime.now(timezone.utc) - job.started_at if job.started_at else None
            )
            if elapsed_time:
                progress_pct = job.progress.get("percentage", 0) / 100
                if progress_pct > 0:
                    total_estimated = elapsed_time / progress_pct
                    remaining = total_estimated - elapsed_time
                    estimated_remaining = str(remaining).split(".")[0]

        return {
            "success": True,
            "jobId": job_id,
            "status": job.status.value,
            "progress": job.progress,
            "estimatedTimeRemaining": estimated_remaining,
            "elapsedTime": (
                str(
                    datetime.now(timezone.utc) - job.started_at
                    if job.started_at
                    else None
                ).split(".")[0]
                if job.started_at
                else None
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error getting training progress: {str(e)}"
        print(f"Training progress error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/training/{job_id}")
async def cancel_training_job(job_id: str, auth=Depends(check_auth)):
    """Cancel a training job"""
    try:
        training_service = TrainingService()
        success = training_service.cancel_job(job_id)

        if not success:
            raise HTTPException(
                status_code=400, detail="Job cannot be cancelled or doesn't exist"
            )

        return {
            "success": True,
            "message": f"Training job {job_id} cancelled successfully",
        }

    except Exception as e:
        error_msg = f"Error cancelling training job: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/training/{job_id}/cleanup")
async def manual_cleanup_training_job(job_id: str, auth=Depends(check_auth)):
    """Manual cleanup for a training job (admin use)"""
    try:
        training_service = TrainingService()
        job = training_service.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404, detail=f"Training job {job_id} not found"
            )

        # Run the comprehensive cleanup
        await training_service._comprehensive_final_cleanup(job_id, job)

        return {
            "success": True,
            "message": f"Manual cleanup completed for job {job_id}",
            "jobId": job_id,
            "jobStatus": job.status.value,
        }

    except Exception as e:
        error_msg = f"Error during manual cleanup: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/training/cleanup-all")
async def cleanup_all_completed_jobs(auth=Depends(check_auth)):
    """Cleanup all completed/failed/cancelled training jobs (admin use)"""
    try:
        training_service = TrainingService()
        all_jobs = training_service.get_all_jobs()

        cleanup_results = []
        completed_states = ["completed", "failed", "cancelled"]

        for job_id, job in all_jobs.items():
            if job.status.value in completed_states:
                try:
                    await training_service._comprehensive_final_cleanup(job_id, job)
                    cleanup_results.append(
                        {"jobId": job_id, "status": job.status.value, "cleaned": True}
                    )
                except Exception as cleanup_error:
                    cleanup_results.append(
                        {
                            "jobId": job_id,
                            "status": job.status.value,
                            "cleaned": False,
                            "error": str(cleanup_error),
                        }
                    )

        return {
            "success": True,
            "message": "Bulk cleanup completed",
            "results": cleanup_results,
            "totalJobs": len(cleanup_results),
        }

    except Exception as e:
        error_msg = f"Error during bulk cleanup: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/datasets")
async def list_datasets(auth=Depends(check_auth)):
    """List all available datasets"""
    try:
        dataset_service = DatasetService()
        datasets = dataset_service.list_available_datasets()

        return {"datasets": datasets, "count": len(datasets)}

    except Exception as e:
        error_msg = f"Error listing datasets: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/datasets/test")
async def test_dataset_download(request: Request, auth=Depends(check_auth)):
    """Test dataset download and preparation"""
    try:
        dataset_info = await request.json()

        # Validate required fields
        required_fields = ["datasetId", "datasetPath", "modelName"]
        for field in required_fields:
            if field not in dataset_info:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        dataset_service = DatasetService()
        yaml_config_path = await dataset_service.download_and_prepare_dataset(
            dataset_info
        )

        return {
            "success": True,
            "message": "Dataset downloaded and prepared successfully",
            "yamlConfigPath": yaml_config_path,
        }

    except Exception as e:
        error_msg = f"Error testing dataset download: {str(e)}"
        print(f"Dataset test error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


# Model Management Endpoints
@router.get("/models")
async def list_available_models(auth=Depends(check_auth)):
    """List all available models for inference (simplified)"""
    try:
        # Get models from the /models/weights directory
        models_dir = Path("/models/weights")
        available_models = []
        
        if models_dir.exists():
            # Look for .pt files in the weights directory
            for model_file in models_dir.glob("*.pt"):
                model_name = model_file.stem  # filename without extension
                available_models.append({
                    "name": model_name,
                    "description": f"Model: {model_name}",
                    "file_path": str(model_file)
                })
        
        # Always include the default best.pt if it exists
        default_model = Path("/models/best.pt")
        if default_model.exists():
            available_models.insert(0, {
                "name": "default",
                "description": "Default trained model",
                "file_path": str(default_model)
            })
        
        return {
            "success": True,
            "models": available_models,
            "count": len(available_models)
        }

    except Exception as e:
        error_msg = f"Error listing available models: {str(e)}"
        print(f"Models listing error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/models/{model_name}")
async def get_model_info(model_name: str, auth=Depends(check_auth)):
    """Get information about a specific model"""
    try:
        models_dir = Path("/models/weights")
        
        # Handle default model
        if model_name == "default":
            model_path = Path("/models/best.pt")
        else:
            model_path = models_dir / f"{model_name}.pt"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get file stats
        file_stats = model_path.stat()
        
        return {
            "success": True,
            "model": {
                "name": model_name,
                "description": f"Model: {model_name}",
                "file_path": str(model_path),
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime, timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error getting model info: {str(e)}"
        print(f"Model info error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)


# System Health and Status Endpoints
@router.get("/system/health")
async def system_health_check(auth=Depends(check_auth)):
    """Basic system health check"""
    try:
        # Check if models directory exists and has models
        models_dir = Path("/models/weights")
        default_model = Path("/models/best.pt")
        
        models_count = len(list(models_dir.glob("*.pt"))) if models_dir.exists() else 0
        has_default_model = default_model.exists()
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage("/models")

        return {
            "success": True,
            "system": {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "models": {
                "totalModels": models_count,
                "hasDefaultModel": has_default_model,
            },
            "storage": {
                "totalGB": round(disk_usage.total / (1024**3), 2),
                "usedGB": round(disk_usage.used / (1024**3), 2),
                "freeGB": round(disk_usage.free / (1024**3), 2),
                "usagePercent": round((disk_usage.used / disk_usage.total) * 100, 2),
            },
        }

    except Exception as e:
        error_msg = f"Error getting system health: {str(e)}"
        print(f"System health error: {error_msg}", flush=True)
        raise HTTPException(status_code=500, detail=error_msg)
