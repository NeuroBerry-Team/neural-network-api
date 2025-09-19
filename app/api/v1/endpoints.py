from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from app.core.security import check_auth
from app.services.inference import InferenceService
from app.services.training import TrainingService
from app.services.dataset import DatasetService
from app.services.utils import cleanup_temp_files
import os
import base64
from pathlib import Path
from datetime import datetime, timezone

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/inferencia")
async def run_inference(
    image: UploadFile = File(...),
    model_name: str = Form("default"),
    auth=Depends(check_auth),
):
    """
    Run inference on uploaded image and return results as base64 encoded data.
    Receives image directly via HTTP upload.
    """
    try:
        # Validate uploaded file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Setup temporary directory for processing
        temp_images_dir = Path(os.environ.get("TEMP_IMAGES_DIR", "/tmp/temp_images"))
        temp_images_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique directory for this inference
        import uuid

        unique_id = uuid.uuid4().hex[:8]
        temp_dir = temp_images_dir / f"inference_{unique_id}"
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded image to temporary file
        download_path = (
            temp_dir
            / f"input_image.{image.filename.split('.')[-1] if '.' in image.filename else 'jpg'}"
        )

        with open(download_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)

        # Execute inference with specified model
        result_image_path = temp_dir / "inference_result.jpg"
        service = InferenceService()

        try:
            metadata = service.predict(
                download_path, result_image_path, model_name=model_name
            )
        except Exception as exc:
            error_msg = f"Error executing inference with model {model_name}: {str(exc)}"
            raise HTTPException(status_code=500, detail=error_msg)

        if not result_image_path.exists():
            raise HTTPException(status_code=500, detail="Result image not found")

        # Read result image and encode as base64
        with open(result_image_path, "rb") as img_file:
            result_image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Clean up temporary files
        temp_files = [download_path, result_image_path]
        cleanup_temp_files(temp_files, temp_dir)

        # Return results with base64 encoded image and metadata
        response_data = {
            "result_image": result_image_b64,
            "metadata": metadata,
            "content_type": "image/jpeg",
        }
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error generating inference with model {model_name}: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


# Training Job Management Endpoints
@router.post("/training")
async def create_training_job(request: Request, auth=Depends(check_auth)):
    """Create a new training job"""
    try:
        training_data = await request.json()

        # Validate required fields
        required_fields = ["modelName", "datasetId", "trainingParams"]
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


@router.post("/datasets/upload")
async def upload_dataset(
    dataset: UploadFile = File(...),
    dataset_id: int = Form(...),
    model_name: str = Form(...),
    auth=Depends(check_auth),
):
    """Upload and prepare dataset for training"""
    try:
        # Validate uploaded file
        if not dataset.filename.endswith((".zip", ".tar.gz", ".tar")):
            raise HTTPException(
                status_code=400, detail="Dataset must be a zip or tar file"
            )

        # Create dataset info object
        dataset_info = {
            "datasetId": dataset_id,
            "modelName": model_name,
            "filename": dataset.filename,
        }

        # Save uploaded dataset to temporary location
        temp_dataset_dir = Path("/tmp/uploaded_datasets")
        temp_dataset_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = temp_dataset_dir / f"dataset_{dataset_id}_{dataset.filename}"

        with open(dataset_path, "wb") as buffer:
            content = await dataset.read()
            buffer.write(content)

        # Prepare dataset using uploaded file
        dataset_service = DatasetService()
        yaml_config_path = await dataset_service.prepare_uploaded_dataset(
            dataset_path, dataset_info
        )

        return {
            "success": True,
            "message": "Dataset uploaded and prepared successfully",
            "yamlConfigPath": yaml_config_path,
        }

    except Exception as e:
        error_msg = f"Error processing uploaded dataset: {str(e)}"
        print(f"Dataset upload error: {error_msg}", flush=True)
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
                available_models.append(
                    {
                        "name": model_name,
                        "description": f"Model: {model_name}",
                        "file_path": str(model_file),
                    }
                )

        # Always include the default best.pt if it exists
        default_model = Path("/models/best.pt")
        if default_model.exists():
            available_models.insert(
                0,
                {
                    "name": "default",
                    "description": "Default trained model",
                    "file_path": str(default_model),
                },
            )

        return {
            "success": True,
            "models": available_models,
            "count": len(available_models),
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
                "last_modified": datetime.fromtimestamp(
                    file_stats.st_mtime, timezone.utc
                ).isoformat(),
            },
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
