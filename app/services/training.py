import uuid
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any
from enum import Enum
from ultralytics import YOLO
from .dataset import DatasetService
from .model_manager import ModelManager
from .callback_service import CallbackService
from .error_handling import ErrorHandlingService


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob:
    def __init__(self, job_id: str, training_data: Dict[str, Any]):
        self.job_id = job_id
        self.training_data = training_data
        self.status = TrainingStatus.PENDING
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.progress: Dict[str, Any] = {}
        self.model_path: Optional[str] = None
        self.model_id: Optional[str] = None
        self.callback_urls = training_data.get('callbackUrls', {})
        self.callback_url: Optional[str] = training_data.get('callbackUrl')  # Legacy support
        
    def to_dict(self) -> Dict[str, Any]:
        started_at = (self.started_at.isoformat()
                      if self.started_at else None)
        completed_at = (self.completed_at.isoformat()
                        if self.completed_at else None)
        
        return {
            "jobId": self.job_id,
            "status": self.status.value,
            "createdAt": self.created_at.isoformat(),
            "startedAt": started_at,
            "completedAt": completed_at,
            "errorMessage": self.error_message,
            "progress": self.progress,
            "modelPath": self.model_path,
            "modelId": self.model_id,
            "callbackUrl": self.callback_url,
            "trainingData": self.training_data
        }


class TrainingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrainingService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self.jobs: Dict[str, TrainingJob] = {}
            self.dataset_service = DatasetService()
            self.model_manager = ModelManager()
            self.callback_service = CallbackService()
            self.error_handler = ErrorHandlingService()
            self._initialized = True
    
    def create_job(self, training_data: Dict[str, Any]) -> str:
        """Create a new training job and return job ID"""
        job_id = str(uuid.uuid4())
        job = TrainingJob(job_id, training_data)
        self.jobs[job_id] = job
        
        print(f"Created training job {job_id}", flush=True)
        return job_id
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> Dict[str, TrainingJob]:
        """Get all jobs"""
        return self.jobs.copy()
    
    def start_training_async(self, job_id: str) -> None:
        """Start training in a background thread"""
        def run_training():
            asyncio.run(self._run_training(job_id))
        
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
    
    async def _run_training(self, job_id: str) -> None:
        """Run the actual training process"""
        job = self.get_job(job_id)
        if not job:
            print(f"Job {job_id} not found", flush=True)
            return
        
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            job.progress = {"stage": "initializing", "percentage": 0}
            
            print(f"Starting training for job {job_id}", flush=True)
            
            # Send training started callback
            asyncio.create_task(
                self.callback_service.notify_training_started(
                    job_id, job.callback_url
                )
            )
            
            # Extract training parameters
            training_params = job.training_data.get('trainingParams', {})
            model_name = job.training_data.get('modelName', 'unnamed_model')
            model_type = job.training_data.get('modelType', 'YOLOv8_m')
            
            # Check if cancelled before downloading dataset
            if job.status == TrainingStatus.CANCELLED:
                print(f"Training job {job_id} was cancelled before dataset download", flush=True)
                return
            
            # Download and prepare dataset
            job.progress = {"stage": "downloading_dataset", "percentage": 15}
            dataset_service = self.dataset_service
            yaml_config = await dataset_service.download_and_prepare_dataset(
                job.training_data)
            
            # Check if cancelled after dataset preparation
            if job.status == TrainingStatus.CANCELLED:
                print(f"Training job {job_id} was cancelled after dataset preparation", flush=True)
                # Cleanup dataset if needed
                try:
                    await dataset_service.cleanup_dataset(job.training_data)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to cleanup dataset after cancellation: "
                          f"{cleanup_error}", flush=True)
                return
            
            # Initialize YOLO model
            job.progress = {"stage": "loading_model", "percentage": 25}
            
            # Load pretrained model (try local first, then download)
            # Convert model type to proper YOLO format (remove underscores)
            # YOLOv8_m -> yolov8m
            model_name = model_type.lower().replace('_', '')
            model_filename = f"{model_name}.pt"
            
            # Try local base_models first
            local_model_path = Path("/models/base_models") / model_filename
            
            if local_model_path.exists():
                print(f"Using local base model: {local_model_path}", flush=True)
                try:
                    model = YOLO(str(local_model_path))
                    print(f"Successfully loaded local model: {local_model_path}",
                          flush=True)
                except Exception as local_error:
                    print(f"Failed to load local model: {local_error}",
                          flush=True)
                    # Fall through to download attempt
                    model = None
            else:
                print(f"Local model not found: {local_model_path}", flush=True)
                model = None
            
            # If local model failed or doesn't exist, try downloading
            if model is None:
                print(f"Downloading pretrained model: {model_filename}",
                      flush=True)
                try:
                    # YOLO will automatically download the model
                    model = YOLO(model_filename)
                    print(f"Successfully downloaded model: {model_filename}",
                          flush=True)
                except Exception as download_error:
                    print(f"Failed to download model {model_filename}: "
                          f"{download_error}", flush=True)
                    raise Exception(f"Could not load or download pretrained "
                                    f"model {model_filename}")
            
            # Check if cancelled before training starts
            if job.status == TrainingStatus.CANCELLED:
                print(f"Training job {job_id} was cancelled before training start", flush=True)
                try:
                    await dataset_service.cleanup_dataset(job.training_data)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to cleanup dataset after cancellation: "
                          f"{cleanup_error}", flush=True)
                return
            
            # Prepare training arguments with custom callback
            cancellation_requested = False
            
            def check_cancellation_callback(trainer):
                """Custom callback to check for job cancellation during training"""
                nonlocal cancellation_requested
                
                if job.status == TrainingStatus.CANCELLED:
                    if not cancellation_requested:
                        print("Training cancellation requested, stopping gracefully", flush=True)
                        cancellation_requested = True
                        # Set trainer stop flags
                        trainer.stop_training = True
                        if hasattr(trainer, 'stopper'):
                            trainer.stopper.possible_stop = True
                            trainer.stopper.stop = True
                    return True
                
                # Update progress based on current epoch
                current_epoch = getattr(trainer, 'epoch', 0) + 1  # Start at 0
                total_epochs = training_params.get('epochs', 50)
                if total_epochs > 0:
                    progress_pct = 30 + int((current_epoch / total_epochs) * 60)
                    job.progress = {
                        "stage": "training",
                        "percentage": min(progress_pct, 90),
                        "epoch": current_epoch,
                        "total_epochs": total_epochs
                    }
                
                return False
            
            # Get model name from training data for project naming
            model_name_from_request = job.training_data.get(
                'modelName', 'unnamed_model')
            
            train_args = {
                'data': yaml_config,
                'epochs': training_params.get('epochs', 50),
                'imgsz': training_params.get('imageSize', 640),
                'batch': training_params.get('batchSize', 16),
                'lr0': training_params.get('learningRate', 0.01),
                'patience': training_params.get('patience', 30),
                'save': True,
                'project': str(Path("/models/weights")),
                'name': model_name_from_request,  # Use actual model name
                'exist_ok': True,
                'cache': False,
                'verbose': True
            }
            
            job.progress = {"stage": "training", "percentage": 30}
            
            # Add multiple callbacks for more aggressive cancellation checking
            model.add_callback('on_train_epoch_start',
                               check_cancellation_callback)
            
            def batch_check(trainer):
                return check_cancellation_callback(trainer)
            
            model.add_callback('on_train_batch_start', batch_check)
            model.add_callback('on_val_start', check_cancellation_callback)
            
            # Start training with better error handling for cancellation
            try:
                results = model.train(**train_args)
                
                # Check if training was cancelled (not failed)
                if job.status == TrainingStatus.CANCELLED:
                    print(f"Training job {job_id} cancelled successfully",
                          flush=True)
                    
                    # Send cancellation callback to notify Flask API
                    failed_callback_url = (job.callback_urls.get('failed') or 
                                           job.callback_url)
                    if failed_callback_url:
                        asyncio.create_task(
                            self._send_failure_callback(
                                job_id, 
                                "Training job cancelled during execution", 
                                failed_callback_url
                            )
                        )
                    
                    return  # Exit without setting to failed
                    
            except Exception as training_error:
                # Check if this is a cancellation-related error
                error_msg = str(training_error)
                if (job.status == TrainingStatus.CANCELLED or
                    cancellation_requested or
                    "No such file or directory" in error_msg):
                    print(f"Training job {job_id} cancelled "
                          f"(with cleanup errors)", flush=True)
                    # Don't change status to failed, keep as cancelled
                    return
                else:
                    # This is a real training failure
                    raise training_error
            
            # Save model path
            weights_dir = Path("/models/weights") / model_name
            best_model_path = weights_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                job.model_path = str(best_model_path)
                job.progress = {"stage": "completed", "percentage": 100}
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                
                # Register the trained model
                try:
                    config_path = weights_dir / "args.yaml"
                    dataset_info = {
                        "dataset_name": job.training_data.get('dataset',
                                                              'unknown'),
                        "classes": yaml_config.get('names', [])
                    }
                    
                    config_path_str = (str(config_path)
                                       if config_path.exists() else None)
                    
                    model_id = self.model_manager.register_trained_model(
                        job_id=job_id,
                        model_name=model_name,
                        model_type="yolo",
                        model_path=str(best_model_path),
                        config_path=config_path_str,
                        dataset_info=dataset_info,
                        description="YOLO model trained from job"
                    )
                    
                    job.model_id = model_id  # Store model ID in job
                    print(f"Registered model {model_id} for job {job_id}",
                          flush=True)
                    
                    # Send training completed callback
                    completed_callback_url = job.callback_urls.get('completed') or job.callback_url
                    if completed_callback_url:
                        asyncio.create_task(
                            self._send_completion_callback(
                                job_id, model_id, str(best_model_path), completed_callback_url
                            )
                        )
                    
                except Exception as model_reg_error:
                    print(f"Warning: Failed to register model: "
                          f"{model_reg_error}", flush=True)
                
                print(f"Training completed for job {job_id}. "
                      f"Model saved at {job.model_path}", flush=True)
                
                # Clean up dataset after successful training
                try:
                    self.dataset_service.cleanup_dataset(yaml_config)
                    print(f"Cleaned up dataset for job {job_id}", flush=True)
                    
                    # Also clean up temporary training artifacts but keep the model
                    model_name = job.training_data.get('modelName', 'unnamed_model')
                    dataset_id = job.training_data.get('datasetId')
                    
                    # Only clean up YAML configs, not the model files
                    config_dir = Path("/models/temp_configs")
                    if config_dir.exists():
                        pattern = f"{model_name}_dataset_{dataset_id}_*.yaml"
                        for config_file in config_dir.glob(pattern):
                            config_file.unlink()
                            print(f"Cleaned up config file: {config_file}", flush=True)
                            
                except Exception as cleanup_error:
                    print(f"Warning: Failed to cleanup after completion: "
                          f"{cleanup_error}", flush=True)
            else:
                raise FileNotFoundError("Trained model file not found")
                
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            print(f"Training failed for job {job_id}: {str(e)}", flush=True)
            
            # Clean up any partial model files on failure
            try:
                model_name = job.training_data.get('modelName', 'unnamed_model')
                weights_dir = Path("/models/weights") / model_name
                if weights_dir.exists():
                    import shutil
                    shutil.rmtree(weights_dir)
                    print(f"Cleaned up partial model files for job {job_id}", flush=True)
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup model files: {cleanup_error}", flush=True)
            
            # Send training failed callback
            failed_callback_url = job.callback_urls.get('failed') or job.callback_url
            if failed_callback_url:
                asyncio.create_task(
                    self._send_failure_callback(job_id, str(e), failed_callback_url)
                )
            
            # Clean up dataset on failure as well
            try:
                yaml_config = locals().get('yaml_config')
                if yaml_config:
                    self.dataset_service.cleanup_dataset(yaml_config)
                    print(f"Cleaned up dataset after failure for job {job_id}",
                          flush=True)
                
                # Also clean up any remaining artifacts
                model_name = job.training_data.get('modelName', 'unnamed_model')
                dataset_id = job.training_data.get('datasetId')
                self.dataset_service.cleanup_training_artifacts(model_name, dataset_id)
                
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup dataset after failure: "
                      f"{cleanup_error}", flush=True)
        
        finally:
            # Final comprehensive cleanup pass - runs regardless of outcome
            # This addresses any leftover files from threading/cancellation issues
            print(f"Running final cleanup validation for job {job_id}", flush=True)
            await self._comprehensive_final_cleanup(job_id, job)
    
    async def _comprehensive_final_cleanup(self, job_id: str, job: TrainingJob) -> None:
        """
        Comprehensive final cleanup that runs after training completion,
        failure, or cancellation to ensure no leftover files remain.
        """
        try:
            model_name = job.training_data.get('modelName', 'unnamed_model')
            dataset_id = job.training_data.get('datasetId')
            
            print(f"Final cleanup scan for job {job_id} - Model: {model_name}, "
                  f"Dataset: {dataset_id}", flush=True)
            
            cleanup_actions = []
            
            # 1. Check and clean up YAML config files
            config_dir = Path("/models/temp_configs")
            if config_dir.exists():
                patterns = [
                    f"{model_name}_*.yaml",
                    f"*_dataset_{dataset_id}_*.yaml",
                    f"{model_name}_dataset_{dataset_id}_*.yaml"
                ]
                
                for pattern in patterns:
                    for config_file in config_dir.glob(pattern):
                        try:
                            config_file.unlink()
                            cleanup_actions.append(f"Removed config: {config_file.name}")
                        except (FileNotFoundError, PermissionError):
                            pass  # Already cleaned up
            
            # 2. Check and clean up dataset directories
            datasets_dir = Path("/models/datasets")
            if datasets_dir.exists():
                # Look for dataset directories related to this job
                patterns = [
                    f"dataset_{dataset_id}_*",
                    f"*_dataset_{dataset_id}_*"
                ]
                
                for pattern in patterns:
                    for dataset_dir in datasets_dir.glob(pattern):
                        if dataset_dir.is_dir():
                            try:
                                import shutil
                                shutil.rmtree(dataset_dir)
                                cleanup_actions.append(f"Removed dataset dir: {dataset_dir.name}")
                            except (FileNotFoundError, PermissionError, OSError):
                                pass  # Already cleaned up or in use
            
            # 3. For cancelled/failed jobs, clean up partial model files
            if job.status in [TrainingStatus.CANCELLED, TrainingStatus.FAILED]:
                weights_dir = Path("/models/weights") / model_name
                if weights_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(weights_dir)
                        cleanup_actions.append(f"Removed partial model: {model_name}")
                    except (FileNotFoundError, PermissionError, OSError):
                        pass  # Already cleaned up
            
            # 4. Check for any orphaned temporary files
            temp_patterns = [
                f"/tmp/*{job_id}*",
                f"/tmp/*{model_name}*"
            ]
            
            for pattern in temp_patterns:
                for temp_file in Path("/tmp").glob(pattern.replace("/tmp/", "")):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            import shutil
                            shutil.rmtree(temp_file)
                        cleanup_actions.append(f"Removed temp: {temp_file.name}")
                    except (FileNotFoundError, PermissionError, OSError):
                        pass  # Already cleaned up
            
            # Report cleanup results
            if cleanup_actions:
                print(f"Final cleanup for job {job_id} completed. Actions taken:", flush=True)
                for action in cleanup_actions:
                    print(f"  - {action}", flush=True)
            else:
                print(f"Final cleanup for job {job_id} - no additional files found", flush=True)
                
        except Exception as cleanup_error:
            print(f"Warning: Final cleanup encountered error for job {job_id}: "
                  f"{cleanup_error}", flush=True)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job and clean up resources"""
        job = self.get_job(job_id)
        pending_or_running = [TrainingStatus.PENDING, TrainingStatus.RUNNING]
        if job and job.status in pending_or_running:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now(timezone.utc)
            print(f"Cancelled training job {job_id}", flush=True)
            
            # Send cancellation callback to notify Flask API to delete the model
            failed_callback_url = job.callback_urls.get('failed') or job.callback_url
            if failed_callback_url:
                asyncio.create_task(
                    self._send_failure_callback(
                        job_id, 
                        "Training job cancelled by user", 
                        failed_callback_url
                    )
                )
            
            # Clean up any partial model files
            try:
                model_name = job.training_data.get('modelName', 'unnamed_model')
                weights_dir = Path("/models/weights") / model_name
                if weights_dir.exists():
                    import shutil
                    shutil.rmtree(weights_dir)
                    print(f"Cleaned up model files for cancelled job {job_id}", flush=True)
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup model files: {cleanup_error}", flush=True)
            
            # Clean up dataset files
            try:
                # Clean up any YAML config files and dataset files
                model_name = job.training_data.get('modelName', 'unnamed_model')
                dataset_id = job.training_data.get('datasetId')
                self.dataset_service.cleanup_training_artifacts(model_name, dataset_id)
                print(f"Cleaned up training artifacts for cancelled job {job_id}", flush=True)
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup training artifacts: {cleanup_error}", flush=True)
            
            # Schedule a final comprehensive cleanup after a short delay
            # to allow any in-progress operations to complete
            asyncio.create_task(self._delayed_final_cleanup(job_id, job))
            
            return True
        return False
    
    async def _delayed_final_cleanup(self, job_id: str, job: TrainingJob) -> None:
        """
        Run final cleanup after a delay to allow threading operations to complete
        """
        import asyncio
        # Wait 5 seconds to allow any threaded operations to finish
        await asyncio.sleep(5)
        
        print(f"Running delayed final cleanup for cancelled job {job_id}", flush=True)
        await self._comprehensive_final_cleanup(job_id, job)
    
    def estimate_training_time(self, training_params: Dict[str, Any]) -> str:
        """Estimate training time based on parameters"""
        epochs = training_params.get('epochs', 50)
        batch_size = training_params.get('batchSize', 16)
        
        # Simple estimation: ~2-3 minutes per epoch for medium datasets
        base_time_per_epoch = 2.5  # minutes
        
        # Adjust based on batch size (smaller batch = more time)
        if batch_size <= 8:
            multiplier = 1.5
        elif batch_size <= 16:
            multiplier = 1.0
        else:
            multiplier = 0.7
        
        estimated_minutes = int(epochs * base_time_per_epoch * multiplier)
        
        if estimated_minutes < 60:
            return f"{estimated_minutes} minutes"
        else:
            hours = estimated_minutes // 60
            minutes = estimated_minutes % 60
            return f"{hours}h {minutes}m"

    async def _send_completion_callback(self, job_id: str, model_id: str, model_path: str, callback_url: str):
        """Send completion callback to Flask API"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                payload = {
                    "jobId": job_id,
                    "modelId": model_id,
                    "modelPath": model_path,
                    "status": "completed"
                }
                response = await client.post(callback_url, json=payload, timeout=10.0)
                if response.status_code == 200:
                    print(f"Completion callback sent successfully for job {job_id}", flush=True)
                else:
                    print(f"Completion callback failed for job {job_id}: {response.status_code}", flush=True)
        except Exception as e:
            print(f"Failed to send completion callback for job {job_id}: {str(e)}", flush=True)

    async def _send_failure_callback(self, job_id: str, error: str, callback_url: str):
        """Send failure callback to Flask API"""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                payload = {
                    "jobId": job_id,
                    "error": error,
                    "status": "failed"
                }
                response = await client.post(callback_url, json=payload, timeout=10.0)
                if response.status_code == 200:
                    print(f"Failure callback sent successfully for job {job_id}", flush=True)
                else:
                    print(f"Failure callback failed for job {job_id}: {response.status_code}", flush=True)
        except Exception as e:
            print(f"Failed to send failure callback for job {job_id}: {str(e)}", flush=True)
