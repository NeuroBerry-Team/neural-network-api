import asyncio
import httpx
from typing import Dict, Any, Optional
from datetime import datetime, timezone


class CallbackService:
    def __init__(self):
        self.timeout = 30.0
        self.max_retries = 3
        self.retry_delay = 2.0
    
    async def notify_training_started(self, job_id: str,
                                      callback_url: Optional[str] = None
                                      ) -> bool:
        """Notify brain-mapper that training has started"""
        if not callback_url:
            return True
        
        payload = {
            "jobId": job_id,
            "status": "started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Training job started successfully"
        }
        
        return await self._send_callback(callback_url, payload,
                                         "training_started")
    
    async def notify_training_progress(self, job_id: str,
                                       progress: Dict[str, Any],
                                       callback_url: Optional[str] = None
                                       ) -> bool:
        """Notify brain-mapper of training progress"""
        if not callback_url:
            return True
        
        payload = {
            "jobId": job_id,
            "status": "progress",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": progress
        }
        
        return await self._send_callback(callback_url, payload,
                                         "training_progress")
    
    async def notify_training_completed(self, job_id: str, model_id: str,
                                        model_path: str,
                                        metrics: Optional[Dict] = None,
                                        callback_url: Optional[str] = None
                                        ) -> bool:
        """Notify brain-mapper that training has completed successfully"""
        if not callback_url:
            return True
        
        payload = {
            "jobId": job_id,
            "status": "completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modelId": model_id,
            "modelPath": model_path,
            "metrics": metrics or {},
            "message": "Training completed successfully"
        }
        
        return await self._send_callback(callback_url, payload,
                                         "training_completed")
    
    async def notify_training_failed(self, job_id: str, error_message: str,
                                     callback_url: Optional[str] = None
                                     ) -> bool:
        """Notify brain-mapper that training has failed"""
        if not callback_url:
            return True
        
        payload = {
            "jobId": job_id,
            "status": "failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_message,
            "message": "Training job failed"
        }
        
        return await self._send_callback(callback_url, payload,
                                         "training_failed")
    
    async def notify_dataset_ready(self, dataset_name: str,
                                   yaml_config_path: str,
                                   callback_url: Optional[str] = None
                                   ) -> bool:
        """Notify brain-mapper that dataset is ready for training"""
        if not callback_url:
            return True
        
        payload = {
            "datasetName": dataset_name,
            "status": "ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "yamlConfigPath": yaml_config_path,
            "message": "Dataset prepared and ready for training"
        }
        
        return await self._send_callback(callback_url, payload,
                                         "dataset_ready")
    
    async def notify_model_activated(self, model_id: str, model_name: str,
                                     callback_url: Optional[str] = None
                                     ) -> bool:
        """Notify brain-mapper that a new model is now active"""
        if not callback_url:
            return True
        
        payload = {
            "modelId": model_id,
            "modelName": model_name,
            "status": "activated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Model {model_name} is now active for inference"
        }
        
        return await self._send_callback(callback_url, payload,
                                         "model_activated")
    
    async def _send_callback(self, callback_url: str, payload: Dict[str, Any],
                             callback_type: str) -> bool:
        """Send callback to brain-mapper with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        callback_url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "Neural-Network-API/1.0"
                        }
                    )
                    
                    if response.status_code in [200, 201, 202]:
                        print(f"Callback {callback_type} sent successfully "
                              f"to {callback_url} (attempt {attempt + 1})",
                              flush=True)
                        return True
                    else:
                        print(f"Callback {callback_type} failed with status "
                              f"{response.status_code}: {response.text} "
                              f"(attempt {attempt + 1})", flush=True)
                
            except httpx.TimeoutException:
                print(f"Callback {callback_type} timed out after "
                      f"{self.timeout}s (attempt {attempt + 1})", flush=True)
            except httpx.ConnectError:
                print(f"Callback {callback_type} connection failed "
                      f"(attempt {attempt + 1})", flush=True)
            except Exception as e:
                print(f"Callback {callback_type} error: {str(e)} "
                      f"(attempt {attempt + 1})", flush=True)
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        print(f"Callback {callback_type} failed after "
              f"{self.max_retries + 1} attempts", flush=True)
        return False
    
    def get_callback_status(self, job_id: str) -> Dict[str, Any]:
        """Get callback status for a job (for debugging/monitoring)"""
        return {
            "jobId": job_id,
            "lastCallback": None,
            "failedCallbacks": 0,
            "totalCallbacks": 0
        }
