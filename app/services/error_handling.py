import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum


class RollbackAction(Enum):
    DELETE_FILE = "delete_file"
    DELETE_DIRECTORY = "delete_directory"
    RESTORE_FILE = "restore_file"
    REVERT_MODEL_STATUS = "revert_model_status"
    CLEANUP_DATASET = "cleanup_dataset"
    NOTIFY_CALLBACK = "notify_callback"


@dataclass
class RollbackStep:
    action: RollbackAction
    target: str
    metadata: Dict[str, Any]
    timestamp: str


class ErrorHandlingService:
    def __init__(self):
        self.rollback_history: Dict[str, List[RollbackStep]] = {}
    
    def register_rollback_step(self, job_id: str, action: RollbackAction,
                               target: str, metadata: Optional[Dict] = None):
        """Register a rollback step for a job"""
        if job_id not in self.rollback_history:
            self.rollback_history[job_id] = []
        
        step = RollbackStep(
            action=action,
            target=target,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.rollback_history[job_id].append(step)
        print(f"Registered rollback step for job {job_id}: "
              f"{action.value} -> {target}", flush=True)
    
    async def execute_rollback(self, job_id: str) -> bool:
        """Execute all rollback steps for a failed job"""
        if job_id not in self.rollback_history:
            print(f"No rollback steps found for job {job_id}", flush=True)
            return True
        
        steps = reversed(self.rollback_history[job_id])  # Reverse order
        success_count = 0
        total_steps = len(self.rollback_history[job_id])
        
        print(f"Starting rollback for job {job_id} "
              f"({total_steps} steps)", flush=True)
        
        for step in steps:
            try:
                if await self._execute_single_rollback_step(step):
                    success_count += 1
                    print(f"Rollback step completed: {step.action.value} -> "
                          f"{step.target}", flush=True)
                else:
                    print(f"Rollback step failed: {step.action.value} -> "
                          f"{step.target}", flush=True)
                    
            except Exception as e:
                print(f"Rollback step error: {step.action.value} -> "
                      f"{step.target}: {str(e)}", flush=True)
        
        # Clean up rollback history
        if job_id in self.rollback_history:
            del self.rollback_history[job_id]
        
        print(f"Rollback completed for job {job_id}: "
              f"{success_count}/{total_steps} steps successful", flush=True)
        
        return success_count == total_steps
    
    async def _execute_single_rollback_step(self, step: RollbackStep) -> bool:
        """Execute a single rollback step"""
        try:
            if step.action == RollbackAction.DELETE_FILE:
                return self._delete_file(step.target)
                
            elif step.action == RollbackAction.DELETE_DIRECTORY:
                return self._delete_directory(step.target)
                
            elif step.action == RollbackAction.RESTORE_FILE:
                backup_path = step.metadata.get('backup_path')
                return self._restore_file(step.target, backup_path)
                
            elif step.action == RollbackAction.REVERT_MODEL_STATUS:
                return await self._revert_model_status(step.target,
                                                       step.metadata)
                
            elif step.action == RollbackAction.CLEANUP_DATASET:
                return await self._cleanup_dataset(step.target, step.metadata)
                
            elif step.action == RollbackAction.NOTIFY_CALLBACK:
                return await self._notify_rollback_callback(step.target,
                                                            step.metadata)
            
            return False
            
        except Exception as e:
            print(f"Error in rollback step {step.action.value}: {str(e)}",
                  flush=True)
            return False
    
    def _delete_file(self, file_path: str) -> bool:
        """Delete a file during rollback"""
        try:
            path = Path(file_path)
            if path.exists() and path.is_file():
                path.unlink()
                return True
            return True  # File doesn't exist, consider it successful
        except Exception:
            return False
    
    def _delete_directory(self, dir_path: str) -> bool:
        """Delete a directory during rollback"""
        try:
            path = Path(dir_path)
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                return True
            return True  # Directory doesn't exist, consider it successful
        except Exception:
            return False
    
    def _restore_file(self, target_path: str, backup_path: str) -> bool:
        """Restore a file from backup during rollback"""
        try:
            if not backup_path:
                return False
            
            backup = Path(backup_path)
            target = Path(target_path)
            
            if backup.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, target)
                return True
            return False
        except Exception:
            return False
    
    async def _revert_model_status(self, model_id: str,
                                   metadata: Dict[str, Any]) -> bool:
        """Revert model status during rollback"""
        try:
            # This would integrate with ModelManager to revert model status
            from .model_manager import ModelManager
            
            model_manager = ModelManager()
            previous_active_model = metadata.get('previous_active_model')
            
            if previous_active_model:
                return model_manager.set_active_model(previous_active_model)
            else:
                # If no previous model, delete the failed model
                return model_manager.delete_model(model_id, force=True)
                
        except Exception:
            return False
    
    async def _cleanup_dataset(self, yaml_config_path: str,
                               metadata: Dict[str, Any]) -> bool:
        """Clean up dataset during rollback"""
        try:
            from .dataset import DatasetService
            
            dataset_service = DatasetService()
            await dataset_service.cleanup_dataset(yaml_config_path)
            return True
            
        except Exception:
            return False
    
    async def _notify_rollback_callback(self, callback_url: str,
                                        metadata: Dict[str, Any]) -> bool:
        """Send rollback notification callback"""
        try:
            from .callback_service import CallbackService
            
            callback_service = CallbackService()
            job_id = metadata.get('job_id', 'unknown')
            error_message = metadata.get('error_message', 'Rollback executed')
            
            return await callback_service.notify_training_failed(
                job_id, f"Training failed and rolled back: {error_message}",
                callback_url
            )
            
        except Exception:
            return False
    
    def get_rollback_history(self, job_id: str) -> List[Dict[str, Any]]:
        """Get rollback history for a job"""
        if job_id not in self.rollback_history:
            return []
        
        return [
            {
                "action": step.action.value,
                "target": step.target,
                "metadata": step.metadata,
                "timestamp": step.timestamp
            }
            for step in self.rollback_history[job_id]
        ]
    
    def cleanup_old_rollback_history(self, max_age_hours: int = 24) -> int:
        """Clean up old rollback history entries"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (
            max_age_hours * 3600
        )
        
        jobs_to_remove = []
        for job_id, steps in self.rollback_history.items():
            if steps:
                latest_step_time = datetime.fromisoformat(
                    steps[-1].timestamp.replace('Z', '+00:00')
                ).timestamp()
                
                if latest_step_time < cutoff_time:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.rollback_history[job_id]
        
        return len(jobs_to_remove)
    
    async def handle_training_failure(self, job_id: str, error: str,
                                      job_data: Dict[str, Any]) -> bool:
        """Comprehensive failure handling with rollback"""
        print(f"Handling training failure for job {job_id}: {error}",
              flush=True)
        
        # Execute rollback
        rollback_success = await self.execute_rollback(job_id)
        
        # Send failure callback if configured
        callback_url = job_data.get('callbackUrl')
        if callback_url:
            try:
                from .callback_service import CallbackService
                callback_service = CallbackService()
                await callback_service.notify_training_failed(
                    job_id, error, callback_url
                )
            except Exception as callback_error:
                print(f"Failed to send failure callback: {callback_error}",
                      flush=True)
        
        return rollback_success
    
    def create_checkpoint(self, job_id: str, checkpoint_data: Dict[str, Any]):
        """Create a checkpoint for recovery purposes"""
        checkpoint_path = Path(f"/models/checkpoints/{job_id}.json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import json
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "job_id": job_id,
                    "data": checkpoint_data
                }, f, indent=2)
                
            print(f"Created checkpoint for job {job_id}", flush=True)
        except Exception as e:
            print(f"Failed to create checkpoint: {str(e)}", flush=True)
    
    def restore_from_checkpoint(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Restore job state from checkpoint"""
        checkpoint_path = Path(f"/models/checkpoints/{job_id}.json")
        
        try:
            if checkpoint_path.exists():
                import json
                with open(checkpoint_path, 'r') as f:
                    checkpoint = json.load(f)
                    
                print(f"Restored checkpoint for job {job_id}", flush=True)
                return checkpoint.get('data')
            
        except Exception as e:
            print(f"Failed to restore checkpoint: {str(e)}", flush=True)
        
        return None
