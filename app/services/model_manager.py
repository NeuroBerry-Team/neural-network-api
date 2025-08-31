import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class ModelInfo:
    """Model information structure"""
    id: str
    name: str
    version: str
    model_type: str
    file_path: str
    config_path: Optional[str]
    created_at: str
    trained_by_job: Optional[str]
    dataset_info: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, float]]
    is_active: bool
    file_size: int
    description: Optional[str] = None


class ModelManager:
    def __init__(self):
        self.models_dir = Path("/models/weights")
        self.active_models_dir = Path("/models/active")
        self.metadata_file = Path("/models/models_metadata.json")
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load models metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.models_metadata = {
                        model_id: ModelInfo(**model_data)
                        for model_id, model_data in data.items()
                    }
            else:
                self.models_metadata = {}
        except Exception as e:
            print(f"Warning: Failed to load models metadata: {e}", flush=True)
            self.models_metadata = {}
    
    def _save_metadata(self) -> None:
        """Save models metadata to file"""
        try:
            data = {
                model_id: asdict(model_info)
                for model_id, model_info in self.models_metadata.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving models metadata: {e}", flush=True)
    
    def register_trained_model(self, job_id: str, model_name: str,
                               model_type: str, model_path: str,
                               config_path: Optional[str] = None,
                               dataset_info: Optional[Dict[str, Any]] = None,
                               description: Optional[str] = None) -> str:
        """
        Register a newly trained model
        
        Returns:
            str: The model ID
        """
        try:
            model_path_obj = Path(model_path)
            
            if not model_path_obj.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Generate model ID and version
            timestamp = int(datetime.now().timestamp())
            model_id = f"{model_name}_{job_id}_{timestamp}"
            version = self._generate_version(model_name)
            
            # Create organized model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Copy model file to organized location
            organized_model_path = model_dir / "model.pt"
            shutil.copy2(model_path_obj, organized_model_path)
            
            # Copy config file if provided
            organized_config_path = None
            if config_path and Path(config_path).exists():
                organized_config_path = model_dir / "config.yaml"
                shutil.copy2(config_path, organized_config_path)
            
            # Create model info
            model_info = ModelInfo(
                id=model_id,
                name=model_name,
                version=version,
                model_type=model_type,
                file_path=str(organized_model_path),
                config_path=(str(organized_config_path)
                             if organized_config_path else None),
                created_at=datetime.now(timezone.utc).isoformat(),
                trained_by_job=job_id,
                dataset_info=dataset_info,
                metrics=None,  # Will be updated later
                is_active=False,
                file_size=organized_model_path.stat().st_size,
                description=description
            )
            
            # Register in metadata
            self.models_metadata[model_id] = model_info
            self._save_metadata()
            
            print(f"Registered model: {model_id} (version {version})",
                  flush=True)
            return model_id
            
        except Exception as e:
            print(f"Error registering model: {e}", flush=True)
            raise
    
    def set_active_model(self, model_id: str) -> bool:
        """Set a model as the active inference model"""
        try:
            if model_id not in self.models_metadata:
                return False
            
            model_info = self.models_metadata[model_id]
            
            # Deactivate all other models
            for mid, minfo in self.models_metadata.items():
                minfo.is_active = False
            
            # Activate the selected model
            model_info.is_active = True
            
            # Copy model to active models directory
            active_model_path = self.active_models_dir / "best.pt"
            shutil.copy2(model_info.file_path, active_model_path)
            
            # Copy config if available
            if model_info.config_path:
                active_config_path = self.active_models_dir / "config.yaml"
                shutil.copy2(model_info.config_path, active_config_path)
            
            self._save_metadata()
            
            print(f"Set model {model_id} as active", flush=True)
            return True
            
        except Exception as e:
            print(f"Error setting active model: {e}", flush=True)
            return False
    
    def get_active_model(self) -> Optional[ModelInfo]:
        """Get the currently active model"""
        for model_info in self.models_metadata.values():
            if model_info.is_active:
                return model_info
        return None
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        return self.models_metadata.get(model_id)
    
    def list_models(self, model_name: Optional[str] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by name"""
        models = list(self.models_metadata.values())
        
        if model_name:
            models = [m for m in models if m.name == model_name]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete a model (cannot delete active model unless forced)"""
        try:
            if model_id not in self.models_metadata:
                return False
            
            model_info = self.models_metadata[model_id]
            
            if model_info.is_active and not force:
                print(f"Cannot delete active model {model_id}. "
                      f"Use force=True to override.", flush=True)
                return False
            
            # Remove model files
            model_path = Path(model_info.file_path)
            if model_path.exists():
                # Remove entire model directory
                model_dir = model_path.parent
                if model_dir.exists():
                    shutil.rmtree(model_dir)
            
            # Remove from metadata
            del self.models_metadata[model_id]
            self._save_metadata()
            
            print(f"Deleted model: {model_id}", flush=True)
            return True
            
        except Exception as e:
            print(f"Error deleting model: {e}", flush=True)
            return False
    
    def update_model_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """Update model performance metrics"""
        try:
            if model_id in self.models_metadata:
                self.models_metadata[model_id].metrics = metrics
                self._save_metadata()
                print(f"Updated metrics for model {model_id}", flush=True)
                return True
            return False
        except Exception as e:
            print(f"Error updating model metrics: {e}", flush=True)
            return False
    
    def _generate_version(self, model_name: str) -> str:
        """Generate version number for a model"""
        # Find existing models with the same name
        existing_models = [m for m in self.models_metadata.values() 
                          if m.name == model_name]
        
        if not existing_models:
            return "1.0.0"
        
        # Get the highest version number
        versions = []
        for model in existing_models:
            try:
                version_parts = model.version.split('.')
                if len(version_parts) >= 2:
                    major = int(version_parts[0])
                    minor = int(version_parts[1])
                    versions.append((major, minor))
            except ValueError:
                continue
        
        if not versions:
            return "1.0.0"
        
        # Increment minor version
        max_version = max(versions)
        return f"{max_version[0]}.{max_version[1] + 1}.0"
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about all models"""
        total_models = len(self.models_metadata)
        active_model = self.get_active_model()
        
        # Group by model type
        by_type = {}
        for model in self.models_metadata.values():
            if model.model_type not in by_type:
                by_type[model.model_type] = 0
            by_type[model.model_type] += 1
        
        # Calculate total storage used
        total_size = sum(model.file_size for model in self.models_metadata.values())
        
        return {
            'total_models': total_models,
            'active_model': active_model.id if active_model else None,
            'models_by_type': by_type,
            'total_storage_bytes': total_size,
            'total_storage_mb': round(total_size / (1024 * 1024), 2)
        }
    
    def get_best_model_path(self) -> str:
        """Get the path to the current best/active model"""
        active_model_path = self.active_models_dir / "best.pt"
        
        # If no active model is set, use the fallback
        if not active_model_path.exists():
            fallback_path = Path("/models/best.pt")
            if fallback_path.exists():
                return str(fallback_path)
            
            # If still no model, return the active path anyway (will be handled by inference service)
            return str(active_model_path)
        
        return str(active_model_path)
    
    def cleanup_old_models(self, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the latest N versions per model name"""
        try:
            # Group models by name
            models_by_name = {}
            for model_id, model_info in self.models_metadata.items():
                if model_info.name not in models_by_name:
                    models_by_name[model_info.name] = []
                models_by_name[model_info.name].append((model_id, model_info))
            
            deleted_count = 0
            
            for model_name, models in models_by_name.items():
                # Sort by creation date (newest first)
                models.sort(key=lambda x: x[1].created_at, reverse=True)
                
                # Keep only the latest versions, but never delete active models
                models_to_delete = models[keep_versions:]
                
                for model_id, model_info in models_to_delete:
                    if not model_info.is_active:
                        if self.delete_model(model_id):
                            deleted_count += 1
            
            print(f"Cleaned up {deleted_count} old model versions", flush=True)
            return deleted_count
            
        except Exception as e:
            print(f"Error during model cleanup: {e}", flush=True)
            return 0
