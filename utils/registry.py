"""
Model Registry Module

Manages model versioning, registration, and stage transitions.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime


class ModelRegistry:
    """Model registry for version management and stage transitions"""

    def __init__(self, registry_path: str = "models/registry.json"):
        self._registry_path = Path(registry_path)
        self._model_registry = {}
        self._logger = logging.getLogger(__name__)
        self._load_model_registry()

    def _load_model_registry(self):
        """Load existing model registry"""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r') as f:
                    self._model_registry = json.load(f)
            except Exception as e:
                self._logger.warning(f"Error loading model registry: {e}")
                self._model_registry = {'models': {}, 'version_counter': 0}
        else:
            self._model_registry = {'models': {}, 'version_counter': 0}
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_model_registry(self):
        """Save model registry to disk"""
        try:
            with open(self._registry_path, 'w') as f:
                json.dump(self._model_registry, f, indent=2)
        except Exception as e:
            self._logger.error(f"Error saving model registry: {e}")

    def register_model(self, model_path: str, model_name: str, model_type: str,
                      metrics: Dict[str, float], metadata: Optional[Dict] = None) -> int:
        """Register new model version"""
        self._model_registry['version_counter'] += 1
        version = self._model_registry['version_counter']

        if model_name not in self._model_registry['models']:
            self._model_registry['models'][model_name] = {'versions': {}, 'stages': {}}

        model_info = {
            'version': version,
            'model_type': model_type,
            'model_path': str(model_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'stage': 'staging'
        }

        self._model_registry['models'][model_name]['versions'][str(version)] = model_info
        self._model_registry['models'][model_name]['stages']['staging'] = version

        self._save_model_registry()
        self._logger.info(f"Registered {model_name} v{version} in staging")
        return version

    def promote_model(self, model_name: str, version: int, stage: str = 'production') -> bool:
        """Promote model to specified stage"""
        if (model_name not in self._model_registry['models'] or
            str(version) not in self._model_registry['models'][model_name]['versions']):
            self._logger.error(f"Model {model_name} v{version} not found")
            return False

        self._model_registry['models'][model_name]['stages'][stage] = version
        self._model_registry['models'][model_name]['versions'][str(version)]['stage'] = stage
        self._model_registry['models'][model_name]['versions'][str(version)]['promoted_at'] = datetime.now().isoformat()

        self._save_model_registry()
        self._logger.info(f"Promoted {model_name} v{version} to {stage}")
        return True

    def get_model_info(self, model_name: str, stage: str = 'production') -> Optional[Dict[str, Any]]:
        """Get model information for specific stage"""
        if (model_name not in self._model_registry['models'] or
            stage not in self._model_registry['models'][model_name]['stages']):
            return None

        version = self._model_registry['models'][model_name]['stages'][stage]
        return self._model_registry['models'][model_name]['versions'][str(version)]

    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self._model_registry['models']

    def get_latest_version(self, model_name: str) -> Optional[int]:
        """Get the latest version number for a model"""
        if model_name not in self._model_registry['models']:
            return None

        versions = self._model_registry['models'][model_name]['versions'].keys()
        return max(int(v) for v in versions) if versions else None

    def get_model_versions(self, model_name: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all versions of a model, optionally filtered by stage"""
        if model_name not in self._model_registry['models']:
            return []

        versions = self._model_registry['models'][model_name]['versions'].values()

        if stage:
            versions = [v for v in versions if v.get('stage') == stage]

        return sorted(versions, key=lambda x: x['version'], reverse=True)


# Global instance for easy access
_model_registry = ModelRegistry()


def register_model(model_path: str, model_name: str, model_type: str,
                  metrics: Dict[str, float], metadata: Optional[Dict] = None) -> int:
    """Register new model version"""
    return _model_registry.register_model(model_path, model_name, model_type, metrics, metadata)


def promote_model(model_name: str, version: int, stage: str = 'production') -> bool:
    """Promote model to specified stage"""
    return _model_registry.promote_model(model_name, version, stage)


def get_model_info(model_name: str, stage: str = 'production') -> Optional[Dict[str, Any]]:
    """Get model information for specific stage"""
    return _model_registry.get_model_info(model_name, stage)


def list_models() -> Dict[str, Any]:
    """List all registered models"""
    return _model_registry.list_models()
