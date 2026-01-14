"""
Configuration Management Module

Handles loading and managing configuration from YAML files with
environment variable substitution.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
import logging


class ConfigManager:
    """Configuration manager with environment variable substitution"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self._logger = logging.getLogger(__name__)

    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load and cache configuration"""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict[str, Any]:
        """Load and process configuration with environment variable substitution"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Recursively substitute environment variables
        config = self._substitute_env_vars(config)
        self._logger.info(f"Configuration loaded from {self.config_path}")
        return config

    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR:default} environment variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_spec = obj[2:-1]
            var_name, default = var_spec.split(':', 1) if ':' in var_spec else (var_spec, obj)
            return os.getenv(var_name, default)
        return obj

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config.get('features', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model training configuration"""
        return self.config.get('model', {})

    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow tracking configuration"""
        return self.config.get('mlflow', {})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.config.get('data', {})

    def reload(self):
        """Reload configuration from disk"""
        self._config = None
        return self.config


# Global instance for easy access
_config_manager = ConfigManager()


def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return _config_manager.config


def reload_config():
    """Reload configuration from disk"""
    return _config_manager.reload()
