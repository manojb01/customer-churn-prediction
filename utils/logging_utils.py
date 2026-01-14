"""
Logging Utilities Module

Provides logging utilities for model performance and data summaries.
"""

import pandas as pd
import logging
from typing import Dict


class MLLogger:
    """Logger for ML-specific logging needs"""

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def log_model_performance(self, model_name: str, metrics: Dict[str, float], stage: str = "evaluation"):
        """Log model performance metrics"""
        self._logger.info(f"=== {model_name.upper()} {stage.upper()} METRICS ===")
        for metric, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                self._logger.info(f"{metric}: {value:.4f}")
            else:
                self._logger.info(f"{metric}: {value}")
        self._logger.info("=" * 50)

    def log_data_summary(self, df: pd.DataFrame, stage: str = "processing"):
        """Log comprehensive dataset summary"""
        self._logger.info(f"=== DATA SUMMARY - {stage.upper()} ===")
        self._logger.info(f"Shape: {df.shape}")
        self._logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        if 'Churn' in df.columns:
            churn_rate = df['Churn'].mean()
            self._logger.info(f"Churn rate: {churn_rate:.2%}")

        self._logger.info("=" * 50)


# Global instance for easy access
_ml_logger = MLLogger()


def log_model_metrics(model_name: str, metrics: Dict[str, float], stage: str = "evaluation"):
    """Log model performance metrics"""
    _ml_logger.log_model_performance(model_name, metrics, stage)


def log_data_summary(df: pd.DataFrame, stage: str = "processing"):
    """Log dataset summary information"""
    _ml_logger.log_data_summary(df, stage)


def get_logger(name: str = __name__) -> logging.Logger:
    """Get configured logger instance"""
    return logging.getLogger(name)
