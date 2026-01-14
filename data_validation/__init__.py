"""
Data Package

Contains data loading, validation, and processing logic.
"""

from .validation import validate_data, ValidationResult, DataValidator

__all__ = [
    "validate_data",
    "ValidationResult",
    "DataValidator"
]
