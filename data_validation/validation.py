"""
Data Validation Module

Comprehensive data validation with quality scoring and reporting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats

from utils.config import get_config


@dataclass
class ValidationResult:
    """Container for data validation results with quality metrics"""
    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Get validation summary string"""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"{status} | Quality: {self.quality_score:.2f} | Issues: {len(self.issues)} | Warnings: {len(self.warnings)}"


class DataValidator:
    """Comprehensive data validator with quality scoring"""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self.config = get_config()

    def validate_data(self, df: pd.DataFrame, stage: str = 'training') -> ValidationResult:
        """
        Comprehensive data validation with quality scoring

        Args:
            df: DataFrame to validate
            stage: Validation stage ('training', 'prediction', 'production')

        Returns:
            ValidationResult with detailed findings and quality score
        """
        self._logger.info(f"Validating dataset at {stage} stage: {df.shape}")

        issues, warnings, metrics = [], [], {}

        # Run all validation checks in one pass
        structure_issues = self._validate_structure(df, stage)
        quality_issues, quality_metrics = self._validate_quality(df)
        stat_warnings = self._validate_statistics(df)

        issues.extend(structure_issues + quality_issues)
        warnings.extend(stat_warnings)
        metrics.update(quality_metrics)

        # Calculate quality score
        quality_score = self._calculate_quality_score(df, issues, warnings)

        # Create comprehensive report
        report = self._create_report(df, issues, warnings, metrics)

        result = ValidationResult(
            is_valid=not any(issue.startswith('CRITICAL:') for issue in issues),
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            report=report
        )

        self._logger.info(f"Validation complete: {result.summary}")
        return result

    def _validate_structure(self, df: pd.DataFrame, stage: str) -> List[str]:
        """Validate dataset structure and schema"""
        issues = []
        validation_rules = self.config.get('data', {}).get('validation_rules', {})

        # Basic structure checks
        if df.empty:
            issues.append("CRITICAL: Dataset is empty")
            return issues

        # Minimum sample size
        min_samples = validation_rules.get('min_samples', 100)
        if len(df) < min_samples:
            issues.append(f"WARNING: Only {len(df)} samples (minimum: {min_samples})")

        # Schema validation
        expected_cols = set(self.config.get('data', {}).get('expected_columns', []))
        if expected_cols:
            missing_cols = expected_cols - set(df.columns)
            if missing_cols:
                issues.append(f"Missing expected columns: {list(missing_cols)}")

        # Target validation for training
        if stage == 'training':
            target_issues = self._validate_target(df)
            issues.extend(target_issues)

        return issues

    def _validate_quality(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Validate data quality metrics"""
        issues = []
        metrics = {}
        validation_rules = self.config.get('data', {}).get('validation_rules', {})

        # Null analysis
        null_percentages = df.isnull().sum() / len(df)
        null_threshold = validation_rules.get('null_threshold', 0.3)
        critical_nulls = null_percentages[null_percentages > null_threshold]

        if not critical_nulls.empty:
            issues.append(f"High null percentages: {dict(critical_nulls.round(3))}")

        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        duplicate_threshold = validation_rules.get('duplicate_threshold', 0.05)

        if duplicate_percentage > duplicate_threshold:
            issues.append(f"High duplicate percentage: {duplicate_percentage:.2%}")

        # Data type validation
        type_issues = self._validate_types(df)
        issues.extend(type_issues)

        metrics.update({
            'null_percentages': null_percentages.to_dict(),
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage
        })

        return issues, metrics

    def _validate_statistics(self, df: pd.DataFrame) -> List[str]:
        """Validate statistical properties"""
        warnings = []
        validation_rules = self.config.get('data', {}).get('validation_rules', {})
        outlier_threshold = validation_rules.get('outlier_threshold', 3.0)

        # Get numerical columns
        numerical_cols = self._get_numerical_columns(df)

        for col in numerical_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Outlier detection
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_percentage = (z_scores > outlier_threshold).sum() / len(z_scores)

                if outlier_percentage > 0.05:
                    warnings.append(f"Column '{col}' has {outlier_percentage:.1%} potential outliers")

                # Constant values
                if df[col].nunique() <= 1:
                    warnings.append(f"Column '{col}' has constant values")

        return warnings

    def _validate_target(self, df: pd.DataFrame) -> List[str]:
        """Validate target variable for training datasets"""
        issues = []
        target_col = 'Churn'

        if target_col not in df.columns:
            issues.append(f"Target column '{target_col}' not found")
            return issues

        # Class balance check
        target_balance = df[target_col].value_counts(normalize=True)
        min_class_percentage = target_balance.min()
        balance_threshold = self.config.get('data', {}).get('validation_rules', {}).get('target_balance_threshold', 0.05)

        if min_class_percentage < balance_threshold:
            issues.append(f"Severe class imbalance: minority class is {min_class_percentage:.1%}")

        # Binary validation
        unique_values = df[target_col].dropna().unique()
        if len(unique_values) != 2 or not all(v in [0, 1] for v in unique_values):
            issues.append(f"Target should be binary [0,1], found: {unique_values}")

        return issues

    def _validate_types(self, df: pd.DataFrame) -> List[str]:
        """Validate column data types"""
        issues = []

        # Check numerical columns
        numerical_cols = self._get_numerical_columns(df)
        for col in numerical_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' should be numerical, found {df[col].dtype}")

        # Check categorical columns
        categorical_cols = self._get_categorical_columns()
        for col in categorical_cols:
            if col in df.columns and df[col].nunique() > 50:
                issues.append(f"Column '{col}' has {df[col].nunique()} unique values (too high for categorical)")

        return issues

    def _get_numerical_columns(self, df: pd.DataFrame = None) -> List[str]:
        """Get numerical columns from config or infer from dataframe"""
        config_numerical = (self.config.get('data', {}).get('numerical_columns', []) or
                           self.config.get('features', {}).get('numerical', []))

        if config_numerical:
            return config_numerical

        if df is not None:
            return df.select_dtypes(include=[np.number]).columns.tolist()

        return []

    def _get_categorical_columns(self) -> List[str]:
        """Get categorical columns from config"""
        return (self.config.get('data', {}).get('categorical_columns', []) or
                self.config.get('features', {}).get('categorical', []))

    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[str], warnings: List[str]) -> float:
        """Calculate overall data quality score (0-1)"""
        base_score = 1.0

        # Apply penalties
        critical_issues = [i for i in issues if i.startswith('CRITICAL:')]
        regular_issues = [i for i in issues if not i.startswith(('CRITICAL:', 'WARNING:'))]
        warning_issues = [i for i in issues if i.startswith('WARNING:')]

        base_score -= len(critical_issues) * 0.4    # Heavy penalty for critical issues
        base_score -= len(regular_issues) * 0.15    # Moderate penalty for issues
        base_score -= len(warning_issues) * 0.05    # Light penalty for warnings
        base_score -= len(warnings) * 0.02          # Very light penalty for warnings

        # Quality bonuses
        completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
        base_score += completeness * 0.1

        return max(0.0, min(1.0, base_score))

    def _create_report(self, df: pd.DataFrame, issues: List[str], warnings: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive validation report"""
        return {
            'dataset_info': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'columns': df.columns.tolist()
            },
            'quality_metrics': metrics,
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_data': df.isnull().sum().to_dict(),
            'summary_stats': df.describe().to_dict() if not df.empty else {},
            'issues': issues,
            'warnings': warnings,
            'validation_timestamp': datetime.now().isoformat()
        }


# Global instance for easy access
_data_validator = DataValidator()


def validate_data(df: pd.DataFrame, stage: str = 'training') -> ValidationResult:
    """Validate dataset with comprehensive checks"""
    return _data_validator.validate_data(df, stage)
