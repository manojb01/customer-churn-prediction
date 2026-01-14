"""
Customer Churn Prediction DAG

This DAG orchestrates the complete churn prediction pipeline:
1. Data loading and validation
2. Feature engineering
3. Model training
4. Model registration and promotion

All business logic is in src/ - this DAG is just orchestration.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path("/usr/local/airflow")
sys.path.insert(0, str(project_root))

from airflow.sdk import task, dag
from pendulum import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
import joblib

# Import from src package
from models.ml_pipeline import MLPipeline
from data_validation.validation import validate_data
from utils import get_config, register_model, promote_model


@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["ml", "churn", "production", "ecommerce"],
    description="Production customer churn prediction pipeline"
)
def churn_prediction_pipeline():
    """Complete production churn prediction pipeline"""

    @task()
    def load_ecommerce_data() -> Dict[str, Any]:
        """Load e-commerce dataset from Excel file"""
        try:
            config = get_config()
            data_config = config.get('data', {})
            excel_path = f"/usr/local/airflow/{data_config['source']['excel_path']}"
            sheet_name = data_config['source']['sheet_name']

            logging.info(f"Loading e-commerce data from {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            logging.info(f"Loaded data shape: {df.shape}")

            # Intelligent data cleaning and imputation
            df = _clean_and_impute_data(df)
            final_rows = len(df)
            logging.info(f"After cleaning and imputation: {final_rows} rows retained")

            # Ensure Churn column exists
            if 'Churn' in df.columns:
                df['Churn'] = df['Churn'].astype(int)
                churn_rate = df['Churn'].mean()
                logging.info(f"Dataset loaded: {len(df)} customers")
                logging.info(f"Churn rate: {churn_rate:.2%}")
            else:
                logging.error("No 'Churn' column found in dataset")
                return {'error': 'No Churn column found'}

            return {
                'data': df.to_dict('records'),
                'n_customers': len(df),
                'churn_rate': float(churn_rate)
            }

        except Exception as e:
            logging.error(f"Error loading e-commerce data: {e}")
            raise RuntimeError(f"Failed to load e-commerce dataset: {e}")

    def _clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent data cleaning and imputation"""
        import numpy as np

        initial_rows = len(df)
        logging.info(f"Starting data cleaning for {initial_rows} rows")

        # Remove rows where target (Churn) is missing
        if 'Churn' in df.columns:
            df = df.dropna(subset=['Churn'])
            logging.info(f"Removed {initial_rows - len(df)} rows with missing Churn values")

        # Handle ID columns - remove duplicates
        if 'CustomerID' in df.columns:
            duplicate_ids = df.duplicated(subset=['CustomerID'], keep='first').sum()
            if duplicate_ids > 0:
                df = df.drop_duplicates(subset=['CustomerID'], keep='first')
                logging.info(f"Removed {duplicate_ids} duplicate customer records")

        # Numerical imputation
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Churn' in numerical_columns:
            numerical_columns.remove('Churn')

        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                if col in ['Tenure', 'OrderCount', 'CouponUsed', 'NumberOfDeviceRegistered']:
                    df[col] = df[col].fillna(df[col].median())
                elif col in ['SatisfactionScore']:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 3)
                elif col in ['CashbackAmount', 'OrderAmountHikeFromlastYear']:
                    df[col] = df[col].fillna(0)
                elif col in ['DaySinceLastOrder']:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(df[col].median())
                logging.info(f"Imputed {col} (numerical)")

        # Categorical imputation
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if 'CustomerID' in categorical_columns:
            categorical_columns.remove('CustomerID')

        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                logging.info(f"Imputed {col} (categorical)")

        # Remove outliers using IQR method
        outlier_columns = ['Tenure', 'CashbackAmount', 'OrderCount']
        for col in outlier_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logging.info(f"Removed {outliers} outliers from {col}")

        final_rows = len(df)
        retention_rate = (final_rows / initial_rows * 100)
        logging.info(f"Data cleaning completed: {final_rows}/{initial_rows} rows retained ({retention_rate:.1f}%)")

        return df

    @task()
    def validate_data_task(data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data validation"""
        df = pd.DataFrame(data_dict['data'])

        # Validate the dataset
        validation_result = validate_data(df)

        if not validation_result.is_valid:
            raise ValueError(f"Data validation failed: {validation_result.issues}")

        logging.info("Data validation passed successfully")
        return {
            'data': df.to_dict('records'),
            'n_customers': len(df),
            'validation_passed': True
        }

    @task()
    def engineer_features(df_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced feature engineering"""
        df = pd.DataFrame(df_dict['data'])

        # Initialize ML pipeline with configuration
        config = get_config()
        ml_pipeline = MLPipeline(config)

        # Perform feature engineering
        target_col = 'Churn'
        feature_set = ml_pipeline.engineer_features(
            df,
            fit=True,
            target_col=target_col
        )

        # Prepare data for training
        X = feature_set.features.drop(columns=[target_col, 'CustomerID'], errors='ignore')
        y = feature_set.features[target_col]

        logging.info(f"Feature engineering completed: {len(X.columns)} features created")

        # Save to temporary files for serialization
        temp_dir = Path("/tmp/airflow_features")
        temp_dir.mkdir(exist_ok=True)

        X_path = temp_dir / "X_features.pkl"
        y_path = temp_dir / "y_target.pkl"
        transformers_path = temp_dir / "transformers.pkl"

        joblib.dump(X, X_path)
        joblib.dump(y, y_path)
        joblib.dump(feature_set.transformers, transformers_path)

        return {
            'X_path': str(X_path),
            'y_path': str(y_path),
            'transformers_path': str(transformers_path),
            'feature_names': list(X.columns),
            'feature_metadata': feature_set.feature_metadata,
            'n_samples': len(X),
            'n_features': len(X.columns)
        }

    @task()
    def train_models(feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train multiple models"""
        # Load data from paths
        X = joblib.load(feature_data['X_path'])
        y = joblib.load(feature_data['y_path'])
        feature_names = feature_data['feature_names']

        logging.info(
            f"Loaded training data: {feature_data['n_samples']} samples, {feature_data['n_features']} features")

        # Initialize ML pipeline
        config = get_config()
        ml_pipeline = MLPipeline(config)

        # Load and attach transformers
        try:
            transformers = joblib.load(feature_data['transformers_path'])
            ml_pipeline.transformers = transformers
        except Exception as e:
            logging.warning(f"Could not load transformers for training context: {e}")

        # Train models
        results = ml_pipeline.train_models(X, y, feature_names)

        # Save best model to production directory
        best_model_name = max(results.keys(),
                              key=lambda k: results[k].test_scores.get('roc_auc', 0))
        best_result = results[best_model_name]

        production_model_path = Path("models/production/churn_model.pkl")
        transformers_path = Path("models/transformers/feature_transformers.pkl")

        ml_pipeline.save_model(best_result, str(production_model_path), str(transformers_path))

        logging.info(
            f"Best model ({best_model_name}) saved to production with ROC-AUC: {best_result.test_scores['roc_auc']:.4f}")

        # Clean feature importance - replace NaN with None for JSON serialization
        import math
        feature_importance = best_result.feature_importance.copy()
        feature_importance = feature_importance.replace({float('nan'): None, float('inf'): None, float('-inf'): None})

        # Clean model metrics
        clean_metrics = {}
        for k, v in best_result.test_scores.items():
            if hasattr(v, 'item'):
                clean_metrics[k] = float(v)
            elif isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    clean_metrics[k] = None
                else:
                    clean_metrics[k] = float(v)
            else:
                clean_metrics[k] = v

        return {
            'best_model_name': best_model_name,
            'model_path': str(production_model_path),
            'transformers_path': str(transformers_path),
            'model_metrics': clean_metrics,
            'feature_importance': feature_importance.to_dict('records')
        }

    @task()
    def register_model_task(training_result: Dict[str, Any]) -> Dict[str, Any]:
        """Register model in model registry"""
        # Register the new model
        model_version = register_model(
            model_path=training_result['model_path'],
            model_name="churn_prediction",
            model_type=training_result['best_model_name'],
            metrics=training_result['model_metrics'],
            metadata={
                'feature_importance': training_result['feature_importance'],
                'transformers_path': training_result['transformers_path']
            }
        )

        # Promote to production if performance threshold is met
        roc_auc = training_result['model_metrics'].get('roc_auc', 0)
        if roc_auc > 0.75:  # Production threshold
            promote_model("churn_prediction", model_version, "production")
            logging.info(f"Model version {model_version} promoted to production")
        else:
            logging.info(f"Model version {model_version} registered but not promoted (ROC-AUC: {roc_auc:.4f})")

        return {
            'model_version': int(model_version),
            'promoted_to_production': bool(roc_auc > 0.75),
            'model_metrics': training_result['model_metrics']
        }

    # Task dependencies
    raw_data = load_ecommerce_data()
    validated_data = validate_data_task(raw_data)
    features = engineer_features(validated_data)
    training_result = train_models(features)
    model_info = register_model_task(training_result)


# Instantiate the DAG
churn_prediction_pipeline()
