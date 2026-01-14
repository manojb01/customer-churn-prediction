"""
Consolidated ML Pipeline for Customer Churn Prediction
Combines feature engineering, model training, and prediction functionality
"""

import pandas as pd
import numpy as np
import warnings
import os

# Filter out convergence warnings from sklearn
warnings.filterwarnings('ignore', category=Warning, module='sklearn.linear_model')
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
from scipy import stats
from dataclasses import dataclass
from contextlib import nullcontext
import joblib
import json
from datetime import datetime
from pathlib import Path
import logging

# Try optional imports
try:
    from xgboost import XGBClassifier
    import mlflow.xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')


@dataclass
class FeatureSet:
    """Container for engineered features."""
    features: pd.DataFrame
    feature_names: List[str]
    feature_metadata: Dict[str, Dict[str, Any]]
    transformers: Dict[str, Any]


@dataclass
class ModelResult:
    """Container for model training results."""
    model: Any
    best_params: Dict[str, Any]
    cv_scores: Dict[str, float]
    test_scores: Dict[str, float]
    feature_importance: pd.DataFrame
    predictions: Dict[str, np.ndarray]
    model_metadata: Dict[str, Any]


class MLPipeline:
    """Consolidated ML Pipeline for Churn Prediction"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize ML Pipeline with configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.transformers = {}
        self.feature_metadata = {}
        
        # MLflow setup
        self.mlflow_enabled = MLFLOW_AVAILABLE
        if self.mlflow_enabled:
            self._init_mlflow()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'features': {
                'numerical': [
                    'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                    'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                    'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
                ],
                'categorical': [
                    'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                    'PreferedOrderCat', 'MaritalStatus'
                ],
                'target': 'Churn',
                'engineering': {
                    'create_interaction_features': True,
                    'create_ratio_features': True,
                    'create_aggregation_features': True,
                    'handle_missing': {'strategy': 'smart'}
                }
            },
            'model': {
                'algorithms': [
                    {
                        'name': 'random_forest',
                        'enabled': True,
                        'params': {
                            'n_estimators': [100, 200],
                            'max_depth': [10, 20, None],
                            'min_samples_split': [2, 5],
                            'class_weight': ['balanced', None]
                        }
                    },
                    {
                        'name': 'logistic_regression',
                        'enabled': True,
                        'params': {
                            'penalty': ['l2'],
                            'C': [0.1, 1, 10],
                            'solver': ['liblinear'],
                            'class_weight': ['balanced', None]
                        }
                    }
                ],
                'training': {
                    'test_size': 0.2,
                    'cv_folds': 5,
                    'scoring_metric': 'roc_auc'
                },
                'optimization': {
                    'search_type': 'bayesian',
                    'n_iterations': 20
                }
            }
        }
    
    def _init_mlflow(self):
        """Initialize MLflow with fallbacks"""
        try:
            # Get MLflow config from the configuration
            mlflow_config = self.config.get('mlflow', {})
            tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5001')
            
            # Override with environment variable if set
            tracking_uri = os.getenv('MLFLOW_TRACKING_URI', tracking_uri)
            
            # If running outside Docker and URI points to 'mlflow' hostname, use localhost
            if 'mlflow:' in tracking_uri and not os.path.exists('/.dockerenv'):
                tracking_uri = tracking_uri.replace('mlflow:', 'localhost:')
                self.logger.info(f"Adjusted MLflow URI for host environment: {tracking_uri}")
            
            experiment_name = mlflow_config.get('experiment_name', 'churn_prediction')
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow initialized successfully with URI: {tracking_uri}")
        except Exception as e:
            self.logger.warning(f"MLflow disabled: {e}")
            self.mlflow_enabled = False
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True, target_col: str = 'Churn') -> FeatureSet:
        """Comprehensive feature engineering"""
        self.logger.info(f"Starting feature engineering (fit={fit})")
        
        df_eng = df.copy()
        
        # Handle CustomerID separately
        customer_id = None
        if 'CustomerID' in df_eng.columns:
            customer_id = df_eng['CustomerID']
            df_eng = df_eng.drop('CustomerID', axis=1)
        
        # Separate target if provided
        target = None
        if target_col and target_col in df_eng.columns:
            target = df_eng[target_col]
            df_eng = df_eng.drop(target_col, axis=1)
        
        # 1. Handle missing values
        df_eng = self._handle_missing_values(df_eng, fit)
        
        # 2. Create domain-specific features
        df_eng = self._create_domain_features(df_eng)
        
        # 3. Create interaction and ratio features
        if self.config['features']['engineering'].get('create_interaction_features', True):
            df_eng = self._create_interaction_features(df_eng)
        
        if self.config['features']['engineering'].get('create_ratio_features', True):
            df_eng = self._create_ratio_features(df_eng)
        
        # 4. Encode categorical variables
        df_eng = self._encode_categorical_features(df_eng, fit)
        
        # 5. Scale numerical features
        df_eng = self._scale_numerical_features(df_eng, fit)
        
        # 6. Feature selection (only during training)
        if fit and target is not None:
            df_eng = self._select_features(df_eng, target)
        elif not fit and 'selected_features' in self.transformers:
            selected_features = self.transformers['selected_features']
            df_eng = df_eng[selected_features]
        
        # Add back CustomerID and target
        if customer_id is not None:
            df_eng.insert(0, 'CustomerID', customer_id)
        if target is not None:
            df_eng[target_col] = target
        
        # Create feature metadata
        self._create_feature_metadata(df_eng, target_col)
        
        self.logger.info(f"Feature engineering completed: {len(df_eng.columns)} features")
        
        return FeatureSet(
            features=df_eng,
            feature_names=list(df_eng.columns),
            feature_metadata=self.feature_metadata,
            transformers=self.transformers
        )
    
    def _handle_missing_values(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Smart missing value handling"""
        strategy = self.config['features']['engineering']['handle_missing'].get('strategy', 'smart')
        
        if strategy == 'drop':
            return df.dropna()
        
        numerical_cols = [col for col in self.config['features']['numerical'] if col in df.columns]
        categorical_cols = [col for col in self.config['features']['categorical'] if col in df.columns]
        
        # Handle numerical columns with KNN imputer for moderate missing data
        if numerical_cols:
            knn_cols = [col for col in numerical_cols 
                       if df[col].isnull().sum() > 0 and 
                       df[col].isnull().sum() / len(df) < 0.3]
            
            if knn_cols and fit:
                self.transformers['knn_imputer'] = KNNImputer(n_neighbors=5)
                df[knn_cols] = self.transformers['knn_imputer'].fit_transform(df[knn_cols])
            elif knn_cols and not fit and 'knn_imputer' in self.transformers:
                df[knn_cols] = self.transformers['knn_imputer'].transform(df[knn_cols])
            
            # Use median for remaining numerical columns
            median_cols = [col for col in numerical_cols 
                          if col not in knn_cols and df[col].isnull().sum() > 0]
            
            if median_cols:
                if fit:
                    self.transformers['median_imputer'] = SimpleImputer(strategy='median')
                    df[median_cols] = self.transformers['median_imputer'].fit_transform(df[median_cols])
                elif 'median_imputer' in self.transformers:
                    df[median_cols] = self.transformers['median_imputer'].transform(df[median_cols])
        
        # Handle categorical columns with mode
        if categorical_cols:
            cat_missing_cols = [col for col in categorical_cols if df[col].isnull().sum() > 0]
            
            if cat_missing_cols:
                if fit:
                    self.transformers['mode_imputer'] = SimpleImputer(strategy='most_frequent')
                    df[cat_missing_cols] = self.transformers['mode_imputer'].fit_transform(df[cat_missing_cols])
                elif 'mode_imputer' in self.transformers:
                    df[cat_missing_cols] = self.transformers['mode_imputer'].transform(df[cat_missing_cols])
        
        return df
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create e-commerce domain-specific features"""
        # Customer engagement score
        if all(col in df.columns for col in ['HourSpendOnApp', 'NumberOfDeviceRegistered']):
            df['engagement_score'] = df['HourSpendOnApp'] * df['NumberOfDeviceRegistered']
        
        # Order behavior features
        if all(col in df.columns for col in ['OrderCount', 'OrderAmountHikeFromlastYear']):
            df['order_growth_flag'] = (df['OrderAmountHikeFromlastYear'] > 0).astype(int)
            df['order_frequency_category'] = pd.cut(
                df['OrderCount'], bins=[0, 1, 3, 5, np.inf],
                labels=['rare', 'occasional', 'regular', 'frequent']
            )
        
        # Customer satisfaction composite
        if all(col in df.columns for col in ['SatisfactionScore', 'Complain']):
            df['satisfaction_composite'] = df['SatisfactionScore'] * (1 - df['Complain'])
        
        # Tenure categories
        if 'Tenure' in df.columns:
            df['tenure_category'] = pd.cut(
                df['Tenure'], bins=[0, 6, 12, 24, np.inf],
                labels=['new', 'establishing', 'established', 'loyal']
            )
            df['is_new_customer'] = (df['Tenure'] < 6).astype(int)
        
        # Recency features
        if 'DaySinceLastOrder' in df.columns:
            df['recency_category'] = pd.cut(
                df['DaySinceLastOrder'], bins=[0, 7, 14, 30, np.inf],
                labels=['very_recent', 'recent', 'moderate', 'dormant']
            )
            df['is_dormant'] = (df['DaySinceLastOrder'] > 30).astype(int)
        
        # Payment preference risk
        if 'PreferredPaymentMode' in df.columns:
            high_risk_modes = ['Cash on Delivery', 'COD']
            df['risky_payment_preference'] = df['PreferredPaymentMode'].isin(high_risk_modes).astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # Tenure × Satisfaction interaction
        if all(col in df.columns for col in ['Tenure', 'SatisfactionScore']):
            df['tenure_satisfaction_interaction'] = df['Tenure'] * df['SatisfactionScore']
        
        # Engagement × Recency interaction
        if all(col in df.columns for col in ['HourSpendOnApp', 'DaySinceLastOrder']):
            df['engagement_recency_interaction'] = df['HourSpendOnApp'] / (df['DaySinceLastOrder'] + 1)
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features"""
        # Cashback per order
        if all(col in df.columns for col in ['CashbackAmount', 'OrderCount']):
            df['cashback_per_order'] = df['CashbackAmount'] / (df['OrderCount'] + 1)
        
        # Orders per tenure month
        if all(col in df.columns for col in ['OrderCount', 'Tenure']):
            df['orders_per_tenure_month'] = df['OrderCount'] / (df['Tenure'] + 1)
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = [col for col in self.config['features']['categorical'] if col in df.columns]
        
        # Add new categorical features
        new_categorical_cols = ['order_frequency_category', 'tenure_category', 'recency_category']
        categorical_cols.extend([col for col in new_categorical_cols if col in df.columns])
        
        if not categorical_cols:
            return df
        
        # One-hot encoding for nominal variables
        nominal_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                       'PreferedOrderCat', 'MaritalStatus']
        nominal_to_encode = [col for col in nominal_cols if col in categorical_cols]
        
        if nominal_to_encode:
            if fit:
                self.transformers['onehot_encoder'] = OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore'
                )
                encoded = self.transformers['onehot_encoder'].fit_transform(df[nominal_to_encode])
                feature_names = self.transformers['onehot_encoder'].get_feature_names_out(nominal_to_encode)
            else:
                encoded = self.transformers['onehot_encoder'].transform(df[nominal_to_encode])
                feature_names = self.transformers['onehot_encoder'].get_feature_names_out(nominal_to_encode)
            
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
            df = df.drop(columns=nominal_to_encode)
            df = pd.concat([df, encoded_df], axis=1)
        
        # Ordinal encoding for ordinal variables
        ordinal_cols = ['order_frequency_category', 'tenure_category', 'recency_category']
        ordinal_to_encode = [col for col in ordinal_cols if col in df.columns]
        
        if ordinal_to_encode:
            ordinal_mappings = {
                'order_frequency_category': ['rare', 'occasional', 'regular', 'frequent'],
                'tenure_category': ['new', 'establishing', 'established', 'loyal'],
                'recency_category': ['dormant', 'moderate', 'recent', 'very_recent']
            }
            
            for col in ordinal_to_encode:
                if col in ordinal_mappings:
                    if fit:
                        self.transformers[f'ordinal_{col}'] = OrdinalEncoder(
                            categories=[ordinal_mappings[col]],
                            handle_unknown='use_encoded_value',
                            unknown_value=-1
                        )
                        df[col] = self.transformers[f'ordinal_{col}'].fit_transform(df[[col]]).ravel()
                    elif f'ordinal_{col}' in self.transformers:
                        df[col] = self.transformers[f'ordinal_{col}'].transform(df[[col]]).ravel()
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Scale numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns from scaling
        exclude_from_scaling = ['CustomerID', 'Churn'] + [
            col for col in numerical_cols if col.endswith('_category') or 
            col.endswith('_flag') or col.startswith('is_')
        ]
        
        cols_to_scale = [col for col in numerical_cols if col not in exclude_from_scaling]
        
        if not cols_to_scale:
            return df
        
        if fit:
            self.transformers['scaler'] = RobustScaler()
            df[cols_to_scale] = self.transformers['scaler'].fit_transform(df[cols_to_scale])
        elif 'scaler' in self.transformers and self.transformers['scaler'] is not None:
            # Ensure columns are in the same order as during training
            scaler = self.transformers['scaler']
            if hasattr(scaler, 'feature_names_in_'):
                trained_features = scaler.feature_names_in_.tolist()
                # Reorder to match training
                cols_to_scale = [c for c in trained_features if c in df.columns]
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        
        return df
    
    def _select_features(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Select most informative features"""
        feature_cols = [col for col in df.columns if col != 'CustomerID']
        
        # Skip feature selection for small datasets
        if len(df) < 10 or len(feature_cols) == 0:
            return df
        
        # Calculate mutual information scores
        try:
            mi_scores = mutual_info_classif(df[feature_cols], target, random_state=42)
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}. Using all features.")
            return df
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select features with MI score > threshold
        mi_threshold = 0.01
        selected_features = feature_importance[
            feature_importance['mi_score'] > mi_threshold
        ]['feature'].tolist()
        
        # Ensure minimum number of features
        min_features = min(20, len(feature_cols))
        if len(selected_features) < min_features:
            selected_features = feature_importance.head(min_features)['feature'].tolist()
        
        # Always include CustomerID if present
        if 'CustomerID' in df.columns:
            selected_features = ['CustomerID'] + selected_features
        
        self.transformers['selected_features'] = selected_features
        self.transformers['feature_importance'] = feature_importance
        
        return df[selected_features]
    
    def _create_feature_metadata(self, df: pd.DataFrame, target_col: Optional[str]) -> None:
        """Create metadata for features"""
        for col in df.columns:
            if col == 'CustomerID' or col == target_col:
                continue
            
            metadata = {
                'dtype': str(df[col].dtype),
                'nunique': int(df[col].nunique()),
                'has_missing': bool(df[col].isnull().any()),
                'feature_type': 'numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                metadata.update({
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                })
            
            self.feature_metadata[col] = metadata
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Dict[str, ModelResult]:
        """Train multiple models with hyperparameter optimization"""
        self.logger.info("Starting model training pipeline")
        
        # Split data
        test_size = self.config['model']['training']['test_size']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Check class imbalance
        imbalance_ratio = y_train.value_counts().min() / y_train.value_counts().max()
        self.logger.info(f"Class imbalance ratio: {imbalance_ratio:.3f}")
        
        # Start single comprehensive MLflow run for all models
        if self.mlflow_enabled:
            enabled_models = [cfg['name'] for cfg in self.config['model']['algorithms'] if cfg.get('enabled', True)]
            # Create short model abbreviations
            model_abbrev = {
                'logistic_regression': 'LR',
                'random_forest': 'RF', 
                'gradient_boosting': 'GB',
                'xgboost': 'XGB',
                'svm': 'SVM'
            }
            model_codes = [model_abbrev.get(model, model[:2].upper()) for model in enabled_models]
            timestamp = pd.Timestamp.now().strftime('%m%d_%H%M')
            mlflow_run = mlflow.start_run(
                run_name=f"churn_{'+'.join(model_codes)}_{len(X_train)//1000}k_{timestamp}"
            )
        else:
            mlflow_run = None
        
        with mlflow_run if mlflow_run else nullcontext():
            # Log overall training parameters
            if self.mlflow_enabled:
                try:
                    enabled_models = [cfg['name'] for cfg in self.config['model']['algorithms'] if cfg.get('enabled', True)]
                    mlflow.log_params({
                        'total_models': len(enabled_models),
                        'model_names': enabled_models,
                        'n_features': len(feature_names),
                        'n_samples_train': len(X_train),
                        'n_samples_test': len(X_test),
                        'imbalance_ratio': imbalance_ratio,
                        'train_test_split': f"{len(X_train)}/{len(X_test)}"
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to log overall MLflow params: {e}")
            
            results = {}
            
            # Train each enabled model within the same run
            for model_config in self.config['model']['algorithms']:
                if not model_config.get('enabled', True):
                    continue
                    
                model_name = model_config['name']
                self.logger.info(f"Training {model_name}")
                
                # Hyperparameter optimization
                best_params = self._optimize_hyperparameters(X_train, y_train, model_config)
                
                # Train final model
                model = self._create_model(model_name, best_params)
                
                # Handle class imbalance
                if imbalance_ratio < 0.3 and IMBLEARN_AVAILABLE:
                    model = self._create_balanced_pipeline(model)
                
                model.fit(X_train, y_train)
                
                # Apply probability calibration for better probability estimates
                model_config_full = model_config
                if model_config_full.get('calibrate_probabilities', False):
                    self.logger.info(f"Applying probability calibration to {model_name}")
                    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                    calibrated_model.fit(X_train, y_train)
                    model = calibrated_model
                    self.logger.info(f"Probability calibration applied to {model_name}")
                
                # Evaluation
                cv_scores = self._evaluate_cross_validation(model, X_train, y_train)
                test_scores, predictions = self._evaluate_model(model, X_test, y_test)
                feature_importance = self._get_feature_importance(model, feature_names)
                
                # Log individual model metrics and artifacts if MLflow is enabled
                if self.mlflow_enabled:
                    try:
                        # Log hyperparameters with model prefix
                        for param, value in best_params.items():
                            mlflow.log_param(f"{model_name}_{param}", value)
                        
                        # Log metrics with model prefix
                        for metric, value in cv_scores.items():
                            mlflow.log_metric(f"{model_name}_cv_{metric}", value)
                        for metric, value in test_scores.items():
                            mlflow.log_metric(f"{model_name}_{metric}", value)
                        
                        # Log model artifacts and visualizations
                        if self.mlflow_enabled:
                            self.logger.info(f"Attempting to log model artifacts for {model_name}")
                            
                            # Save model locally and log as artifact
                            import tempfile
                            import joblib
                            import json
                            import os
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, classification_report
                            import numpy as np
                            
                            metadata_file = None
                            try:
                                # Try MLflow's sklearn log_model first
                                self.logger.info(f"Logging model to MLflow: {model_name}")
                                try:
                                    mlflow.sklearn.log_model(
                                        sk_model=model,
                                        artifact_path=f"{model_name}_model",
                                        registered_model_name=None,  # Don't auto-register individual models
                                        metadata={
                                            'model_name': model_name,
                                            'best_params': str(best_params),
                                            'model_type': type(model).__name__,
                                            'training_timestamp': pd.Timestamp.now().isoformat(),
                                            'test_scores': {k: float(v) for k, v in test_scores.items()}
                                        }
                                    )
                                    self.logger.info(f"✓ mlflow.sklearn.log_model() completed: {model_name}")
                                except Exception as mlflow_err:
                                    self.logger.warning(f"⚠️  mlflow.sklearn.log_model() error for {model_name}: {mlflow_err}")
                                
                                # Fallback: Save as pickle artifact
                                model_pkl_file = None
                                try:
                                    model_pkl_file = tempfile.NamedTemporaryFile(mode='wb', suffix=f'_{model_name}_model.pkl', delete=False)
                                    model_pkl_file.close()
                                    joblib.dump(model, model_pkl_file.name)
                                    mlflow.log_artifact(model_pkl_file.name, f"{model_name}_model")
                                    self.logger.info(f"✓ Logged {model_name} as pickle artifact")
                                except Exception as pkl_err:
                                    self.logger.error(f"✗ Failed to save {model_name} pickle: {pkl_err}")
                                finally:
                                    if model_pkl_file:
                                        try:
                                            os.unlink(model_pkl_file.name)
                                        except:
                                            pass
                                
                                # Save and log model metadata as additional artifact
                                metadata_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{model_name}_metadata.json', delete=False)
                                metadata = {
                                    'model_name': model_name,
                                    'best_params': best_params,
                                    'model_type': type(model).__name__,
                                    'training_timestamp': pd.Timestamp.now().isoformat(),
                                    'test_scores': test_scores
                                }
                                json.dump(metadata, metadata_file, indent=2)
                                metadata_file.close()
                                mlflow.log_artifact(metadata_file.name, f"{model_name}_model")
                                self.logger.info(f"✓ Logged model metadata: {metadata_file.name}")
                                
                                # Generate and log visualizations
                                self._log_model_visualizations(model, model_name, X_test, y_test, test_scores)
                                
                            except Exception as e:
                                self.logger.error(f"✗ Failed to log MLflow artifacts for {model_name}: {e}")
                                import traceback
                                self.logger.error(f"Traceback: {traceback.format_exc()}")
                            finally:
                                # Cleanup temp files
                                if metadata_file:
                                    try:
                                        os.unlink(metadata_file.name)
                                    except:
                                        pass
                    except Exception as e:
                        self.logger.error(f"Failed during model training: {e}")
                        import traceback
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                
                # Create result
                result = ModelResult(
                    model=model,
                    best_params=best_params,
                    cv_scores=cv_scores,
                    test_scores=test_scores,
                    feature_importance=feature_importance,
                    predictions={
                        'y_test': y_test,
                        'y_pred': predictions['y_pred'],
                        'y_proba': predictions['y_proba']
                    },
                    model_metadata={
                        'model_name': model_name,
                        'training_date': datetime.now().isoformat(),
                        'mlflow_run_id': mlflow.active_run().info.run_id if self.mlflow_enabled and mlflow.active_run() else None,
                        'feature_names': feature_names
                    }
                )
                
                results[model_name] = result
                self.logger.info(f"Model {model_name} completed - ROC AUC: {test_scores['roc_auc']:.4f}")
        
            # Select best model and log best model information
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k].test_scores.get('roc_auc', 0))
            self.logger.info(f"Best model: {best_model_name}")
            
            # Log best model information and comparison charts in the same run
            if self.mlflow_enabled and results:
                try:
                    # Log best model summary metrics
                    best_result = results[best_model_name]
                    mlflow.log_param("best_model", best_model_name)
                    mlflow.log_metric("best_model_roc_auc", best_result.test_scores['roc_auc'])
                    
                    # Log best model artifacts
                    self._log_best_model_artifacts(best_result, best_model_name)
                    
                    # Log model comparison charts
                    self._log_model_comparison_charts(results)
                    
                    self.logger.info("Logged best model information and comparison charts")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to log best model information: {e}")
        
        return results
    
    def _log_best_model_artifacts(self, best_result: ModelResult, best_model_name: str) -> None:
        """Log best model artifacts within the current MLflow run"""
        if not self.mlflow_enabled:
            self.logger.info("MLflow not enabled, skipping best model artifact logging")
            return
            
        try:
            import tempfile
            import joblib
            import json
            import os
            
            self.logger.info(f"Logging best model artifacts for {best_model_name}")
            
            metadata_file = None
            try:
                # Log the best model using MLflow's native model logging
                self.logger.info(f"Logging best model to MLflow: {best_model_name}")
                # First attempt: Use MLflow's sklearn log_model (may fail silently)
                try:
                    mlflow.sklearn.log_model(
                        sk_model=best_result.model,
                        artifact_path="best_model",
                        registered_model_name="customer_churn_model",  # Register directly to Model Registry
                        metadata={
                            'model_name': best_model_name,
                            'best_params': str(best_result.best_params),
                            'model_type': type(best_result.model).__name__,
                            'test_scores': {k: float(v) for k, v in best_result.test_scores.items()},
                            'training_timestamp': pd.Timestamp.now().isoformat(),
                            'is_best_model': True
                        }
                    )
                    self.logger.info(f"✓ mlflow.sklearn.log_model() completed for: {best_model_name}")
                except Exception as mlflow_err:
                    self.logger.warning(f"⚠️  mlflow.sklearn.log_model() encountered error: {mlflow_err}")
                
                # Fallback: ALWAYS save model as pickle artifact (ensures it's available)
                model_pkl_file = None
                try:
                    model_pkl_file = tempfile.NamedTemporaryFile(mode='wb', suffix='_model.pkl', delete=False)
                    model_pkl_file.close()
                    joblib.dump(best_result.model, model_pkl_file.name)
                    mlflow.log_artifact(model_pkl_file.name, "best_model")
                    self.logger.info(f"✓ Logged model as pickle artifact: {model_pkl_file.name}")
                except Exception as pkl_err:
                    self.logger.error(f"✗ Failed to save model pickle: {pkl_err}")
                finally:
                    if model_pkl_file:
                        try:
                            os.unlink(model_pkl_file.name)
                        except:
                            pass
                
                # Save and log best model metadata as additional artifact
                metadata_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_best_{best_model_name}_metadata.json', delete=False)
                best_metadata = {
                    'model_name': best_model_name,
                    'best_params': best_result.best_params,
                    'model_type': type(best_result.model).__name__,
                    'test_scores': best_result.test_scores,
                    'training_timestamp': pd.Timestamp.now().isoformat(),
                    'is_best_model': True,
                    'feature_importance': best_result.feature_importance.to_dict() if hasattr(best_result.feature_importance, 'to_dict') else best_result.feature_importance
                }
                json.dump(best_metadata, metadata_file, indent=2)
                metadata_file.close()
                mlflow.log_artifact(metadata_file.name, "best_model")
                self.logger.info(f"✓ Logged best model metadata: {metadata_file.name}")
                
                # Also log fitted transformers if available
                transformers_temp_file = None
                try:
                    if hasattr(self, 'transformers') and self.transformers:
                        import tempfile as _tf
                        import os as _os
                        transformers_temp_file = _tf.NamedTemporaryFile(suffix='_feature_transformers.pkl', delete=False)
                        transformers_temp_file.close()  # Close before joblib writes
                        joblib.dump(self.transformers, transformers_temp_file.name)
                        mlflow.log_artifact(transformers_temp_file.name, "best_model")
                        self.logger.info(f"✓ Logged feature transformers artifact: {transformers_temp_file.name}")
                        # Clean up
                        try:
                            _os.unlink(transformers_temp_file.name)
                        except:
                            pass
                except Exception as te:
                    self.logger.warning(f"Could not log transformers: {te}")
                    import traceback
                    self.logger.warning(traceback.format_exc())
                
                # Log model metadata as tags
                mlflow.set_tag("best_model_type", best_model_name)
                mlflow.set_tag("best_roc_auc", f"{best_result.test_scores['roc_auc']:.4f}")
                mlflow.set_tag("best_model_selected", "true")
                self.logger.info("✓ Logged best model tags")
                
            except Exception as inner_e:
                self.logger.error(f"✗ Failed to log best model artifacts: {inner_e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            finally:
                # Cleanup temp files
                if metadata_file:
                    try:
                        os.unlink(metadata_file.name)
                    except:
                        pass
                    
        except Exception as e:
            self.logger.error(f"Failed in best model artifact logging: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _log_model_visualizations(self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, test_scores: Dict[str, float]):
        """Generate and log comprehensive model visualizations"""
        if not self.mlflow_enabled:
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, classification_report
        import tempfile
        import os
        import numpy as np
        
        try:
            self.logger.info(f"Generating visualizations for {model_name}...")
            # Set style for better looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # 1. Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Churn', 'Churn'], 
                       yticklabels=['No Churn', 'Churn'])
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save and log confusion matrix
            cm_file = tempfile.NamedTemporaryFile(suffix=f'_{model_name}_confusion_matrix.png', delete=False)
            plt.savefig(cm_file.name, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(cm_file.name, f"visualizations/{model_name}")
            plt.close()
            
            # 2. ROC Curve
            if hasattr(model, 'predict_proba'):
                plt.figure(figsize=(8, 6))
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_scores.get("roc_auc", 0):.3f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_name} - ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save and log ROC curve
                roc_file = tempfile.NamedTemporaryFile(suffix=f'_{model_name}_roc_curve.png', delete=False)
                plt.savefig(roc_file.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(roc_file.name, f"visualizations/{model_name}")
                plt.close()
                
                # 3. Precision-Recall Curve
                plt.figure(figsize=(8, 6))
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {test_scores.get("avg_precision", 0):.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{model_name} - Precision-Recall Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save and log PR curve
                pr_file = tempfile.NamedTemporaryFile(suffix=f'_{model_name}_pr_curve.png', delete=False)
                plt.savefig(pr_file.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(pr_file.name, f"visualizations/{model_name}")
                plt.close()
            
            # 4. Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                feature_names = X_test.columns[:20]  # Top 20 features
                importances = model.feature_importances_[:20]
                indices = np.argsort(importances)[::-1]
                
                plt.bar(range(len(indices)), importances[indices])
                plt.title(f'{model_name} - Top 20 Feature Importances')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
                
                # Save and log feature importance
                fi_file = tempfile.NamedTemporaryFile(suffix=f'_{model_name}_feature_importance.png', delete=False)
                plt.savefig(fi_file.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(fi_file.name, f"visualizations/{model_name}")
                plt.close()
            
            # 5. Classification Report as text file
            report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
            report_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{model_name}_classification_report.txt', delete=False)
            report_file.write(f"Classification Report for {model_name}\n")
            report_file.write("="*50 + "\n")
            report_file.write(report)
            report_file.close()
            mlflow.log_artifact(report_file.name, f"reports/{model_name}")
            
            # Cleanup temp files
            temp_files = [cm_file.name, report_file.name]
            if hasattr(model, 'predict_proba'):
                temp_files.extend([roc_file.name, pr_file.name])
            if hasattr(model, 'feature_importances_'):
                temp_files.append(fi_file.name)
                
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            self.logger.info(f"✓ Successfully logged visualizations for {model_name}")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to generate visualizations for {model_name}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _log_model_comparison_charts(self, results: Dict[str, 'ModelResult']):
        """Generate and log model comparison charts"""
        if not self.mlflow_enabled:
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        import tempfile
        import os
        import numpy as np
        
        try:
            self.logger.info("Generating model comparison charts...")
            # Prepare data for comparison
            model_names = list(results.keys())
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            
            # Create metrics comparison data
            comparison_data = {}
            for metric in metrics:
                comparison_data[metric] = [results[model].test_scores.get(metric, 0) for model in model_names]
            
            # 1. Metrics Comparison Bar Chart
            plt.figure(figsize=(14, 10))
            x = np.arange(len(model_names))
            width = 0.15
            colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
            
            for i, metric in enumerate(metrics):
                plt.bar(x + i * width, comparison_data[metric], width, 
                       label=metric.upper(), color=colors[i], alpha=0.8)
            
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * 2, model_names, rotation=45, ha='right')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save and log comparison chart
            comparison_file = tempfile.NamedTemporaryFile(suffix='_model_comparison.png', delete=False)
            plt.savefig(comparison_file.name, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(comparison_file.name, "comparisons")
            plt.close()
            
            # 2. ROC AUC Ranking Chart
            plt.figure(figsize=(10, 8))
            roc_scores = comparison_data['roc_auc']
            sorted_indices = np.argsort(roc_scores)[::-1]
            sorted_models = [model_names[i] for i in sorted_indices]
            sorted_scores = [roc_scores[i] for i in sorted_indices]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_models)))
            bars = plt.bar(range(len(sorted_models)), sorted_scores, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Models (Ranked by ROC-AUC)')
            plt.ylabel('ROC-AUC Score')
            plt.title('Model Ranking by ROC-AUC Score')
            plt.xticks(range(len(sorted_models)), sorted_models, rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save and log ranking chart
            ranking_file = tempfile.NamedTemporaryFile(suffix='_model_ranking.png', delete=False)
            plt.savefig(ranking_file.name, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(ranking_file.name, "comparisons")
            plt.close()
            
            # 3. Performance Heatmap
            plt.figure(figsize=(10, 8))
            heatmap_data = []
            for model_name in model_names:
                model_scores = [results[model_name].test_scores.get(metric, 0) for metric in metrics]
                heatmap_data.append(model_scores)
            
            sns.heatmap(heatmap_data, 
                       xticklabels=[m.upper() for m in metrics],
                       yticklabels=model_names,
                       annot=True, fmt='.3f', cmap='RdYlBu_r',
                       center=0.5, vmin=0, vmax=1)
            plt.title('Model Performance Heatmap')
            plt.tight_layout()
            
            # Save and log heatmap
            heatmap_file = tempfile.NamedTemporaryFile(suffix='_performance_heatmap.png', delete=False)
            plt.savefig(heatmap_file.name, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(heatmap_file.name, "comparisons")
            plt.close()
            
            # 4. Create detailed comparison report
            report_content = "Model Comparison Report\n"
            report_content += "=" * 50 + "\n\n"
            
            for model_name in sorted_models:
                report_content += f"{model_name.upper()}:\n"
                report_content += "-" * 20 + "\n"
                for metric in metrics:
                    score = results[model_name].test_scores.get(metric, 0)
                    report_content += f"  {metric.upper()}: {score:.4f}\n"
                
                # Add best parameters if available
                if hasattr(results[model_name], 'best_params'):
                    report_content += f"  Best Parameters: {results[model_name].best_params}\n"
                report_content += "\n"
            
            # Save and log comparison report
            report_file = tempfile.NamedTemporaryFile(mode='w', suffix='_model_comparison_report.txt', delete=False)
            report_file.write(report_content)
            report_file.close()
            mlflow.log_artifact(report_file.name, "reports")
            
            # Cleanup temp files
            temp_files = [comparison_file.name, ranking_file.name, heatmap_file.name, report_file.name]
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            self.logger.info("✓ Successfully logged model comparison charts and reports")
            
        except Exception as e:
            self.logger.error(f"✗ Failed to generate model comparison charts: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                 model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna (fallback to first param values if unavailable)"""
        model_name = model_config['name']
        param_space = model_config['params']
        if not OPTUNA_AVAILABLE:
            return {k: (v[0] if isinstance(v, list) and v else v) for k,v in param_space.items()}
        
        def objective(trial):
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, type(None))) for v in param_values):
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                    elif all(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            try:
                model = self._create_model(model_name, params)
                cv_folds = self.config['model']['training']['cv_folds']
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                scores = cross_validate(
                    model, X_train, y_train, cv=cv, 
                    scoring=self.config['model']['training']['scoring_metric'],
                    n_jobs=-1, error_score='raise'
                )
                return scores['test_score'].mean()
            except Exception:
                return 0.0
        
        # Run optimization
        n_trials = self.config['model']['optimization']['n_iterations']
        sampler = TPESampler(seed=42)
        
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
        
        return study.best_params
    
    def _create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Create model instance"""
        params = {k: v for k, v in params.items() if v is not None}
        
        if model_name == 'random_forest':
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
            return XGBClassifier(**params, random_state=42, n_jobs=-1, eval_metric='logloss')
        elif model_name == 'logistic_regression':
            # For logistic regression, only use n_jobs if solver supports it
            solver = params.get('solver', 'lbfgs')
            if solver in ['liblinear', 'newton-cg', 'newton-cholesky']:
                # These solvers don't support n_jobs
                return LogisticRegression(**params, random_state=42)
            else:
                # Other solvers support n_jobs
                return LogisticRegression(**params, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _create_balanced_pipeline(self, model: Any) -> Any:
        """Create balanced pipeline with SMOTE"""
        if IMBLEARN_AVAILABLE:
            return ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('model', model)
            ])
        else:
            if hasattr(model, 'class_weight'):
                model.class_weight = 'balanced'
            return model
    
    def _evaluate_cross_validation(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Evaluate model using cross-validation"""
        cv_folds = self.config['model']['training']['cv_folds']
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv, scoring=metrics,
            n_jobs=-1, return_train_score=True
        )
        
        cv_scores = {}
        for metric in metrics:
            cv_scores[f'cv_{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            cv_scores[f'cv_{metric}_std'] = cv_results[f'test_{metric}'].std()
        
        return cv_scores
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate model on test set"""
        y_pred = model.predict(X_test)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'predict_proba'):
            y_proba = model.named_steps['model'].predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred
        
        # Calculate metrics
        scores = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba)
        }
        
        predictions = {'y_pred': y_pred, 'y_proba': y_proba}
        return scores, predictions
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance"""
        # Handle pipeline case
        if hasattr(model, 'named_steps'):
            model = model.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            importance = np.zeros(len(feature_names))
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        feature_importance['importance_normalized'] = (
            feature_importance['importance'] / feature_importance['importance'].sum()
        )
        
        return feature_importance
    
    def predict(self, model: Any, X: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions with a trained model"""
        predictions = model.predict(X)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['model'], 'predict_proba'):
            probabilities = model.named_steps['model'].predict_proba(X)[:, 1]
        else:
            probabilities = predictions
        
        # Risk segmentation with business-actionable thresholds
        # High risk: >35% probability (requires immediate intervention)
        # Medium risk: 15-35% (monitor and engage proactively)
        # Low risk: <15% (standard retention)
        high_risk = probabilities > 0.35
        medium_risk = (probabilities > 0.15) & (probabilities <= 0.35)
        low_risk = probabilities <= 0.15
        
        return {
            'predictions': predictions,
            'churn_probability': probabilities,
            'risk_segment': np.where(high_risk, 'high', 
                                   np.where(medium_risk, 'medium', 'low')),
            'high_risk_count': int(high_risk.sum()),
            'medium_risk_count': int(medium_risk.sum()),
            'low_risk_count': int(low_risk.sum())
        }
    
    def save_model(self, model_result: ModelResult, model_path: str, transformers_path: str):
        """Save model and transformers"""
        model_path = Path(model_path)
        transformers_path = Path(transformers_path)
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        transformers_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model_result.model, model_path)
        
        # Save transformers
        joblib.dump(self.transformers, transformers_path)
        
        # Save metadata
        metadata_path = model_path.with_suffix('.json')
        metadata = {
            'model_metadata': model_result.model_metadata,
            'best_params': model_result.best_params,
            'cv_scores': model_result.cv_scores,
            'test_scores': model_result.test_scores,
            'feature_importance': model_result.feature_importance.to_dict('records')
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Transformers saved to {transformers_path}")
    
    def load_model_and_transformers(self, model_path: str, transformers_path: str):
        """Load model and transformers"""
        model = joblib.load(model_path)
        self.transformers = joblib.load(transformers_path)
        self.logger.info(f"Model and transformers loaded successfully")
        return model