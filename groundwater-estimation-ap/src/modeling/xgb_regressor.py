"""
Phase 2: Core AI Model (XGBoost Estimation)
Train and evaluate XGBoost model for groundwater level prediction.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from typing import Tuple, Dict, List


class GroundwaterEstimator:
    """XGBoost-based groundwater level estimator."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the estimator."""
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            tree_method='hist'
        )
        self.feature_importance_ = None
        self.training_history_ = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        validation_split: float = 0.1
    ) -> Dict:
        """
        Train the XGBoost model with validation.
        
        Args:
            X: Feature matrix
            y: Target variable (groundwater depth)
            test_size: Proportion of data for testing
            validation_split: Proportion of training data for validation
        
        Returns:
            Dictionary with training metrics
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Get feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mape = self.calculate_mape(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mape_threshold_met': mape < 0.05,
            'feature_importance': self.feature_importance_
        }
        
        self.training_history_ = metrics
        return metrics
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAPE as decimal (e.g., 0.05 for 5%)
        """
        return np.mean(np.abs((y_true - y_pred) / y_true))
    
    def calculate_error_rate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.05
    ) -> Dict:
        """
        Calculate percentage error and check against threshold.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Error threshold (default: 0.05 for 5%)
        
        Returns:
            Dictionary with error metrics and warning status
        """
        error_rate = self.calculate_mape(y_true, y_pred)
        
        return {
            'error_rate': error_rate,
            'exceeds_threshold': error_rate > threshold,
            'threshold': threshold,
            'warning': f"⚠️ Error rate {error_rate:.2%} exceeds threshold {threshold:.2%}" 
                      if error_rate > threshold else "✓ Error rate within acceptable range"
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted groundwater levels
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
    
    def get_feature_importance_top_n(self, n: int = 5) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
        
        Returns:
            DataFrame with top features and importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Model has not been trained yet")
        return self.feature_importance_.head(n)
