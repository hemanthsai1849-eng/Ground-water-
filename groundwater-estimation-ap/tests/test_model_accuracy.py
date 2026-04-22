"""
Unit tests for model accuracy against the 5% error threshold.
"""

import pytest
import numpy as np
import pandas as pd
from src.modeling.xgb_regressor import GroundwaterEstimator


class TestModelAccuracy:
    """Test suite for model accuracy validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = pd.DataFrame({
            'rainfall': np.random.randn(100),
            'soil_permeability': np.random.randn(100),
            'elevation': np.random.randn(100),
            'distance_to_nearest_piezometer': np.random.randn(100)
        })
        y = pd.Series(
            50 + 10 * X['rainfall'] + 5 * X['soil_permeability'] - 3 * X['elevation'] + np.random.randn(100)
        )
        return X, y
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        estimator = GroundwaterEstimator()
        assert estimator.model is not None
        assert estimator.feature_importance_ is None
    
    def test_mape_calculation(self):
        """Test MAPE calculation."""
        estimator = GroundwaterEstimator()
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 320])
        
        mape = estimator.calculate_mape(y_true, y_pred)
        expected_mape = np.mean([0.1, 0.05, 0.067]) 
        
        assert abs(mape - expected_mape) < 0.01
    
    def test_error_rate_below_threshold(self):
        """Test error rate detection when below threshold."""
        estimator = GroundwaterEstimator()
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([102, 198, 305, 398, 502])  # ~2% error
        
        result = estimator.calculate_error_rate(y_true, y_pred, threshold=0.05)
        
        assert result['error_rate'] < 0.05
        assert not result['exceeds_threshold']
    
    def test_error_rate_above_threshold(self):
        """Test error rate detection when above threshold."""
        estimator = GroundwaterEstimator()
        y_true = np.array([100, 200, 300])
        y_pred = np.array([120, 240, 360])  # 20% error
        
        result = estimator.calculate_error_rate(y_true, y_pred, threshold=0.05)
        
        assert result['error_rate'] > 0.05
        assert result['exceeds_threshold']
    
    def test_model_training(self, sample_data):
        """Test model training and metric collection."""
        X, y = sample_data
        estimator = GroundwaterEstimator()
        
        metrics = estimator.train(X, y, test_size=0.2)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'mape_threshold_met' in metrics
        assert metrics['test_samples'] > 0
    
    def test_feature_importance_extraction(self, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        estimator = GroundwaterEstimator()
        estimator.train(X, y)
        
        top_features = estimator.get_feature_importance_top_n(n=2)
        
        assert len(top_features) == 2
        assert 'feature' in top_features.columns
        assert 'importance' in top_features.columns
        assert top_features['importance'].iloc[0] > top_features['importance'].iloc[1]
    
    def test_prediction(self, sample_data):
        """Test prediction on new data."""
        X, y = sample_data
        estimator = GroundwaterEstimator()
        estimator.train(X, y)
        
        predictions = estimator.predict(X.iloc[:5])
        
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestErrorThreshold:
    """Test that model meets the 5% error threshold requirement."""
    
    def test_synthetic_data_accuracy(self):
        """Test model accuracy on synthetic data with known relationship."""
        np.random.seed(42)
        
        # Create synthetic data with known relationship
        n_samples = 200
        X = pd.DataFrame({
            'rainfall': np.random.uniform(500, 1500, n_samples),
            'soil_permeability': np.random.uniform(1, 10, n_samples),
            'elevation': np.random.uniform(100, 1000, n_samples),
            'distance_to_nearest_piezometer': np.random.uniform(0, 50, n_samples)
        })
        
        # Known relationship: water_level = 50 + 0.01*rainfall + 2*soil_permeability - 0.02*elevation
        y = 50 + 0.01 * X['rainfall'] + 2 * X['soil_permeability'] - 0.02 * X['elevation']
        y = y + np.random.normal(0, 1, n_samples)  # Add small noise
        
        estimator = GroundwaterEstimator()
        metrics = estimator.train(X, y, test_size=0.3)
        
        # Check threshold
        assert metrics['mape'] < 0.05, f"MAPE {metrics['mape']:.4f} exceeds 5% threshold"
        assert metrics['mape_threshold_met'] is True
