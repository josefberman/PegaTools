"""
Tests for Darts-based forecasters.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from pega_tools.forecast.data import TimeSeriesData
from pega_tools.utils import PegaException

# Try to import Darts forecasters, skip tests if not available
try:
    from pega_tools.forecast.darts_forecasters import (
        DartsAutoARIMA,
        DartsAutoETS,
        DartsAutoTheta,
        DartsNaiveMean,
        DartsNaiveDrift,
        DartsNaiveMovingAverage,
        DartsForecasterFactory,
        create_ensemble_forecast,
        DARTS_AVAILABLE
    )
    DARTS_TESTS_ENABLED = DARTS_AVAILABLE
except ImportError:
    DARTS_TESTS_ENABLED = False


@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    # Create a simple trend + seasonal pattern
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    values = [10 + 0.5 * i + 2 * np.sin(i / 7 * 2 * np.pi) for i in range(30)]
    return TimeSeriesData(values, dates, frequency='daily')


@pytest.mark.skipif(not DARTS_TESTS_ENABLED, reason="Darts library not available")
class TestDartsForecasters:
    """Test cases for Darts-based forecasters."""
    
    def test_auto_arima_basic(self, sample_time_series):
        """Test basic AutoARIMA functionality."""
        forecaster = DartsAutoARIMA(seasonal=False)
        
        # Test fitting
        forecaster.fit(sample_time_series)
        assert forecaster.is_fitted
        
        # Test prediction
        predictions = forecaster.predict(5)
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_auto_ets_basic(self, sample_time_series):
        """Test basic AutoETS functionality."""
        forecaster = DartsAutoETS()
        
        forecaster.fit(sample_time_series)
        assert forecaster.is_fitted
        
        predictions = forecaster.predict(3)
        assert len(predictions) == 3
    
    def test_naive_mean_basic(self, sample_time_series):
        """Test basic NaiveMean functionality."""
        forecaster = DartsNaiveMean()
        
        forecaster.fit(sample_time_series)
        assert forecaster.is_fitted
        
        predictions = forecaster.predict(5)
        assert len(predictions) == 5
        
        # All predictions should be the same (mean of training data)
        expected_mean = np.mean(sample_time_series.values)
        assert all(abs(p - expected_mean) < 1e-6 for p in predictions)
    
    def test_naive_drift_basic(self, sample_time_series):
        """Test basic NaiveDrift functionality."""
        forecaster = DartsNaiveDrift()
        
        forecaster.fit(sample_time_series)
        assert forecaster.is_fitted
        
        predictions = forecaster.predict(3)
        assert len(predictions) == 3
    
    def test_naive_moving_average_basic(self, sample_time_series):
        """Test basic NaiveMovingAverage functionality."""
        forecaster = DartsNaiveMovingAverage(window=5)
        
        forecaster.fit(sample_time_series)
        assert forecaster.is_fitted
        
        predictions = forecaster.predict(3)
        assert len(predictions) == 3
    
    def test_forecaster_not_fitted_error(self):
        """Test error when predicting without fitting."""
        forecaster = DartsNaiveMean()
        
        with pytest.raises(PegaException) as exc_info:
            forecaster.predict(5)
        
        assert "must be fitted" in str(exc_info.value)
    
    def test_model_info(self, sample_time_series):
        """Test model info retrieval."""
        forecaster = DartsNaiveMean()
        
        # Before fitting
        info = forecaster.get_model_info()
        assert info["status"] == "not_fitted"
        
        # After fitting
        forecaster.fit(sample_time_series)
        info = forecaster.get_model_info()
        assert info["is_fitted"] is True
        assert info["model_type"] == "DartsNaiveMean"
        assert info["training_samples"] == len(sample_time_series.values)
    
    def test_predict_with_confidence(self, sample_time_series):
        """Test confidence interval predictions."""
        forecaster = DartsNaiveMean()
        forecaster.fit(sample_time_series)
        
        result = forecaster.predict_with_confidence(3)
        
        assert "predictions" in result
        assert "lower_bound" in result
        assert "upper_bound" in result
        assert len(result["predictions"]) == 3
        assert len(result["lower_bound"]) == 3
        assert len(result["upper_bound"]) == 3


@pytest.mark.skipif(not DARTS_TESTS_ENABLED, reason="Darts library not available")
class TestDartsForecasterFactory:
    """Test cases for the forecaster factory."""
    
    def test_get_available_models(self):
        """Test getting list of available models."""
        models = DartsForecasterFactory.get_available_models()
        
        expected_models = [
            'auto_arima', 'auto_ets', 'auto_theta', 'auto_tbats',
            'kalman', 'naive_mean', 'naive_seasonal', 'naive_drift',
            'naive_moving_average'
        ]
        
        for model in expected_models:
            assert model in models
    
    def test_create_forecaster_naive_mean(self):
        """Test creating NaiveMean forecaster."""
        forecaster = DartsForecasterFactory.create_forecaster('naive_mean')
        assert isinstance(forecaster, DartsNaiveMean)
    
    def test_create_forecaster_with_params(self):
        """Test creating forecaster with parameters."""
        forecaster = DartsForecasterFactory.create_forecaster(
            'naive_moving_average', window=7
        )
        assert isinstance(forecaster, DartsNaiveMovingAverage)
    
    def test_create_forecaster_invalid_type(self):
        """Test error for invalid forecaster type."""
        with pytest.raises(PegaException) as exc_info:
            DartsForecasterFactory.create_forecaster('invalid_model')
        
        assert "Unknown model type" in str(exc_info.value)


@pytest.mark.skipif(not DARTS_TESTS_ENABLED, reason="Darts library not available")
class TestEnsembleForecasting:
    """Test cases for ensemble forecasting."""
    
    def test_ensemble_forecast_basic(self, sample_time_series):
        """Test basic ensemble forecasting."""
        models = ['naive_mean', 'naive_drift']
        
        result = create_ensemble_forecast(
            sample_time_series, models, periods=3
        )
        
        assert 'ensemble_predictions' in result
        assert 'individual_predictions' in result
        assert 'model_weights' in result
        assert 'model_info' in result
        
        assert len(result['ensemble_predictions']) == 3
        assert len(result['individual_predictions']) == 2
        assert len(result['model_weights']) == 2
    
    def test_ensemble_forecast_with_weights(self, sample_time_series):
        """Test ensemble forecasting with custom weights."""
        models = ['naive_mean', 'naive_drift']
        weights = [0.7, 0.3]
        
        result = create_ensemble_forecast(
            sample_time_series, models, periods=3, weights=weights
        )
        
        assert result['model_weights'] == weights
    
    def test_ensemble_forecast_invalid_weights(self, sample_time_series):
        """Test error for invalid weights."""
        models = ['naive_mean', 'naive_drift']
        
        # Wrong number of weights
        with pytest.raises(PegaException) as exc_info:
            create_ensemble_forecast(
                sample_time_series, models, periods=3, weights=[0.5]
            )
        assert "Number of weights must match" in str(exc_info.value)
        
        # Weights don't sum to 1
        with pytest.raises(PegaException) as exc_info:
            create_ensemble_forecast(
                sample_time_series, models, periods=3, weights=[0.6, 0.6]
            )
        assert "Weights must sum to 1.0" in str(exc_info.value)


@pytest.mark.skipif(DARTS_TESTS_ENABLED, reason="Testing fallback when Darts not available")
class TestDartsNotAvailable:
    """Test behavior when Darts is not available."""
    
    def test_import_fallback(self):
        """Test that forecast package still works without Darts."""
        from pega_tools.forecast import LinearPredictor, TimeSeriesData
        
        # Basic forecasters should still be available
        assert LinearPredictor is not None
        assert TimeSeriesData is not None 