"""
Advanced forecasting models using the Darts library.

This module provides a collection of forecasting models built on top of the Darts library,
offering state-of-the-art time series forecasting capabilities with automatic parameter tuning.
"""

from typing import List, Optional, Union, Any, Dict
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from darts import TimeSeries
    from darts.models import (
        AutoARIMA,
        AutoETS,
        AutoTheta,
        AutoTBATS,
        KalmanForecaster,
        NaiveMean,
        NaiveSeasonal,
        NaiveDrift,
        NaiveMovingAverage
    )
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None

from .data import TimeSeriesData
from ..utils import PegaException


class DartsForecasterBase:
    """Base class for Darts-based forecasters."""
    
    def __init__(self):
        """Initialize the base forecaster."""
        if not DARTS_AVAILABLE:
            raise PegaException(
                "Darts library is not installed. Install with: pip install darts>=0.26.0"
            )
        
        self.model = None
        self.is_fitted = False
        self.training_data = None
        self.darts_series = None
    
    def _check_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise PegaException("Model must be fitted before making predictions")
    
    def _convert_to_darts_series(self, data: TimeSeriesData) -> TimeSeries:
        """Convert TimeSeriesData to Darts TimeSeries."""
        if len(data.timestamps) != len(data.values):
            raise PegaException("Timestamps and values must have the same length")
        
        # Create pandas DataFrame
        df = pd.DataFrame({
            'timestamp': data.timestamps,
            'value': data.values
        })
        df.set_index('timestamp', inplace=True)
        
        # Convert to Darts TimeSeries
        return TimeSeries.from_dataframe(df)
    
    def fit(self, data: TimeSeriesData) -> None:
        """
        Fit the forecasting model to the data.
        
        Args:
            data: Time series data to fit
            
        Raises:
            PegaException: If fitting fails
        """
        try:
            self.training_data = data
            self.darts_series = self._convert_to_darts_series(data)
            self.model.fit(self.darts_series)
            self.is_fitted = True
        except Exception as e:
            raise PegaException(f"Failed to fit {self.__class__.__name__}: {str(e)}")
    
    def predict(self, periods: int) -> List[float]:
        """
        Make predictions for future periods.
        
        Args:
            periods: Number of periods to predict
            
        Returns:
            List of predicted values
            
        Raises:
            PegaException: If prediction fails
        """
        self._check_fitted()
        
        try:
            forecast = self.model.predict(n=periods)
            return forecast.values().flatten().tolist()
        except Exception as e:
            raise PegaException(f"Failed to predict with {self.__class__.__name__}: {str(e)}")
    
    def predict_with_confidence(self, periods: int, num_samples: int = 100) -> Dict[str, List[float]]:
        """
        Make predictions with confidence intervals.
        
        Args:
            periods: Number of periods to predict
            num_samples: Number of samples for confidence estimation
            
        Returns:
            Dictionary with predictions, lower_bound, and upper_bound
        """
        self._check_fitted()
        
        try:
            if hasattr(self.model, 'predict') and hasattr(self.model, 'sample'):
                # For models that support probabilistic forecasting
                forecast = self.model.predict(n=periods, num_samples=num_samples)
                
                # Calculate confidence intervals
                quantiles = forecast.quantile_timeseries([0.1, 0.5, 0.9])
                
                return {
                    'predictions': quantiles[0.5].values().flatten().tolist(),
                    'lower_bound': quantiles[0.1].values().flatten().tolist(),
                    'upper_bound': quantiles[0.9].values().flatten().tolist()
                }
            else:
                # For deterministic models, return predictions without bounds
                predictions = self.predict(periods)
                return {
                    'predictions': predictions,
                    'lower_bound': predictions,  # Same as predictions
                    'upper_bound': predictions   # Same as predictions
                }
        except Exception as e:
            raise PegaException(f"Failed to predict with confidence for {self.__class__.__name__}: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        info = {
            "model_type": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "training_samples": len(self.training_data.values) if self.training_data else 0
        }
        
        # Add model-specific parameters if available
        if hasattr(self.model, 'model_params'):
            info["parameters"] = self.model.model_params
        
        return info


class DartsAutoARIMA(DartsForecasterBase):
    """AutoARIMA forecaster using Darts."""
    
    def __init__(self, seasonal: bool = True, stepwise: bool = True, 
                 suppress_warnings: bool = True, **kwargs):
        """
        Initialize AutoARIMA forecaster.
        
        Args:
            seasonal: Whether to include seasonal components
            stepwise: Whether to use stepwise algorithm for parameter selection
            suppress_warnings: Whether to suppress ARIMA warnings
            **kwargs: Additional parameters for AutoARIMA
        """
        super().__init__()
        self.model = AutoARIMA(
            seasonal=seasonal,
            stepwise=stepwise,
            suppress_warnings=suppress_warnings,
            **kwargs
        )


class DartsAutoETS(DartsForecasterBase):
    """AutoETS (Exponential Smoothing) forecaster using Darts."""
    
    def __init__(self, seasonal_periods: Optional[int] = None, **kwargs):
        """
        Initialize AutoETS forecaster.
        
        Args:
            seasonal_periods: Number of periods in a seasonal cycle
            **kwargs: Additional parameters for AutoETS
        """
        super().__init__()
        self.model = AutoETS(seasonal_periods=seasonal_periods, **kwargs)


class DartsAutoTheta(DartsForecasterBase):
    """AutoTheta forecaster using Darts."""
    
    def __init__(self, seasonal_periods: Optional[int] = None, **kwargs):
        """
        Initialize AutoTheta forecaster.
        
        Args:
            seasonal_periods: Number of periods in a seasonal cycle
            **kwargs: Additional parameters for AutoTheta
        """
        super().__init__()
        self.model = AutoTheta(seasonal_periods=seasonal_periods, **kwargs)


class DartsAutoTBATS(DartsForecasterBase):
    """AutoTBATS forecaster using Darts."""
    
    def __init__(self, seasonal_periods: Optional[List[int]] = None, **kwargs):
        """
        Initialize AutoTBATS forecaster.
        
        Args:
            seasonal_periods: List of seasonal periods for multiple seasonalities
            **kwargs: Additional parameters for AutoTBATS
        """
        super().__init__()
        self.model = AutoTBATS(seasonal_periods=seasonal_periods, **kwargs)


class DartsKalmanForecaster(DartsForecasterBase):
    """Kalman Filter forecaster using Darts."""
    
    def __init__(self, dim_x: int = 2, **kwargs):
        """
        Initialize Kalman forecaster.
        
        Args:
            dim_x: Dimension of the state vector
            **kwargs: Additional parameters for KalmanForecaster
        """
        super().__init__()
        self.model = KalmanForecaster(dim_x=dim_x, **kwargs)


class DartsNaiveMean(DartsForecasterBase):
    """Naive Mean forecaster using Darts."""
    
    def __init__(self, **kwargs):
        """
        Initialize Naive Mean forecaster.
        
        Args:
            **kwargs: Additional parameters for NaiveMean
        """
        super().__init__()
        self.model = NaiveMean(**kwargs)


class DartsNaiveSeasonal(DartsForecasterBase):
    """Naive Seasonal forecaster using Darts."""
    
    def __init__(self, K: int = 1, **kwargs):
        """
        Initialize Naive Seasonal forecaster.
        
        Args:
            K: The number of last seasons to use for prediction
            **kwargs: Additional parameters for NaiveSeasonal
        """
        super().__init__()
        self.model = NaiveSeasonal(K=K, **kwargs)


class DartsNaiveDrift(DartsForecasterBase):
    """Naive Drift forecaster using Darts."""
    
    def __init__(self, **kwargs):
        """
        Initialize Naive Drift forecaster.
        
        Args:
            **kwargs: Additional parameters for NaiveDrift
        """
        super().__init__()
        self.model = NaiveDrift(**kwargs)


class DartsNaiveMovingAverage(DartsForecasterBase):
    """Naive Moving Average forecaster using Darts."""
    
    def __init__(self, window: int = 3, **kwargs):
        """
        Initialize Naive Moving Average forecaster.
        
        Args:
            window: Size of the moving average window
            **kwargs: Additional parameters for NaiveMovingAverage
        """
        super().__init__()
        self.model = NaiveMovingAverage(window=window, **kwargs)


class DartsForecasterFactory:
    """Factory class to create Darts forecasters."""
    
    AVAILABLE_MODELS = {
        'auto_arima': DartsAutoARIMA,
        'auto_ets': DartsAutoETS,
        'auto_theta': DartsAutoTheta,
        'auto_tbats': DartsAutoTBATS,
        'kalman': DartsKalmanForecaster,
        'naive_mean': DartsNaiveMean,
        'naive_seasonal': DartsNaiveSeasonal,
        'naive_drift': DartsNaiveDrift,
        'naive_moving_average': DartsNaiveMovingAverage
    }
    
    @classmethod
    def create_forecaster(cls, model_type: str, **kwargs) -> DartsForecasterBase:
        """
        Create a forecaster instance.
        
        Args:
            model_type: Type of forecaster to create
            **kwargs: Parameters for the forecaster
            
        Returns:
            Forecaster instance
            
        Raises:
            PegaException: If model type is not supported
        """
        if model_type not in cls.AVAILABLE_MODELS:
            available = ', '.join(cls.AVAILABLE_MODELS.keys())
            raise PegaException(f"Unknown model type '{model_type}'. Available: {available}")
        
        forecaster_class = cls.AVAILABLE_MODELS[model_type]
        return forecaster_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available forecaster models."""
        return list(cls.AVAILABLE_MODELS.keys())


def create_ensemble_forecast(data: TimeSeriesData, 
                           models: List[str], 
                           periods: int,
                           weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Create ensemble forecasts using multiple models.
    
    Args:
        data: Time series data
        models: List of model types to include in ensemble
        periods: Number of periods to forecast
        weights: Optional weights for each model (must sum to 1.0)
        
    Returns:
        Dictionary with ensemble predictions and individual model results
    """
    if not DARTS_AVAILABLE:
        raise PegaException("Darts library is required for ensemble forecasting")
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    if len(weights) != len(models):
        raise PegaException("Number of weights must match number of models")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise PegaException("Weights must sum to 1.0")
    
    factory = DartsForecasterFactory()
    forecasters = []
    predictions = []
    model_info = []
    
    # Fit all models and make predictions
    for model_type in models:
        try:
            forecaster = factory.create_forecaster(model_type)
            forecaster.fit(data)
            pred = forecaster.predict(periods)
            
            forecasters.append(forecaster)
            predictions.append(pred)
            model_info.append({
                'model_type': model_type,
                'success': True,
                'info': forecaster.get_model_info()
            })
        except Exception as e:
            model_info.append({
                'model_type': model_type,
                'success': False,
                'error': str(e)
            })
    
    if not predictions:
        raise PegaException("No models produced valid predictions")
    
    # Calculate weighted ensemble predictions
    ensemble_pred = np.zeros(periods)
    total_weight = 0
    
    for i, pred in enumerate(predictions):
        if i < len(weights):
            ensemble_pred += np.array(pred) * weights[i]
            total_weight += weights[i]
    
    # Normalize by actual total weight (in case some models failed)
    if total_weight > 0:
        ensemble_pred /= total_weight
    
    return {
        'ensemble_predictions': ensemble_pred.tolist(),
        'individual_predictions': {
            models[i]: pred for i, pred in enumerate(predictions)
        },
        'model_weights': weights[:len(predictions)],
        'model_info': model_info
    } 