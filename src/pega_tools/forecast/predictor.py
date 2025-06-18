"""
Prediction models for forecasting.
"""

import numpy as np
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from .data import TimeSeriesData


class BasePredictor(ABC):
    """Base class for prediction models."""
    
    def __init__(self):
        self.is_fitted = False
        self.coefficients = None
    
    @abstractmethod
    def fit(self, data: TimeSeriesData) -> None:
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, periods: int) -> List[float]:
        """Make predictions for future periods."""
        pass
    
    def _check_fitted(self) -> None:
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")


class LinearPredictor(BasePredictor):
    """Linear trend prediction model."""
    
    def fit(self, data: TimeSeriesData) -> None:
        """
        Fit linear trend model to the data.
        
        Args:
            data: Time series data to fit
        """
        values = data.values
        x = np.arange(len(values))
        
        # Calculate linear regression coefficients
        n = len(values)
        sum_x = np.sum(x)
        sum_y = np.sum(values)
        sum_xy = np.sum(x * values)
        sum_x2 = np.sum(x * x)
        
        # y = a + bx
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        self.intercept = (sum_y - self.slope * sum_x) / n
        
        self.coefficients = (self.intercept, self.slope)
        self.is_fitted = True
        self._last_x = len(values) - 1
    
    def predict(self, periods: int) -> List[float]:
        """
        Predict future values using linear trend.
        
        Args:
            periods: Number of periods to predict
            
        Returns:
            List of predicted values
        """
        self._check_fitted()
        
        predictions = []
        for i in range(1, periods + 1):
            x = self._last_x + i
            prediction = self.intercept + self.slope * x
            predictions.append(prediction)
        
        return predictions


class ExponentialPredictor(BasePredictor):
    """Exponential smoothing prediction model."""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize exponential smoothing model.
        
        Args:
            alpha: Smoothing parameter (0 < alpha < 1)
        """
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.last_value = None
    
    def fit(self, data: TimeSeriesData) -> None:
        """
        Fit exponential smoothing model to the data.
        
        Args:
            data: Time series data to fit
        """
        values = data.values
        
        # Initialize with first value
        smoothed = [values[0]]
        
        # Apply exponential smoothing
        for i in range(1, len(values)):
            smoothed_value = self.alpha * values[i] + (1 - self.alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)
        
        self.last_value = smoothed[-1]
        self.is_fitted = True
    
    def predict(self, periods: int) -> List[float]:
        """
        Predict future values using exponential smoothing.
        
        Args:
            periods: Number of periods to predict
            
        Returns:
            List of predicted values
        """
        self._check_fitted()
        
        # For simple exponential smoothing, all future predictions 
        # are equal to the last smoothed value
        return [self.last_value] * periods 