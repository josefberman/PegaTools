"""
Forecasting utilities for Pega Tools.

This module provides forecasting and prediction functionality including
basic forecasters and advanced Darts-based models.
"""

from .predictor import LinearPredictor, ExponentialPredictor
from .metrics import calculate_mape, calculate_rmse, calculate_mae, calculate_r_squared, evaluate_forecast
from .data import TimeSeriesData

# Import Darts forecasters with graceful fallback
try:
    from .darts_forecasters import (
        DartsAutoARIMA,
        DartsAutoETS,
        DartsAutoTheta,
        DartsAutoTBATS,
        DartsKalmanForecaster,
        DartsNaiveMean,
        DartsNaiveSeasonal,
        DartsNaiveDrift,
        DartsNaiveMovingAverage,
        DartsForecasterFactory,
        create_ensemble_forecast
    )
    
    __all__ = [
        "LinearPredictor",
        "ExponentialPredictor",
        "calculate_mape",
        "calculate_rmse",
        "calculate_mae",
        "calculate_r_squared",
        "evaluate_forecast",
        "TimeSeriesData",
        # Darts forecasters
        "DartsAutoARIMA",
        "DartsAutoETS",
        "DartsAutoTheta",
        "DartsAutoTBATS",
        "DartsKalmanForecaster",
        "DartsNaiveMean",
        "DartsNaiveSeasonal",
        "DartsNaiveDrift",
        "DartsNaiveMovingAverage",
        "DartsForecasterFactory",
        "create_ensemble_forecast"
    ]
    
except ImportError:
    # Darts not available, only export basic functionality
    __all__ = [
        "LinearPredictor",
        "ExponentialPredictor",
        "calculate_mape",
        "calculate_rmse",
        "calculate_mae",
        "calculate_r_squared",
        "evaluate_forecast",
        "TimeSeriesData"
    ] 