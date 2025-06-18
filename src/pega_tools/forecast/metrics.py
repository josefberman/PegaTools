"""
Forecast accuracy metrics for model evaluation.
"""

import numpy as np
from typing import List, Union


def calculate_mape(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE value as percentage
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have same length")
    
    # Avoid division by zero
    non_zero_mask = actual != 0
    if not np.any(non_zero_mask):
        raise ValueError("Cannot calculate MAPE when all actual values are zero")
    
    mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / 
                         actual[non_zero_mask])) * 100
    return mape


def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have same length")
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_mae(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have same length")
    
    mae = np.mean(np.abs(actual - predicted))
    return mae


def calculate_r_squared(actual: List[float], predicted: List[float]) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        R-squared value
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have same length")
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def evaluate_forecast(actual: List[float], predicted: List[float]) -> dict:
    """
    Calculate multiple forecast accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with various accuracy metrics
    """
    metrics = {
        'mape': calculate_mape(actual, predicted),
        'rmse': calculate_rmse(actual, predicted),
        'mae': calculate_mae(actual, predicted),
        'r_squared': calculate_r_squared(actual, predicted)
    }
    
    return metrics 