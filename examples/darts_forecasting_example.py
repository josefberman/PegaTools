#!/usr/bin/env python3
"""
Example demonstrating advanced forecasting capabilities using Darts library.

This script shows how to use the various Darts-based forecasters available
in pega_tools.forecast for sophisticated time series forecasting.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pega_tools.forecast import (
    TimeSeriesData,
    DartsForecasterFactory,
    create_ensemble_forecast
)

# Try importing specific forecasters
try:
    from pega_tools.forecast import (
        DartsAutoARIMA,
        DartsAutoETS,
        DartsNaiveMean,
        DartsNaiveDrift,
        DartsNaiveMovingAverage
    )
    DARTS_AVAILABLE = True
except ImportError:
    print("Darts library not available. Install with: pip install darts>=0.26.0")
    DARTS_AVAILABLE = False


def create_sample_data():
    """Create sample time series data with trend and seasonality."""
    print("Creating sample time series data...")
    
    # Generate 100 days of data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    
    # Create synthetic data: trend + seasonal + noise
    trend = [10 + 0.1 * i for i in range(100)]
    seasonal = [3 * np.sin(2 * np.pi * i / 7) for i in range(100)]  # Weekly seasonality
    noise = np.random.normal(0, 0.5, 100)
    
    values = [t + s + n for t, s, n in zip(trend, seasonal, noise)]
    
    return TimeSeriesData(values, dates, frequency='daily')


def demonstrate_individual_forecasters(data):
    """Demonstrate individual forecasting models."""
    print("\n=== Individual Forecasters ===")
    
    # Get available models
    factory = DartsForecasterFactory()
    available_models = factory.get_available_models()
    print(f"Available models: {available_models}")
    
    # Test a few key models
    models_to_test = ['naive_mean', 'naive_drift', 'naive_moving_average']
    
    if len(data.values) >= 20:  # Need enough data for complex models
        models_to_test.extend(['auto_arima', 'auto_ets'])
    
    results = {}
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting {model_name}...")
            
            # Create and fit forecaster
            forecaster = factory.create_forecaster(model_name)
            forecaster.fit(data)
            
            # Make predictions
            predictions = forecaster.predict(periods=10)
            
            # Get model info
            info = forecaster.get_model_info()
            
            results[model_name] = {
                'predictions': predictions,
                'info': info
            }
            
            print(f"✓ {model_name} - Predictions: {predictions[:3]}... (showing first 3)")
            print(f"  Training samples: {info['training_samples']}")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")
    
    return results


def demonstrate_confidence_intervals(data):
    """Demonstrate confidence interval predictions."""
    print("\n=== Confidence Intervals ===")
    
    try:
        # Use a simple model for demonstration
        forecaster = DartsNaiveMean()
        forecaster.fit(data)
        
        result = forecaster.predict_with_confidence(periods=5, num_samples=100)
        
        print("Predictions with confidence intervals:")
        for i in range(5):
            pred = result['predictions'][i]
            lower = result['lower_bound'][i]
            upper = result['upper_bound'][i]
            print(f"  Period {i+1}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")
            
    except Exception as e:
        print(f"Confidence interval demo failed: {str(e)}")


def demonstrate_ensemble_forecasting(data):
    """Demonstrate ensemble forecasting with multiple models."""
    print("\n=== Ensemble Forecasting ===")
    
    try:
        # Define models for ensemble
        models = ['naive_mean', 'naive_drift', 'naive_moving_average']
        
        # Create ensemble with equal weights
        print("Creating ensemble with equal weights...")
        result = create_ensemble_forecast(data, models, periods=7)
        
        print("Ensemble Results:")
        print(f"  Ensemble predictions: {[f'{p:.2f}' for p in result['ensemble_predictions']]}")
        print(f"  Model weights: {result['model_weights']}")
        
        # Show individual model contributions
        print("Individual model predictions:")
        for model, preds in result['individual_predictions'].items():
            print(f"  {model}: {[f'{p:.2f}' for p in preds]}")
        
        # Create ensemble with custom weights
        print("\nCreating ensemble with custom weights...")
        custom_weights = [0.5, 0.3, 0.2]  # Favor naive_mean
        result_weighted = create_ensemble_forecast(
            data, models, periods=7, weights=custom_weights
        )
        
        print(f"Weighted ensemble predictions: {[f'{p:.2f}' for p in result_weighted['ensemble_predictions']]}")
        
    except Exception as e:
        print(f"Ensemble forecasting demo failed: {str(e)}")


def demonstrate_advanced_models(data):
    """Demonstrate advanced forecasting models."""
    print("\n=== Advanced Models ===")
    
    if len(data.values) < 30:
        print("Skipping advanced models - need more data points")
        return
    
    advanced_models = {
        'auto_arima': {'seasonal': True, 'stepwise': True},
        'auto_ets': {'seasonal_periods': 7},  # Weekly seasonality
        'auto_theta': {'seasonal_periods': 7}
    }
    
    for model_name, params in advanced_models.items():
        try:
            print(f"\nTesting {model_name} with parameters: {params}")
            
            forecaster = DartsForecasterFactory.create_forecaster(model_name, **params)
            forecaster.fit(data)
            
            predictions = forecaster.predict(periods=5)
            print(f"✓ {model_name} predictions: {[f'{p:.2f}' for p in predictions]}")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")


def plot_results(data, predictions_dict):
    """Plot the original data and forecasts."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        plt.figure(figsize=(12, 8))
        
        # Plot original data
        plt.plot(data.timestamps, data.values, 'b-', label='Historical Data', linewidth=2)
        
        # Generate future timestamps for predictions
        last_date = data.timestamps[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(10)]
        
        # Plot predictions from different models
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (model_name, results) in enumerate(predictions_dict.items()):
            if 'predictions' in results:
                color = colors[i % len(colors)]
                plt.plot(future_dates, results['predictions'], 
                        color=color, linestyle='--', marker='o', 
                        label=f'{model_name} forecast')
        
        plt.title('Time Series Forecasting Comparison')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('darts_forecasting_example.png', dpi=300, bbox_inches='tight')
        print("\n📊 Plot saved as 'darts_forecasting_example.png'")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")
    except Exception as e:
        print(f"\nPlot generation failed: {str(e)}")


def main():
    """Main example function."""
    print("🚀 Darts Forecasting Example")
    print("=" * 50)
    
    if not DARTS_AVAILABLE:
        print("❌ Darts library is not available.")
        print("Install with: pip install darts>=0.26.0")
        return
    
    # Create sample data
    data = create_sample_data()
    print(f"Created time series with {len(data.values)} data points")
    print(f"Date range: {data.timestamps[0]} to {data.timestamps[-1]}")
    print(f"Value range: {min(data.values):.2f} to {max(data.values):.2f}")
    
    # Demonstrate different forecasting capabilities
    individual_results = demonstrate_individual_forecasters(data)
    demonstrate_confidence_intervals(data)
    demonstrate_ensemble_forecasting(data)
    demonstrate_advanced_models(data)
    
    # Create visualization
    if individual_results:
        plot_results(data, individual_results)
    
    print("\n✅ Darts forecasting demonstration completed!")
    print("\nKey takeaways:")
    print("- Multiple forecasting models available through DartsForecasterFactory")
    print("- Ensemble forecasting combines multiple models for better accuracy")
    print("- Confidence intervals provide uncertainty estimates")
    print("- Advanced models like AutoARIMA automatically tune parameters")


if __name__ == "__main__":
    main() 