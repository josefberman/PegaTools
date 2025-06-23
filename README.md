# Pega Tools

A comprehensive collection of specialized tools and utilities for data analysis, network monitoring, geographic calculations, and forecasting. While originally designed with Pega systems in mind, these tools are versatile and can be used for various data processing and analysis tasks.

## 🚀 Features

### 📍 Geographic Tools
- **Coordinate handling** with validation and conversion
- **Distance calculations** using Haversine formula
- **Geocoding services** for address-to-coordinate conversion
- **Multiple coordinate formats** (decimal, DMS, DM)
- **Geographic visualization** with scatter plots, 2D histograms, and gradient analysis
- **Temporal and spatial gradient visualization** for time-series geographic data
- **Animated geographic visualizations** for dynamic data presentation

### 📈 Forecasting Tools
- **Time series data management** with statistics and trend analysis
- **Basic prediction models** (Linear, Exponential)
- **Advanced Darts-based forecasters** (AutoARIMA, AutoETS, AutoTheta, AutoTBATS, Kalman)
- **Naive forecasting methods** (Mean, Seasonal, Drift, Moving Average)
- **Ensemble forecasting** with multiple model combinations
- **Confidence intervals** and probabilistic forecasting
- **Forecast accuracy metrics** (MAPE, RMSE, MAE, R²)
- **Model factory pattern** for easy forecaster creation

### 🔧 JSON Processing Tools
- **Advanced JSON manipulation** with file I/O
- **Schema validation** with custom rules
- **Data transformation** and normalization
- **Utilities** for flattening, merging, and comparison

### 🌐 Network Analysis Tools
- **PCAP file parsing** and analysis
- **Network traffic monitoring** and statistics
- **Packet filtering** by various criteria
- **Security analysis** and anomaly detection

## 📦 Installation

### From PyPI (when published)
```bash
pip install pega-tools
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/pega-tools.git
cd pega-tools

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install with advanced forecasting capabilities
pip install -e ".[forecasting]"

# Install with visualization support
pip install -e ".[viz]"
```

### Optional Dependencies

The package supports different installation levels:

- **Basic**: Core functionality only
- **Forecasting**: Includes Darts library for advanced forecasting models
- **Visualization**: Includes matplotlib for plotting and visualization

```bash
# Install specific optional dependencies
pip install pega-tools[forecasting]  # Advanced forecasting
pip install pega-tools[viz]          # Visualization support
pip install pega-tools[forecasting,viz]  # Both
```

## 🛠 Usage Examples

### Geographic Tools

```python
from pega_tools.geographic import Coordinate, distance_between, Geocoder, format_coordinates

# Create coordinates and calculate distance
ny = Coordinate(40.7128, -74.0060)  # New York
la = Coordinate(34.0522, -118.2437)  # Los Angeles
distance = distance_between(ny, la, unit="miles")
print(f"Distance: {distance:.1f} miles")

# Format coordinates in different styles
print(format_coordinates(ny, "dms"))  # Degrees, minutes, seconds
print(format_coordinates(ny, "decimal"))  # Decimal degrees

# Geocoding
geocoder = Geocoder()
location = geocoder.geocode("Times Square, New York")
if location:
    print(f"Times Square: {location}")
```

#### Geographic Visualization
```python
from pega_tools.geographic import (
    plot_geographic_scatter,
    plot_geographic_histogram_2d,
    plot_temporal_gradient_histogram,
    plot_spatial_gradient_histogram,
    create_geographic_animation
)
import matplotlib.pyplot as plt

# Create sample geographic data
coordinates = [
    Coordinate(40.7128, -74.0060),  # New York
    Coordinate(34.0522, -118.2437),  # Los Angeles
    Coordinate(41.8781, -87.6298),  # Chicago
    Coordinate(29.7604, -95.3698),  # Houston
    Coordinate(39.9526, -75.1652),  # Philadelphia
]

values = [100, 85, 92, 78, 88]  # Associated values
timestamps = [
    datetime(2023, 1, 1, 10, 0),
    datetime(2023, 1, 1, 11, 0),
    datetime(2023, 1, 1, 12, 0),
    datetime(2023, 1, 1, 13, 0),
    datetime(2023, 1, 1, 14, 0),
]

# Scatter plot with values
fig = plot_geographic_scatter(
    coordinates=coordinates,
    values=values,
    title="City Activity Levels",
    marker_size=values,
    color_map="plasma",
    background_map=True
)
plt.savefig("city_activity.png")
plt.close(fig)

# 2D histogram
fig = plot_geographic_histogram_2d(
    coordinates=coordinates,
    title="Geographic Density",
    bins=20,
    color_map="viridis",
    log_scale=True
)
plt.savefig("geographic_density.png")
plt.close(fig)

# Temporal gradient analysis
fig = plot_temporal_gradient_histogram(
    coordinates=coordinates,
    timestamps=timestamps,
    title="Temporal Geographic Analysis",
    color_map="plasma"
)
plt.savefig("temporal_analysis.png")
plt.close(fig)

# Spatial gradient analysis
fig = plot_spatial_gradient_histogram(
    coordinates=coordinates,
    values=values,
    title="Spatial Value Distribution",
    gradient_method="density",
    color_map="coolwarm"
)
plt.savefig("spatial_gradient.png")
plt.close(fig)

# Create animation (requires matplotlib.animation)
try:
    output_file = create_geographic_animation(
        coordinates=coordinates,
        timestamps=timestamps,
        values=values,
        output_file="geographic_animation.gif"
    )
    print(f"Animation saved as: {output_file}")
except ImportError:
    print("Animation requires matplotlib.animation support")

### Forecasting Tools

#### Basic Forecasting
```python
from pega_tools.forecast import LinearPredictor, TimeSeriesData, calculate_mape

# Create time series data
values = [10, 12, 14, 16, 18, 20, 22, 24]
ts_data = TimeSeriesData(values, frequency='daily')

# Fit a linear predictor
predictor = LinearPredictor()
predictor.fit(ts_data)

# Make predictions
future_values = predictor.predict(periods=3)
print(f"Next 3 predictions: {future_values}")

# Calculate accuracy metrics
actual = [26, 28, 30]
mape = calculate_mape(actual, future_values)
print(f"MAPE: {mape:.2f}%")
```

#### Advanced Forecasting with Darts
```python
from pega_tools.forecast import DartsForecasterFactory, TimeSeriesData
from datetime import datetime, timedelta

# Create time series with timestamps
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
values = [10 + 0.1*i + 3*np.sin(2*np.pi*i/7) for i in range(100)]  # Trend + seasonality
ts_data = TimeSeriesData(values, dates, frequency='daily')

# Use AutoARIMA for automatic model selection
forecaster = DartsForecasterFactory.create_forecaster('auto_arima', seasonal=True)
forecaster.fit(ts_data)

# Make predictions with confidence intervals
result = forecaster.predict_with_confidence(periods=7)
print(f"Predictions: {result['predictions']}")
print(f"Lower bounds: {result['lower_bound']}")
print(f"Upper bounds: {result['upper_bound']}")

# Get model information
info = forecaster.get_model_info()
print(f"Model type: {info['model_type']}")
print(f"Training samples: {info['training_samples']}")
```

#### Ensemble Forecasting
```python
from pega_tools.forecast import create_ensemble_forecast

# Combine multiple models for better accuracy
models = ['auto_arima', 'auto_ets', 'naive_drift']
weights = [0.5, 0.3, 0.2]  # Custom weights

result = create_ensemble_forecast(
    ts_data, 
    models=models, 
    periods=10,
    weights=weights
)

print(f"Ensemble predictions: {result['ensemble_predictions']}")
print(f"Individual model results: {result['individual_predictions']}")

# Available forecasting models
factory = DartsForecasterFactory()
available_models = factory.get_available_models()
print(f"Available models: {available_models}")
```

#### Model Comparison and Selection
```python
# Test multiple models and compare performance
models_to_test = ['auto_arima', 'auto_ets', 'auto_theta', 'naive_seasonal']
results = {}

for model_name in models_to_test:
    try:
        forecaster = DartsForecasterFactory.create_forecaster(model_name)
        forecaster.fit(ts_data)
        predictions = forecaster.predict(periods=5)
        
        # Calculate accuracy if you have test data
        # accuracy = calculate_mape(actual_test_values, predictions)
        
        results[model_name] = {
            'predictions': predictions,
            'model_info': forecaster.get_model_info()
        }
        print(f"✓ {model_name}: Success")
    except Exception as e:
        print(f"✗ {model_name}: {e}")

# Best model selection based on your criteria
best_model = min(results.keys())  # Your selection logic here
print(f"Selected model: {best_model}")
```

#### Available Forecasting Models

The Darts integration provides access to state-of-the-art forecasting models:

**🤖 Automatic Models** (Recommended for most use cases)
- **AutoARIMA** - Automatic ARIMA model selection with seasonal support
- **AutoETS** - Exponential smoothing with automatic parameter optimization
- **AutoTheta** - Theta method for trend and seasonality forecasting
- **AutoTBATS** - TBATS model for complex multiple seasonalities

**🧠 Advanced Models**
- **KalmanForecaster** - Kalman Filter-based state space modeling

**📊 Naive Models** (Fast baselines)
- **NaiveMean** - Simple mean-based forecasting
- **NaiveSeasonal** - Seasonal naive method
- **NaiveDrift** - Linear trend extrapolation
- **NaiveMovingAverage** - Moving average forecasting

**🔧 Model Features**
- ✅ **Automatic parameter tuning** for most models
- ✅ **Seasonal pattern detection** and handling
- ✅ **Confidence intervals** for uncertainty quantification
- ✅ **Ensemble forecasting** for improved accuracy
- ✅ **Graceful fallback** when Darts is not installed

### JSON Processing Tools

```python
from pega_tools.json import JSONProcessor, JSONValidator, flatten_json

# Process JSON files
processor = JSONProcessor()
data = processor.load_from_file("data.json")

# Flatten nested JSON
flat_data = flatten_json(data)
print("Flattened keys:", list(flat_data.keys()))

# Validate JSON structure
validator = JSONValidator()
schema = {
    "required": ["name", "email"],
    "types": {"name": "string", "email": "string"},
    "formats": {"email": "email"}
}

user_data = {"name": "John Doe", "email": "john@example.com"}
try:
    from pega_tools.json import validate_schema
    validate_schema(user_data, schema)
    print("✓ Valid JSON structure")
except Exception as e:
    print(f"✗ Invalid: {e}")
```

### Network Analysis Tools

```python
from pega_tools.pcap import PCAPParser, PacketAnalyzer, PacketFilter

# Parse PCAP file
parser = PCAPParser()
packets = parser.parse_file("network_capture.pcap")
print(f"Parsed {len(packets)} packets")

# Analyze traffic
analyzer = PacketAnalyzer(packets)
report = analyzer.generate_summary_report()

print(f"Total packets: {report['basic_stats']['total_packets']}")
print(f"Duration: {report['basic_stats']['duration_seconds']:.1f} seconds")
print("Protocol distribution:", report['protocol_distribution'])

# Filter packets
packet_filter = PacketFilter()
tcp_packets = packet_filter.filter_by_protocol(packets, "TCP")
print(f"TCP packets: {len(tcp_packets)}")

# Get top talkers
top_hosts = analyzer.get_top_talkers(limit=5)
for host in top_hosts:
    print(f"Host {host['host']}: {host['total_packets']} packets")
```

## 🎯 Command Line Interface

```bash
# Check Pega instance health
pega-tools health --url https://your-pega-instance.com

# Show version
pega-tools version

# Get help
pega-tools --help
```

## 🏗 Project Structure

```
pega_tools/
├── geographic/          # Geographic utilities and calculations
│   ├── coordinates.py   # Coordinate handling and distance calculations
│   ├── geocoding.py     # Address geocoding services
│   ├── utils.py         # Geographic utility functions
│   └── visualization.py # Geographic visualization tools
│
├── forecast/            # Forecasting and prediction tools
│   ├── data.py          # Time series data management
│   ├── predictor.py     # Basic prediction models (linear, exponential)
│   ├── darts_forecasters.py  # Advanced Darts-based forecasters
│   └── metrics.py       # Forecast accuracy metrics
│
├── json/                # JSON processing and manipulation
│   ├── processor.py     # JSON file I/O and parsing
│   ├── validator.py     # JSON validation and schema checking
│   ├── transformer.py   # Data transformation utilities
│   └── utils.py         # JSON utilities (flatten, merge, etc.)
│
└── pcap/                # Network packet analysis
    ├── parser.py        # PCAP file parsing
    ├── analyzer.py      # Network traffic analysis
    ├── filters.py       # Packet filtering
    └── statistics.py    # Network statistics calculation
```

## 🧪 Development

### Running Tests
```bash
# Run all tests
make test
# or
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/pega_tools --cov-report=html
```

### Code Quality
```bash
# Format code
make format
# or
black src/ tests/

# Lint code
make lint
# or
flake8 src/ tests/

# Type checking
make type-check
# or
mypy src/pega_tools/
```

### Building
```bash
# Clean and build package
make build

# Build wheel only
python -m build

# Install locally
pip install -e .
```

## 📊 Dependencies

### Core Dependencies
- `requests` - HTTP client for API interactions
- `click` - Command-line interface framework
- `numpy` - Numerical computing for forecasting
- `darts` - Advanced time series forecasting library (optional)

### Optional Dependencies
- `matplotlib` - Plotting and visualization (viz extra)
- `pandas` - Data manipulation (included with darts)
- `scikit-learn` - Machine learning utilities (included with darts)
- `torch` - Deep learning framework (included with darts)

### Development Dependencies
- `pytest` - Testing framework
- `black` - Code formatting
- `flake8` - Code linting
- `mypy` - Type checking
- `pytest-cov` - Coverage reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`make test`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new functionality
- Update documentation as needed

## 📋 Use Cases

### Data Analysis & Forecasting
- **Time series forecasting** with state-of-the-art models
- **Seasonal pattern detection** and trend analysis
- **Ensemble modeling** for improved prediction accuracy
- **Confidence interval estimation** for risk assessment
- Geographic data processing and visualization
- JSON data transformation and validation

### Business Intelligence
- **Sales forecasting** with multiple model comparison
- **Demand planning** using seasonal and trend components
- **Anomaly detection** in time series data
- **Model performance evaluation** and selection

### Network Monitoring
- Network traffic analysis and monitoring
- Security incident investigation
- Performance optimization and capacity planning

### System Integration
- API data processing and validation
- Configuration file management and transformation
- Data format conversion and standardization

## 📁 Examples

The `examples/` directory contains comprehensive demonstrations:

- **`darts_forecasting_example.py`** - Complete Darts forecasting tutorial
  - Individual model demonstrations
  - Ensemble forecasting examples
  - Confidence interval calculations
  - Model comparison and selection
  - Visualization with matplotlib

- **`geographic_visualization_example.py`** - Geographic visualization tutorial
  - Scatter plots with customizable parameters
  - 2D histogram visualizations
  - Temporal and spatial gradient analysis
  - Animated geographic visualizations
  - Custom parameter configurations

Run the examples:
```bash
cd examples/
python darts_forecasting_example.py
python geographic_visualization_example.py
```

**Note**: Install forecasting dependencies first:
```bash
pip install pega-tools[forecasting,viz]
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📈 Changelog

### 0.2.0 (2024) - Advanced Forecasting Update
- 🚀 **NEW**: Darts library integration for advanced forecasting
- 🤖 **NEW**: AutoARIMA, AutoETS, AutoTheta, AutoTBATS forecasters
- 🧠 **NEW**: Kalman Filter and naive forecasting methods
- 📊 **NEW**: Ensemble forecasting with multiple model combinations
- 📈 **NEW**: Confidence intervals and probabilistic forecasting
- 🏭 **NEW**: Model factory pattern for easy forecaster creation
- ⚡ **NEW**: Graceful fallback when Darts is not installed
- 📦 **NEW**: Optional dependencies for forecasting and visualization
- 🧪 **ENHANCED**: Comprehensive test suite for all forecasting models
- 📚 **ENHANCED**: Updated documentation with advanced examples

### 0.1.0 (2024) - Initial Release
- ✨ Initial release with four comprehensive sub-packages
- 📍 Geographic tools with coordinate handling and geocoding
- 📈 Basic forecasting tools with linear and exponential models
- 🔧 JSON processing with validation and transformation
- 🌐 PCAP analysis with network monitoring capabilities
- 🎯 Command-line interface for common operations
- 📚 Comprehensive documentation and examples
- 🧪 Full test suite with CI/CD integration

## 🆘 Support

- 📖 Documentation: [Project Wiki](https://github.com/yourusername/pega-tools/wiki)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/pega-tools/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/pega-tools/discussions)

---

**Made with ❤️ for data analysis and network monitoring** 