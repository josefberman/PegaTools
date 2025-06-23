# Pega Tools - Knowledge Base & How-To Guide

## 📚 Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Installation & Setup](#installation--setup)
3. [Geographic Tools](#geographic-tools)
4. [Forecasting Tools](#forecasting-tools)
5. [JSON Processing Tools](#json-processing-tools)
6. [Network Analysis Tools](#network-analysis-tools)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)
10. [Examples & Recipes](#examples--recipes)

---

## 🚀 Quick Start Guide

### 30-Second Setup
```bash
# Install basic package
pip install pega-tools

# Install with advanced forecasting
pip install pega-tools[forecasting]

# Install with visualization
pip install pega-tools[viz]

# Install everything
pip install pega-tools[forecasting,viz]
```

### First Example
```python
from pega_tools.forecast import TimeSeriesData, DartsForecasterFactory

# Create sample data
values = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
data = TimeSeriesData(values, frequency='daily')

# Create and use forecaster
forecaster = DartsForecasterFactory.create_forecaster('naive_mean')
forecaster.fit(data)
predictions = forecaster.predict(3)
print(f"Next 3 values: {predictions}")
```

---

## 🔧 Installation & Setup

### Installation Options

| Installation Type | Command | Use Case |
|------------------|---------|----------|
| **Basic** | `pip install pega-tools` | Core functionality only |
| **Forecasting** | `pip install pega-tools[forecasting]` | Advanced time series models |
| **Visualization** | `pip install pega-tools[viz]` | Plotting and charts |
| **Complete** | `pip install pega-tools[forecasting,viz]` | All features |
| **Development** | `pip install -e ".[dev]"` | Development setup |

### Dependency Matrix

| Feature | Required Packages | Optional Packages |
|---------|------------------|-------------------|
| **Core** | requests, click, numpy | - |
| **Forecasting** | darts, pandas, scikit-learn | matplotlib |
| **Geographic** | - | geopy, folium |
| **Network** | - | scapy, pyshark |

### Environment Setup
```bash
# Create virtual environment
python -m venv pega_env
source pega_env/bin/activate  # Linux/Mac
# or
pega_env\Scripts\activate     # Windows

# Install package
pip install pega-tools[forecasting,viz]

# Verify installation
python -c "from pega_tools import __version__; print(__version__)"
```

---

## 📍 Geographic Tools

### Core Classes
- **`Coordinate`** - Handle latitude/longitude pairs
- **`Geocoder`** - Convert addresses to coordinates
- **Distance functions** - Calculate distances between points

### How-To Examples

#### Working with Coordinates
```python
from pega_tools.geographic import Coordinate, distance_between

# Create coordinates
home = Coordinate(40.7128, -74.0060)  # New York
office = Coordinate(40.7589, -73.9851)  # Times Square

# Validate coordinates
if home.is_valid():
    print(f"Home: {home.latitude}, {home.longitude}")

# Calculate distance
distance = distance_between(home, office, unit="miles")
print(f"Distance: {distance:.2f} miles")
```

#### Coordinate Formatting
```python
from pega_tools.geographic import format_coordinates

# Different formats
coord = Coordinate(40.7128, -74.0060)
print(format_coordinates(coord, "decimal"))     # 40.7128, -74.0060
print(format_coordinates(coord, "dms"))         # 40°42'46"N, 74°0'22"W
print(format_coordinates(coord, "dm"))          # 40°42.768'N, 74°0.36'W
```

#### Geocoding Addresses
```python
from pega_tools.geographic import Geocoder

geocoder = Geocoder()

# Forward geocoding (address → coordinates)
location = geocoder.geocode("1600 Pennsylvania Avenue, Washington DC")
if location:
    print(f"White House: {location.latitude}, {location.longitude}")

# Reverse geocoding (coordinates → address)
address = geocoder.reverse_geocode(Coordinate(38.8977, -77.0365))
if address:
    print(f"Address: {address}")
```

### Common Use Cases
- **Route Planning** - Calculate distances between multiple points
- **Geofencing** - Check if coordinates are within boundaries
- **Location Validation** - Verify address accuracy
- **Mapping Integration** - Prepare data for mapping services

---

## 🎨 Geographic Visualization Tools

### Core Functions
- **`plot_geographic_scatter()`** - Create scatter plots of geographic coordinates
- **`plot_geographic_histogram_2d()`** - Generate 2D histograms of geographic data
- **`plot_temporal_gradient_histogram()`** - Visualize temporal gradients in geographic data
- **`plot_spatial_gradient_histogram()`** - Visualize spatial gradients in geographic data
- **`create_geographic_animation()`** - Create animated visualizations of geographic data over time

### How-To Examples

#### Basic Geographic Scatter Plot
```python
from pega_tools.geographic import plot_geographic_scatter, Coordinate
import matplotlib.pyplot as plt

# Create sample coordinates
coordinates = [
    Coordinate(40.7128, -74.0060),  # New York
    Coordinate(34.0522, -118.2437),  # Los Angeles
    Coordinate(41.8781, -87.6298),  # Chicago
]

# Basic scatter plot
fig = plot_geographic_scatter(
    coordinates=coordinates,
    title="Major US Cities",
    figsize=(10, 8),
    marker_size=100,
    color_map="viridis",
    background_map=True
)
plt.savefig("cities_scatter.png")
plt.close(fig)
```

#### Scatter Plot with Values
```python
# Scatter plot with associated values
values = [100, 85, 92]  # Activity levels

fig = plot_geographic_scatter(
    coordinates=coordinates,
    values=values,
    title="City Activity Levels",
    marker_size=values,  # Size based on values
    color_map="plasma",
    colorbar_label="Activity Level",
    alpha=0.8,
    background_map=True
)
plt.savefig("cities_activity.png")
plt.close(fig)
```

#### 2D Histogram Visualization
```python
from pega_tools.geographic import plot_geographic_histogram_2d

# Generate more data points for histogram
import numpy as np
np.random.seed(42)

# Create clustered data around cities
all_coordinates = []
for i in range(100):
    # Randomly select a base city
    base_cities = [(40.7128, -74.0060), (34.0522, -118.2437), (41.8781, -87.6298)]
    base_lat, base_lon = base_cities[i % 3]
    
    # Add some noise
    lat = base_lat + np.random.normal(0, 0.01)
    lon = base_lon + np.random.normal(0, 0.01)
    all_coordinates.append(Coordinate(lat, lon))

# Create 2D histogram
fig = plot_geographic_histogram_2d(
    coordinates=all_coordinates,
    title="Geographic Density Distribution",
    figsize=(12, 8),
    bins=30,
    color_map="viridis",
    log_scale=True,
    show_colorbar=True,
    colorbar_label="Point Density",
    background_map=True
)
plt.savefig("geographic_density.png")
plt.close(fig)
```

#### Temporal Gradient Analysis
```python
from pega_tools.geographic import plot_temporal_gradient_histogram
from datetime import datetime, timedelta

# Create timestamps for temporal analysis
timestamps = []
for i in range(100):
    timestamp = datetime.now() - timedelta(
        days=np.random.uniform(0, 7),
        hours=np.random.uniform(0, 24)
    )
    timestamps.append(timestamp)

fig = plot_temporal_gradient_histogram(
    coordinates=all_coordinates,
    timestamps=timestamps,
    title="Temporal Geographic Analysis",
    figsize=(15, 6),
    bins=25,
    color_map="plasma",
    log_scale=False,
    background_map=True
)
plt.savefig("temporal_gradient.png")
plt.close(fig)
```

#### Spatial Gradient Analysis
```python
from pega_tools.geographic import plot_spatial_gradient_histogram

# Create associated values for spatial analysis
spatial_values = [np.random.exponential(10) for _ in range(100)]

fig = plot_spatial_gradient_histogram(
    coordinates=all_coordinates,
    values=spatial_values,
    title="Spatial Value Distribution",
    figsize=(15, 6),
    bins=30,
    color_map="coolwarm",
    gradient_method="density",  # or "interpolation"
    background_map=True
)
plt.savefig("spatial_gradient.png")
plt.close(fig)
```

#### Animated Geographic Visualization
```python
from pega_tools.geographic import create_geographic_animation

# Create animation of geographic data over time
try:
    output_file = create_geographic_animation(
        coordinates=all_coordinates,
        timestamps=timestamps,
        values=spatial_values,
        output_file="geographic_animation.gif",
        duration=5,
        figsize=(10, 8)
    )
    print(f"Animation saved as: {output_file}")
except ImportError:
    print("Animation requires matplotlib.animation support")
```

### Customization Options

#### Advanced Scatter Plot Parameters
```python
fig = plot_geographic_scatter(
    coordinates=coordinates,
    values=values,
    title="Custom Scatter Plot",
    figsize=(12, 10),
    marker_size=[v/2 for v in values],  # Variable marker sizes
    color_map="Spectral",
    alpha=0.6,
    show_colorbar=True,
    colorbar_label="Custom Values",
    background_map=True,
    edgecolors='black',  # Additional matplotlib parameters
    linewidth=0.5
)
```

#### Advanced Histogram Parameters
```python
fig = plot_geographic_histogram_2d(
    coordinates=coordinates,
    title="Custom 2D Histogram",
    figsize=(12, 10),
    bins=(50, 50),  # Different bins for lat/lon
    color_map="jet",
    log_scale=True,
    show_colorbar=True,
    colorbar_label="Custom Count",
    background_map=True,
    alpha=0.8
)
```

### Common Use Cases
- **Population Density Analysis** - Visualize population distribution across regions
- **Traffic Pattern Analysis** - Analyze movement patterns over time
- **Environmental Monitoring** - Track changes in environmental data across locations
- **Business Intelligence** - Visualize sales or activity data by location
- **Research Visualization** - Present geographic research findings
- **Real-time Monitoring** - Create animated visualizations for live data

### Performance Tips
- Use appropriate bin sizes for histograms (too many bins can create noise)
- Consider log scale for data with wide value ranges
- Use interpolation method for smooth spatial gradients when scipy is available
- Limit animation frames for large datasets to improve performance

---

## 📈 Forecasting Tools

### Model Categories

#### 🤖 Automatic Models (Recommended)
- **AutoARIMA** - Best for general time series with trend/seasonality
- **AutoETS** - Excellent for exponential smoothing scenarios
- **AutoTheta** - Good for trend-based forecasting
- **AutoTBATS** - Handles complex multiple seasonalities

#### 📊 Naive Models (Fast Baselines)
- **NaiveMean** - Simple average of historical data
- **NaiveSeasonal** - Repeats seasonal patterns
- **NaiveDrift** - Linear trend extrapolation
- **NaiveMovingAverage** - Rolling average forecasting

### How-To Examples

#### Basic Forecasting Workflow
```python
from pega_tools.forecast import TimeSeriesData, DartsForecasterFactory
from datetime import datetime, timedelta

# 1. Prepare data
dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
values = [10 + 0.5*i + 2*np.sin(i/7*2*np.pi) for i in range(30)]
data = TimeSeriesData(values, dates, frequency='daily')

# 2. Choose and create model
forecaster = DartsForecasterFactory.create_forecaster('auto_arima')

# 3. Fit model
forecaster.fit(data)

# 4. Make predictions
predictions = forecaster.predict(periods=7)
print(f"7-day forecast: {predictions}")

# 5. Get model info
info = forecaster.get_model_info()
print(f"Model: {info['model_type']}, Samples: {info['training_samples']}")
```

#### Confidence Intervals
```python
# Get predictions with uncertainty bounds
result = forecaster.predict_with_confidence(periods=5)

print("Predictions with 80% confidence intervals:")
for i in range(5):
    pred = result['predictions'][i]
    lower = result['lower_bound'][i]
    upper = result['upper_bound'][i]
    print(f"Day {i+1}: {pred:.2f} [{lower:.2f}, {upper:.2f}]")
```

#### Ensemble Forecasting
```python
from pega_tools.forecast import create_ensemble_forecast

# Combine multiple models
models = ['auto_arima', 'auto_ets', 'naive_drift']
weights = [0.5, 0.3, 0.2]  # Custom weights

result = create_ensemble_forecast(
    data, 
    models=models, 
    periods=10,
    weights=weights
)

print(f"Ensemble forecast: {result['ensemble_predictions']}")
print(f"Individual models: {result['individual_predictions']}")
```

#### Model Selection & Comparison
```python
# Test multiple models
models_to_test = ['auto_arima', 'auto_ets', 'naive_seasonal', 'naive_drift']
results = {}

for model_name in models_to_test:
    try:
        forecaster = DartsForecasterFactory.create_forecaster(model_name)
        forecaster.fit(data)
        predictions = forecaster.predict(5)
        
        results[model_name] = {
            'predictions': predictions,
            'model_info': forecaster.get_model_info()
        }
        print(f"✓ {model_name}: Success")
    except Exception as e:
        print(f"✗ {model_name}: {e}")

# Select best model (implement your criteria)
best_model = 'auto_arima'  # Your selection logic
print(f"Selected: {best_model}")
```

### Model Selection Guide

| Data Characteristics | Recommended Model | Why |
|---------------------|------------------|-----|
| **Trend + Seasonality** | AutoARIMA | Handles both components automatically |
| **Exponential Growth** | AutoETS | Optimized for exponential patterns |
| **Multiple Seasonalities** | AutoTBATS | Designed for complex seasonal patterns |
| **Simple Trend** | AutoTheta | Effective for trend-based data |
| **Irregular/Noisy** | Ensemble | Combines multiple approaches |
| **Quick Baseline** | NaiveDrift | Fast and simple |

### Performance Optimization

#### Memory Management
```python
# For large datasets, use chunking
def forecast_large_dataset(data, chunk_size=1000):
    if len(data.values) > chunk_size:
        # Use last chunk_size points for training
        recent_data = TimeSeriesData(
            data.values[-chunk_size:],
            data.timestamps[-chunk_size:] if data.timestamps else None,
            data.frequency
        )
        return forecast_large_dataset(recent_data, chunk_size)
    else:
        forecaster = DartsForecasterFactory.create_forecaster('auto_arima')
        forecaster.fit(data)
        return forecaster.predict(periods=7)
```

#### Parallel Processing
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_model_comparison(data, models):
    def test_model(model_name):
        try:
            forecaster = DartsForecasterFactory.create_forecaster(model_name)
            forecaster.fit(data)
            return model_name, forecaster.predict(5)
        except Exception as e:
            return model_name, None
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(test_model, models))
    
    return {name: pred for name, pred in results if pred is not None}
```

---

## 🔧 JSON Processing Tools

### Core Classes
- **`JSONProcessor`** - File I/O and parsing
- **`JSONValidator`** - Schema validation
- **`JSONTransformer`** - Data transformation
- **Utility functions** - Flatten, merge, compare

### How-To Examples

#### Basic JSON Operations
```python
from pega_tools.json import JSONProcessor, flatten_json, unflatten_json

# Load and save JSON
processor = JSONProcessor()
data = processor.load_from_file("config.json")
processor.save_to_file(data, "backup.json", prettify=True)

# Flatten nested JSON
nested = {"user": {"name": "John", "details": {"age": 30, "city": "NYC"}}}
flat = flatten_json(nested)
print(flat)  # {"user.name": "John", "user.details.age": 30, ...}

# Unflatten back
original = unflatten_json(flat)
print(original)  # Original nested structure
```

#### JSON Validation
```python
from pega_tools.json import JSONValidator, validate_schema

validator = JSONValidator()

# Define schema
schema = {
    "required": ["name", "email", "age"],
    "types": {"name": "string", "email": "string", "age": "integer"},
    "formats": {"email": "email"},
    "ranges": {"age": {"min": 0, "max": 150}}
}

# Validate data
user_data = {"name": "John Doe", "email": "john@example.com", "age": 30}

try:
    validate_schema(user_data, schema)
    print("✓ Valid JSON")
except Exception as e:
    print(f"✗ Invalid: {e}")

# Field validation
if validator.validate_field("john@example.com", "email"):
    print("✓ Valid email")
```

#### Data Transformation
```python
from pega_tools.json import JSONTransformer

transformer = JSONTransformer()

# Transform field values
data = {"users": [{"name": "john doe"}, {"name": "jane smith"}]}
transformed = transformer.transform_field(data, "name", str.title)
print(transformed)  # Names are now title case

# Normalize data
normalized = transformer.normalize_data(data)

# Type conversion
converted = transformer.convert_types(data, {"age": int, "active": bool})
```

#### Advanced JSON Utilities
```python
from pega_tools.json import merge_json, find_json_differences, extract_json_paths

# Merge JSON objects
base = {"name": "John", "age": 30}
update = {"age": 31, "city": "NYC"}
merged = merge_json(base, update)
print(merged)  # {"name": "John", "age": 31, "city": "NYC"}

# Find differences
old_data = {"name": "John", "age": 30, "city": "Boston"}
new_data = {"name": "John", "age": 31, "city": "NYC"}
diff = find_json_differences(old_data, new_data)
print(diff)  # Shows changed fields

# Extract all paths
paths = extract_json_paths({"user": {"profile": {"name": "John"}}})
print(paths)  # ["user", "user.profile", "user.profile.name"]
```

### Common Patterns

#### Configuration Management
```python
class ConfigManager:
    def __init__(self, config_file):
        self.processor = JSONProcessor()
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        try:
            return self.processor.load_from_file(self.config_file)
        except FileNotFoundError:
            return {}
    
    def save_config(self):
        self.processor.save_to_file(self.config, self.config_file, prettify=True)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def set(self, key, value):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        self.save_config()

# Usage
config = ConfigManager("app_config.json")
config.set("database.host", "localhost")
config.set("database.port", 5432)
print(config.get("database.host"))  # localhost
```

---

## 🌐 Network Analysis Tools

### Core Classes
- **`PCAPParser`** - Parse PCAP files
- **`PacketAnalyzer`** - Analyze network traffic
- **`PacketFilter`** - Filter packets by criteria
- **`NetworkStatistics`** - Calculate network metrics

### How-To Examples

#### Basic PCAP Analysis
```python
from pega_tools.pcap import PCAPParser, PacketAnalyzer

# Parse PCAP file
parser = PCAPParser()
packets = parser.parse_file("network_capture.pcap")
print(f"Loaded {len(packets)} packets")

# Analyze traffic
analyzer = PacketAnalyzer(packets)
report = analyzer.generate_summary_report()

print(f"Duration: {report['basic_stats']['duration_seconds']:.1f} seconds")
print(f"Total bytes: {report['basic_stats']['total_bytes']:,}")
print("Protocol distribution:", report['protocol_distribution'])
```

#### Packet Filtering
```python
from pega_tools.pcap import PacketFilter

filter_obj = PacketFilter()

# Filter by protocol
tcp_packets = filter_obj.filter_by_protocol(packets, "TCP")
udp_packets = filter_obj.filter_by_protocol(packets, "UDP")

# Filter by IP address
local_traffic = filter_obj.filter_by_ip(packets, "192.168.1.0/24")

# Filter by time range
from datetime import datetime, timedelta
start_time = datetime.now() - timedelta(hours=1)
end_time = datetime.now()
recent_packets = filter_obj.filter_by_time_range(packets, start_time, end_time)

# Complex filtering
web_traffic = filter_obj.filter_by_protocol(
    filter_obj.filter_by_port(packets, [80, 443]), 
    "TCP"
)
```

#### Network Statistics
```python
from pega_tools.pcap import NetworkStatistics

stats = NetworkStatistics(packets)

# Basic metrics
metrics = stats.calculate_basic_metrics()
print(f"Packet rate: {metrics['packet_rate']:.2f} packets/sec")
print(f"Throughput: {metrics['throughput_mbps']:.2f} Mbps")

# Top talkers
top_hosts = stats.get_top_hosts(limit=10)
for host in top_hosts[:5]:
    print(f"{host['host']}: {host['bytes']:,} bytes")

# Protocol analysis
protocol_stats = stats.analyze_protocols()
for protocol, stats in protocol_stats.items():
    print(f"{protocol}: {stats['packet_count']} packets")
```

#### Security Analysis
```python
# Detect suspicious activity
def detect_anomalies(packets):
    analyzer = PacketAnalyzer(packets)
    
    # High volume connections
    top_talkers = analyzer.get_top_talkers(limit=5)
    suspicious_hosts = [h for h in top_talkers if h['total_packets'] > 10000]
    
    # Port scanning detection
    filter_obj = PacketFilter()
    syn_packets = filter_obj.filter_by_tcp_flags(packets, ["SYN"])
    
    # Group by source IP
    port_scan_candidates = {}
    for packet in syn_packets:
        src_ip = packet.src_ip
        if src_ip not in port_scan_candidates:
            port_scan_candidates[src_ip] = set()
        port_scan_candidates[src_ip].add(packet.dst_port)
    
    # Flag IPs scanning many ports
    scanners = {ip: ports for ip, ports in port_scan_candidates.items() 
                if len(ports) > 50}
    
    return {
        'high_volume_hosts': suspicious_hosts,
        'potential_scanners': scanners
    }

anomalies = detect_anomalies(packets)
if anomalies['potential_scanners']:
    print("⚠️ Potential port scanners detected:")
    for ip, ports in anomalies['potential_scanners'].items():
        print(f"  {ip}: {len(ports)} ports scanned")
```

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### Installation Problems

**Issue**: `ModuleNotFoundError: No module named 'darts'`
```bash
# Solution: Install forecasting dependencies
pip install pega-tools[forecasting]
```

**Issue**: `Microsoft Visual C++ 14.0 is required`
```bash
# Solution: Install pre-compiled packages or Visual Studio Build Tools
pip install --only-binary=all pega-tools[forecasting]
```

#### Forecasting Issues

**Issue**: `Model must be fitted before making predictions`
```python
# Solution: Always fit before predict
forecaster = DartsForecasterFactory.create_forecaster('auto_arima')
forecaster.fit(data)  # Don't forget this!
predictions = forecaster.predict(5)
```

**Issue**: `Not enough data points for model`
```python
# Solution: Check minimum data requirements
if len(data.values) < 10:
    print("Warning: Consider using naive models for small datasets")
    forecaster = DartsForecasterFactory.create_forecaster('naive_mean')
else:
    forecaster = DartsForecasterFactory.create_forecaster('auto_arima')
```

#### Data Format Issues

**Issue**: `Invalid coordinate format`
```python
# Solution: Validate coordinates
coord = Coordinate(lat, lon)
if not coord.is_valid():
    print(f"Invalid coordinate: {lat}, {lon}")
    # Handle invalid data
```

**Issue**: `JSON parsing errors`
```python
# Solution: Use try-catch with validation
try:
    data = processor.load_from_file("data.json")
    validate_schema(data, schema)
except Exception as e:
    print(f"Data error: {e}")
    # Handle invalid JSON
```

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all operations will show detailed logs
forecaster = DartsForecasterFactory.create_forecaster('auto_arima')
```

### Performance Issues

**Memory optimization for large datasets**:
```python
# Use generators for large files
def process_large_pcap(filename):
    parser = PCAPParser()
    for packet_batch in parser.parse_file_chunked(filename, chunk_size=1000):
        # Process batch
        yield analyze_batch(packet_batch)

# Limit forecasting data
def prepare_forecast_data(values, max_points=1000):
    if len(values) > max_points:
        return values[-max_points:]  # Use recent data only
    return values
```

---

## ✨ Best Practices

### Code Organization

#### Project Structure
```
your_project/
├── src/
│   ├── analysis/
│   │   ├── forecasting.py    # Forecasting logic
│   │   ├── geographic.py     # Location analysis
│   │   └── network.py        # Network analysis
│   ├── data/
│   │   ├── processors.py     # Data processing
│   │   └── validators.py     # Data validation
│   └── utils/
│       └── helpers.py        # Utility functions
├── config/
│   ├── settings.json         # Configuration
│   └── schemas.json          # Validation schemas
├── tests/
└── examples/
```

#### Configuration Management
```python
# config/settings.json
{
    "forecasting": {
        "default_model": "auto_arima",
        "confidence_level": 0.8,
        "max_training_points": 1000
    },
    "geographic": {
        "default_units": "miles",
        "precision": 6
    }
}

# src/utils/config.py
from pega_tools.json import JSONProcessor

class Settings:
    def __init__(self):
        self.processor = JSONProcessor()
        self.config = self.processor.load_from_file("config/settings.json")
    
    def get_forecast_config(self):
        return self.config.get("forecasting", {})
```

### Error Handling

#### Robust Forecasting
```python
def safe_forecast(data, model_preferences=['auto_arima', 'auto_ets', 'naive_drift']):
    """Try multiple models until one works."""
    for model_name in model_preferences:
        try:
            forecaster = DartsForecasterFactory.create_forecaster(model_name)
            forecaster.fit(data)
            return forecaster, model_name
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    raise Exception("No forecasting model could be fitted")

# Usage
try:
    forecaster, model_used = safe_forecast(data)
    predictions = forecaster.predict(7)
    print(f"Using {model_used}: {predictions}")
except Exception as e:
    print(f"Forecasting failed: {e}")
```

#### Data Validation Pipeline
```python
def validate_and_process_data(raw_data, schema):
    """Complete data validation and processing pipeline."""
    try:
        # 1. Basic validation
        if not raw_data:
            raise ValueError("Empty data provided")
        
        # 2. Schema validation
        validate_schema(raw_data, schema)
        
        # 3. Data cleaning
        cleaned_data = clean_data(raw_data)
        
        # 4. Type conversion
        processed_data = convert_types(cleaned_data)
        
        return processed_data, True
    
    except Exception as e:
        print(f"Data validation failed: {e}")
        return None, False
```

### Testing Strategies

#### Unit Testing
```python
import pytest
from pega_tools.forecast import TimeSeriesData, DartsForecasterFactory

class TestForecasting:
    def test_basic_forecasting(self):
        data = TimeSeriesData([1, 2, 3, 4, 5], frequency='daily')
        forecaster = DartsForecasterFactory.create_forecaster('naive_mean')
        forecaster.fit(data)
        predictions = forecaster.predict(2)
        assert len(predictions) == 2
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_ensemble_forecasting(self):
        data = TimeSeriesData(list(range(20)), frequency='daily')
        result = create_ensemble_forecast(
            data, ['naive_mean', 'naive_drift'], periods=3
        )
        assert 'ensemble_predictions' in result
        assert len(result['ensemble_predictions']) == 3
```

#### Integration Testing
```python
def test_full_pipeline():
    """Test complete analysis pipeline."""
    # 1. Load data
    processor = JSONProcessor()
    raw_data = processor.load_from_file("test_data.json")
    
    # 2. Validate
    assert validate_schema(raw_data, test_schema)
    
    # 3. Process
    ts_data = TimeSeriesData(raw_data['values'])
    
    # 4. Forecast
    forecaster = DartsForecasterFactory.create_forecaster('auto_arima')
    forecaster.fit(ts_data)
    predictions = forecaster.predict(5)
    
    # 5. Validate results
    assert len(predictions) == 5
    assert all(p > 0 for p in predictions)  # Business logic validation
```

---

## 📖 API Reference

### Quick Reference Tables

#### Forecasting Models
| Model | Class | Best For | Parameters |
|-------|-------|----------|------------|
| AutoARIMA | `DartsAutoARIMA` | General time series | `seasonal`, `stepwise` |
| AutoETS | `DartsAutoETS` | Exponential smoothing | `seasonal_periods` |
| AutoTheta | `DartsAutoTheta` | Trend forecasting | `seasonal_periods` |
| NaiveMean | `DartsNaiveMean` | Simple baseline | None |
| NaiveDrift | `DartsNaiveDrift` | Linear trend | None |

#### Geographic Functions
| Function | Purpose | Parameters | Returns |
|----------|---------|------------|---------|
| `distance_between()` | Calculate distance | `coord1, coord2, unit` | `float` |
| `format_coordinates()` | Format display | `coord, format_type` | `string` |
| `Geocoder.geocode()` | Address → Coord | `address` | `Coordinate` |
| `plot_geographic_scatter()` | Create scatter plot | `coordinates, values, title, figsize, ...` | `plt.Figure` |
| `plot_geographic_histogram_2d()` | Create 2D histogram | `coordinates, bins, title, figsize, ...` | `plt.Figure` |
| `plot_temporal_gradient_histogram()` | Temporal gradient | `coordinates, timestamps, bins, ...` | `plt.Figure` |
| `plot_spatial_gradient_histogram()` | Spatial gradient | `coordinates, values, bins, ...` | `plt.Figure` |
| `create_geographic_animation()` | Create animation | `coordinates, timestamps, values, ...` | `str` |

#### JSON Utilities
| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `flatten_json()` | Flatten nested JSON | `dict` | `dict` |
| `merge_json()` | Merge objects | `dict, dict` | `dict` |
| `validate_schema()` | Validate structure | `data, schema` | `bool/Exception` |

### Method Signatures

#### TimeSeriesData
```python
class TimeSeriesData:
    def __init__(self, values, timestamps=None, frequency=None)
    def get_statistics(self) -> dict
    def detect_trend(self) -> str
    def get_seasonal_info(self) -> dict
```

#### DartsForecasterFactory
```python
class DartsForecasterFactory:
    @classmethod
    def create_forecaster(cls, model_type: str, **kwargs) -> DartsForecasterBase
    
    @classmethod
    def get_available_models(cls) -> List[str]
```

#### Forecaster Base Methods
```python
class DartsForecasterBase:
    def fit(self, data: TimeSeriesData) -> None
    def predict(self, periods: int) -> List[float]
    def predict_with_confidence(self, periods: int) -> Dict[str, List[float]]
    def get_model_info(self) -> Dict[str, Any]
```

---

## 🍳 Examples & Recipes

### Recipe 1: Sales Forecasting Dashboard
```python
def create_sales_forecast_dashboard(sales_data):
    """Complete sales forecasting with multiple models."""
    
    # Prepare data
    ts_data = TimeSeriesData(
        sales_data['daily_sales'], 
        sales_data['dates'], 
        frequency='daily'
    )
    
    # Test multiple models
    models = ['auto_arima', 'auto_ets', 'auto_theta']
    results = {}
    
    for model_name in models:
        forecaster = DartsForecasterFactory.create_forecaster(model_name)
        forecaster.fit(ts_data)
        
        # 30-day forecast with confidence
        forecast_result = forecaster.predict_with_confidence(30)
        
        results[model_name] = {
            'predictions': forecast_result['predictions'],
            'confidence_lower': forecast_result['lower_bound'],
            'confidence_upper': forecast_result['upper_bound'],
            'model_info': forecaster.get_model_info()
        }
    
    # Create ensemble
    ensemble_result = create_ensemble_forecast(
        ts_data, models, periods=30, weights=[0.4, 0.4, 0.2]
    )
    
    return {
        'individual_models': results,
        'ensemble': ensemble_result,
        'data_stats': ts_data.get_statistics()
    }
```

### Recipe 2: Network Security Monitoring
```python
def analyze_network_security(pcap_file):
    """Comprehensive network security analysis."""
    
    # Parse packets
    parser = PCAPParser()
    packets = parser.parse_file(pcap_file)
    
    # Basic analysis
    analyzer = PacketAnalyzer(packets)
    basic_report = analyzer.generate_summary_report()
    
    # Security-focused analysis
    filter_obj = PacketFilter()
    
    # Detect port scanning
    syn_packets = filter_obj.filter_by_tcp_flags(packets, ["SYN"])
    port_scan_analysis = detect_port_scanning(syn_packets)
    
    # Analyze traffic patterns
    stats = NetworkStatistics(packets)
    anomalies = detect_traffic_anomalies(stats)
    
    # Geographic analysis of IPs
    unique_ips = get_unique_external_ips(packets)
    geo_analysis = analyze_ip_geolocation(unique_ips)
    
    return {
        'basic_stats': basic_report,
        'security_alerts': {
            'port_scans': port_scan_analysis,
            'traffic_anomalies': anomalies,
            'geographic_distribution': geo_analysis
        }
    }
```

### Recipe 3: Configuration Management System
```python
class ConfigurationManager:
    """Advanced configuration management with validation."""
    
    def __init__(self, config_dir="config/"):
        self.config_dir = config_dir
        self.processor = JSONProcessor()
        self.validator = JSONValidator()
        self.schemas = self.load_schemas()
    
    def load_schemas(self):
        """Load validation schemas."""
        schema_file = f"{self.config_dir}schemas.json"
        return self.processor.load_from_file(schema_file)
    
    def load_config(self, config_name):
        """Load and validate configuration."""
        config_file = f"{self.config_dir}{config_name}.json"
        config = self.processor.load_from_file(config_file)
        
        # Validate against schema
        if config_name in self.schemas:
            validate_schema(config, self.schemas[config_name])
        
        return config
    
    def update_config(self, config_name, updates):
        """Update configuration with validation."""
        current_config = self.load_config(config_name)
        updated_config = merge_json(current_config, updates)
        
        # Validate updated config
        if config_name in self.schemas:
            validate_schema(updated_config, self.schemas[config_name])
        
        # Save updated config
        config_file = f"{self.config_dir}{config_name}.json"
        self.processor.save_to_file(updated_config, config_file, prettify=True)
        
        return updated_config
    
    def get_nested_value(self, config_name, key_path, default=None):
        """Get nested configuration value."""
        config = self.load_config(config_name)
        keys = key_path.split('.')
        
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

# Usage
config_mgr = ConfigurationManager()
db_config = config_mgr.load_config("database")
api_key = config_mgr.get_nested_value("api", "keys.openai", "default_key")
```

### Recipe 4: Multi-Location Distance Analysis
```python
def analyze_delivery_routes(locations, depot_location):
    """Optimize delivery routes using geographic tools."""
    
    depot = Coordinate(depot_location['lat'], depot_location['lon'])
    delivery_points = [
        Coordinate(loc['lat'], loc['lon']) for loc in locations
    ]
    
    # Calculate distances from depot
    distances = []
    for i, point in enumerate(delivery_points):
        distance = distance_between(depot, point, unit="miles")
        distances.append({
            'location_id': locations[i]['id'],
            'distance': distance,
            'coordinates': point
        })
    
    # Sort by distance
    distances.sort(key=lambda x: x['distance'])
    
    # Calculate total route distance (simple nearest neighbor)
    total_distance = 0
    current_location = depot
    
    for delivery in distances:
        leg_distance = distance_between(current_location, delivery['coordinates'])
        total_distance += leg_distance
        current_location = delivery['coordinates']
    
    # Return to depot
    total_distance += distance_between(current_location, depot)
    
    return {
        'optimized_route': distances,
        'total_distance': total_distance,
        'route_summary': {
            'stops': len(distances),
            'farthest_stop': distances[-1]['distance'],
            'average_distance': sum(d['distance'] for d in distances) / len(distances)
        }
    }
```

### Recipe 5: Geographic Data Visualization Dashboard
```python
def create_geographic_dashboard(coordinates, timestamps, values, output_dir="dashboard"):
    """Create a comprehensive geographic visualization dashboard."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic scatter plot
    fig1 = plot_geographic_scatter(
        coordinates=coordinates,
        values=values,
        title="Geographic Distribution",
        figsize=(12, 8),
        marker_size=values,
        color_map="plasma",
        background_map=True
    )
    fig1.savefig(f"{output_dir}/scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Density histogram
    fig2 = plot_geographic_histogram_2d(
        coordinates=coordinates,
        title="Geographic Density",
        figsize=(12, 8),
        bins=40,
        color_map="viridis",
        log_scale=True,
        background_map=True
    )
    fig2.savefig(f"{output_dir}/density_histogram.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Temporal analysis
    fig3 = plot_temporal_gradient_histogram(
        coordinates=coordinates,
        timestamps=timestamps,
        title="Temporal Patterns",
        figsize=(15, 6),
        bins=30,
        color_map="plasma",
        background_map=True
    )
    fig3.savefig(f"{output_dir}/temporal_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Spatial gradient
    fig4 = plot_spatial_gradient_histogram(
        coordinates=coordinates,
        values=values,
        title="Spatial Value Distribution",
        figsize=(15, 6),
        bins=35,
        color_map="coolwarm",
        gradient_method="density",
        background_map=True
    )
    fig4.savefig(f"{output_dir}/spatial_gradient.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Animation
    try:
        animation_file = create_geographic_animation(
            coordinates=coordinates,
            timestamps=timestamps,
            values=values,
            output_file=f"{output_dir}/geographic_animation.gif",
            duration=8,
            figsize=(12, 8)
        )
        print(f"Animation created: {animation_file}")
    except ImportError:
        print("Animation not available (requires matplotlib.animation)")
    
    # 6. Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Geographic Data Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            .plot img {{ max-width: 100%; height: auto; }}
            .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🌍 Geographic Data Dashboard</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <h3>📊 Summary Statistics</h3>
            <p><strong>Total Points:</strong> {len(coordinates)}</p>
            <p><strong>Date Range:</strong> {min(timestamps).strftime('%Y-%m-%d')} to {max(timestamps).strftime('%Y-%m-%d')}</p>
            <p><strong>Value Range:</strong> {min(values):.2f} to {max(values):.2f}</p>
            <p><strong>Average Value:</strong> {sum(values)/len(values):.2f}</p>
        </div>
        
        <div class="plot">
            <h3>📍 Geographic Distribution</h3>
            <img src="scatter_plot.png" alt="Scatter Plot">
        </div>
        
        <div class="plot">
            <h3>📈 Density Distribution</h3>
            <img src="density_histogram.png" alt="Density Histogram">
        </div>
        
        <div class="plot">
            <h3>⏰ Temporal Analysis</h3>
            <img src="temporal_analysis.png" alt="Temporal Analysis">
        </div>
        
        <div class="plot">
            <h3>🌊 Spatial Gradient</h3>
            <img src="spatial_gradient.png" alt="Spatial Gradient">
        </div>
        
        <div class="plot">
            <h3>🎬 Animation</h3>
            <img src="geographic_animation.gif" alt="Geographic Animation">
        </div>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/dashboard.html", "w") as f:
        f.write(html_content)
    
    print(f"✅ Dashboard created in '{output_dir}/' directory")
    print(f"📄 Open '{output_dir}/dashboard.html' in your browser to view the dashboard")
    
    return {
        'scatter_plot': f"{output_dir}/scatter_plot.png",
        'density_histogram': f"{output_dir}/density_histogram.png",
        'temporal_analysis': f"{output_dir}/temporal_analysis.png",
        'spatial_gradient': f"{output_dir}/spatial_gradient.png",
        'animation': f"{output_dir}/geographic_animation.gif",
        'html_report': f"{output_dir}/dashboard.html"
    }

# Usage example
def example_dashboard():
    """Example of creating a geographic dashboard."""
    from datetime import datetime, timedelta
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    
    # Create coordinates around multiple cities
    base_cities = [
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437),  # Los Angeles
        (41.8781, -87.6298),  # Chicago
        (29.7604, -95.3698),  # Houston
    ]
    
    coordinates = []
    timestamps = []
    values = []
    
    for i in range(n_points):
        # Select random base city
        base_lat, base_lon = base_cities[i % len(base_cities)]
        
        # Add noise
        lat = base_lat + np.random.normal(0, 0.02)
        lon = base_lon + np.random.normal(0, 0.02)
        coordinates.append(Coordinate(lat, lon))
        
        # Generate timestamp over last month
        timestamp = datetime.now() - timedelta(
            days=np.random.uniform(0, 30),
            hours=np.random.uniform(0, 24)
        )
        timestamps.append(timestamp)
        
        # Generate value
        values.append(np.random.exponential(15))
    
    # Create dashboard
    dashboard_files = create_geographic_dashboard(coordinates, timestamps, values)
    return dashboard_files

---

## 🎯 Performance Tips

### Memory Optimization
- Use chunked processing for large datasets
- Implement data generators for streaming
- Clear unused variables with `del`
- Use appropriate data types (int32 vs int64)

### Speed Optimization
- Choose simpler models for quick results
- Use parallel processing for multiple analyses
- Cache frequently used results
- Implement early stopping for iterative processes

### Resource Management
```python
# Context manager for resource cleanup
from contextlib import contextmanager

@contextmanager
def managed_forecaster(model_type):
    forecaster = DartsForecasterFactory.create_forecaster(model_type)
    try:
        yield forecaster
    finally:
        # Cleanup if needed
        del forecaster

# Usage
with managed_forecaster('auto_arima') as forecaster:
    forecaster.fit(data)
    predictions = forecaster.predict(5)
```

---

This knowledge base provides comprehensive coverage of all pega_tools functionality with practical examples, troubleshooting guides, and best practices for real-world usage. 