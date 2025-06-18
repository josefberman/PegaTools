# Pega Tools

A comprehensive collection of specialized tools and utilities for data analysis, network monitoring, geographic calculations, and forecasting. While originally designed with Pega systems in mind, these tools are versatile and can be used for various data processing and analysis tasks.

## 🚀 Features

### 📍 Geographic Tools
- **Coordinate handling** with validation and conversion
- **Distance calculations** using Haversine formula
- **Geocoding services** for address-to-coordinate conversion
- **Multiple coordinate formats** (decimal, DMS, DM)

### 📈 Forecasting Tools
- **Time series data management** with statistics
- **Linear and exponential prediction models**
- **Forecast accuracy metrics** (MAPE, RMSE, MAE, R²)
- **Trend analysis** and pattern detection

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

### Forecasting Tools

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
│   └── utils.py         # Geographic utility functions
│
├── forecast/            # Forecasting and prediction tools
│   ├── data.py          # Time series data management
│   ├── predictor.py     # Prediction models (linear, exponential)
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

### Data Analysis
- Geographic data processing and visualization
- Time series forecasting and trend analysis
- JSON data transformation and validation

### Network Monitoring
- Network traffic analysis and monitoring
- Security incident investigation
- Performance optimization

### System Integration
- API data processing and validation
- Configuration file management
- Data format conversion

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📈 Changelog

### 0.1.0 (2024)
- ✨ Initial release with four comprehensive sub-packages
- 📍 Geographic tools with coordinate handling and geocoding
- 📈 Forecasting tools with multiple prediction models
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