# XGhostNet Installation Guide

## Prerequisites

- Python 3.8 or higher
- macOS (tested on macOS 14+)
- Homebrew (for system dependencies)

## Quick Start (Core Installation)

1. **Clone the repository and navigate to the backbone directory:**
   ```bash
   cd backbone
   ```

2. **Install core dependencies (no SDR hardware required):**
   ```bash
   pip install -r requirements_core.txt
   ```

3. **Run the application:**
   ```bash
   python src/main.py
   ```

## Full Installation (with SDR Support)

### Step 1: Install System Dependencies

Install required system libraries via Homebrew:

```bash
# Install SoapySDR and related libraries
brew install soapysdr sdrplay hackrf uhd

# Optional: Install additional SDR libraries
brew install rtl-sdr airspy
```

### Step 2: Install Python Dependencies

1. **Install core dependencies:**
   ```bash
   pip install -r requirements_core.txt
   ```

2. **Install SDR dependencies:**
   ```bash
   pip install -r requirements_sdr.txt
   ```

3. **Install additional SDR packages (if needed):**
   ```bash
   # Run the SDR installation helper script
   python install_sdr_packages.py
   
   # Or install manually:
   pip install soapysdr
   pip install hackrf-python  # For HackRF devices
   pip install uhd-python     # For USRP devices
   pip install airspy         # For Airspy devices
   pip install sdrplay        # For SDRplay devices
   ```

### Step 3: Verify Installation

Test that the SDR libraries are working:

```python
# Test RTL-SDR
try:
    import rtlsdr
    print("RTL-SDR: OK")
except ImportError:
    print("RTL-SDR: Not available")

# Test SoapySDR
try:
    import SoapySDR
    print("SoapySDR: OK")
except ImportError:
    print("SoapySDR: Not available")
```

## Troubleshooting

### Common Issues

1. **SoapySDR installation fails:**
   - Make sure you have Homebrew installed
   - Run `brew install soapysdr` first
   - Then try `pip install soapysdr`

2. **Permission errors with SDR devices:**
   - On macOS, you may need to grant permissions to Terminal/your IDE
   - Go to System Preferences > Security & Privacy > Privacy > Full Disk Access

3. **RTL-SDR not detected:**
   - Make sure the device is properly connected
   - Install drivers if needed: `brew install rtl-sdr`

### Alternative Installation Methods

If you're having trouble with the standard installation, try:

1. **Using conda (recommended for scientific computing):**
   ```bash
   conda install -c conda-forge soapysdr pyrtlsdr
   ```

2. **Manual compilation:**
   ```bash
   # Clone and build SoapySDR from source
   git clone https://github.com/pothosware/SoapySDR.git
   cd SoapySDR
   mkdir build && cd build
   cmake ..
   make -j4
   sudo make install
   ```

## Development Setup

For development, install additional tools:

```bash
# Install development dependencies
pip install -r requirements_core.txt

# Install pre-commit hooks (optional)
pre-commit install
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Configuration

1. Copy the example configuration:
   ```bash
   cp config/default_config.json config/my_config.json
   ```

2. Edit the configuration file to match your setup

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your system meets the prerequisites
3. Check the project's issue tracker
4. Ensure all system dependencies are properly installed

## Notes

- The core functionality works without SDR hardware
- SDR support requires specific hardware and drivers
- Some SDR libraries may have licensing restrictions
- Performance may vary depending on your hardware
