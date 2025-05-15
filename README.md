# GhostNet: Covert Passive RF Mapping and Mobility Intelligence System

GhostNet is a world-first covert mobility intelligence system that passively maps and tracks the movement of personnel, drones, and ground vehicles using ambient RF emissions without any active signal emission.

## Core Features

- Wideband passive RF scanning (70 MHz – 6 GHz)
- AI-based signal classification and feature extraction
- Distributed receiver mesh for location triangulation
- Spatiotemporal heatmaps and motion trails
- Ghost fingerprinting for RF signature identification
- Offline-capable AI models for edge deployment

## Hardware Requirements

- SDR Hardware (one or more of):
  - HackRF One
  - RTL-SDR
  - BladeRF
- GPS module for time synchronization
- Network connectivity for mesh operation

## System Dependencies

### Required System Libraries

#### Ubuntu/Debian
```bash
# Update package lists
sudo apt update

# Core build dependencies
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev

# SDR Hardware Support
sudo apt install -y \
    rtl-sdr \
    librtlsdr-dev \
    libsoapysdr-dev \
    soapysdr-module-all \
    libuhd-dev \
    uhd-host \
    libiio-dev \
    libiio-utils \
    libairspy-dev

# Optional: Additional SDR modules
sudo apt install -y \
    soapysdr-module-rtlsdr \
    soapysdr-module-hackrf \
    soapysdr-module-bladerf \
    soapysdr-module-uhd \
    soapysdr-module-limesdr \
    soapysdr-module-airspy \
    soapysdr-module-sdrplay

# SDRplay
# Note: SDRplay API must be installed manually from https://www.sdrplay.com/downloads/
```

#### macOS
```bash
# Using Homebrew
brew install \
    python3 \
    cmake \
    libusb \
    soapysdr \
    rtl-sdr \
    uhd \
    iio \
    airspy

# Optional: Additional SDR support
brew install \
    hackrf \
    bladerf \
    limesdr

# SDRplay
# Note: SDRplay API must be installed manually from https://www.sdrplay.com/downloads/
```

#### Windows
1. Install Python 3.8 or later from https://www.python.org/downloads/
2. Install Visual Studio Build Tools 2019 or later
3. Install SDR hardware drivers:
   - RTL-SDR: https://zadig.akeo.ie/
   - SoapySDR: https://github.com/pothosware/SoapySDR/wiki/BuildGuide
   - UHD: https://files.ettus.com/manual/page_install.html
   - libiio: https://wiki.analog.com/resources/tools-software/linux-software/libiio
   - LimeSDR: https://wiki.myriadrf.org/Lime_Suite
   - Airspy: https://airspy.com/download/
   - SDRplay: https://www.sdrplay.com/downloads/

### Python Package Installation

1. Create and activate a virtual environment:
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

2. Install Python packages:
```bash
pip install -r requirements.txt
```

### Hardware Setup

1. Connect your SDR device to your computer
2. Verify device detection:
```bash
# For RTL-SDR
rtl_test

# For SoapySDR devices
SoapySDRUtil --find

# For USRP
uhd_usrp_probe

# For PlutoSDR
iio_info

# For LimeSDR
LimeUtil --find

# For Airspy
airspy_info

# For SDRplay
# Check device manager or system information
```

3. Set up udev rules (Linux only):
```bash
# Create udev rules file
sudo nano /etc/udev/rules.d/99-sdr.rules

# Add rules for your devices
# RTL-SDR
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"

# HackRF
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="6089", GROUP="plugdev", MODE="0666"

# BladeRF
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="6066", GROUP="plugdev", MODE="0666"

# Airspy
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="60a1", GROUP="plugdev", MODE="0666"

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Troubleshooting

1. Device not detected:
   - Check USB connection
   - Verify driver installation
   - Check udev rules (Linux)
   - Try different USB port
   - Check device manager (Windows)

2. Permission issues (Linux):
   - Add user to plugdev group: `sudo usermod -a -G plugdev $USER`
   - Log out and log back in
   - Check udev rules

3. Driver issues:
   - Reinstall drivers
   - Check manufacturer's website for updates
   - Verify system compatibility

4. Python package issues:
   - Update pip: `pip install --upgrade pip`
   - Reinstall packages: `pip install -r requirements.txt --force-reinstall`
   - Check Python version compatibility

### Supported Hardware

- RTL-SDR (RTL2832U based dongles)
- HackRF One
- BladeRF
- USRP (B200, B210, X300, X310, N200, N210)
- LimeSDR
- Airspy (R2, Mini)
- SDRplay (RSP1, RSP1A, RSP2, RSPduo)
- PlutoSDR

### Notes

- Some SDR devices require specific firmware versions
- Check manufacturer's documentation for hardware-specific requirements
- Some features may require additional system libraries
- Performance may vary based on system specifications
- USB 3.0 ports recommended for high sample rates

## Installation

### System Dependencies

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install rtl-sdr
brew install hackrf
brew install bladerf
brew install soapysdr
```

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y rtl-sdr hackrf bladerf soapysdr
```

#### Windows
- Install Zadig for USB driver installation
- Install SDR drivers from respective manufacturer websites
- Install SoapySDR from https://github.com/pothosware/SoapySDR/wiki/BuildGuide

### Python Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ghostnet.git
cd ghostnet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Configure the system:
```bash
cp config/ghostnet_config.json config/local_config.json
# Edit local_config.json with your settings
```

## Project Structure

```
ghostnet/
├── src/
│   ├── core/           # Core signal processing and AI components
│   │   ├── signal_processor.py
│   │   └── feature_extractor.py
│   ├── hardware/       # SDR interface and hardware abstraction
│   │   └── sdr_interface.py
│   ├── mesh/           # Distributed receiver coordination
│   │   ├── receiver_mesh.py
│   │   └── triangulation.py
│   ├── models/         # AI models and training code
│   │   └── signal_classifier.py
│   └── web/            # Web interface and API
├── tests/              # Unit and integration tests
├── data/               # Training data and model storage
├── config/             # Configuration files
└── docs/              # Documentation
```

## Usage

1. Start the main GhostNet system:
```bash
python src/main.py --config config/local_config.json
```

2. Access the web interface:
```bash
python src/web/app.py
```

## Configuration

The system is configured through a JSON configuration file. Key settings include:

- Signal processing parameters (sample rate, FFT size)
- SDR device settings (frequency, gain, bandwidth)
- Mesh network configuration (node ID, position, port)
- Triangulation parameters (minimum receivers, max distance)
- Logging and storage settings
- Visualization options

See `config/ghostnet_config.json` for a complete configuration template.

## Signal Processing Pipeline

1. **RF Acquisition**
   - Wideband scanning using SDR hardware
   - Sample rate conversion and filtering
   - IQ data capture

2. **Feature Extraction**
   - Spectral analysis
   - Burst detection
   - Doppler shift analysis
   - Signal strength measurement

3. **Classification**
   - AI-based signal classification
   - Target type identification
   - Confidence scoring

4. **Triangulation**
   - Multi-receiver coordination
   - Time difference of arrival
   - Signal strength mapping
   - 3D position estimation

5. **Visualization**
   - Real-time heatmaps
   - Motion trails
   - Target tracking
   - System status monitoring

## Development

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation for API changes
- Use type hints for better code maintainability

## Security Notice

This system is designed for authorized use only. Users are responsible for complying with all applicable laws and regulations regarding RF monitoring and surveillance.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GNU Radio for SDR support
- PyTorch for AI capabilities
- RTL-SDR community for hardware support

## Docker Setup

### Prerequisites
- Docker Engine 20.10.0 or later
- Docker Compose 2.0.0 or later
- USB device access (for SDR hardware)
- Git

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/ghostnet.git
cd ghostnet

# Build and start the container
docker-compose up --build
```

### Docker Configuration

#### Base Image
The system uses a multi-stage build to minimize the final image size:
```dockerfile
# Base stage for building dependencies
FROM python:3.9-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    libsoapysdr0.8 \
    soapysdr-module-all \
    rtl-sdr \
    uhd-host \
    libiio0 \
    libairspy0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
```

#### Docker Compose Configuration
```yaml
version: '3.8'

services:
  ghostnet:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ghostnet
    restart: unless-stopped
    privileged: true  # Required for USB device access
    devices:
      - /dev/bus/usb:/dev/bus/usb  # USB device passthrough
    volumes:
      - ./config:/app/config  # Configuration files
      - ./data:/app/data      # Data storage
    ports:
      - "8000:8000"  # Web interface
      - "5000:5000"  # Mesh networking
    environment:
      - PYTHONUNBUFFERED=1
      - TZ=UTC
    networks:
      - ghostnet_net

networks:
  ghostnet_net:
    driver: bridge
```

### USB Device Access

#### Linux
1. Create udev rules for SDR devices:
```bash
# Create udev rules file
sudo nano /etc/udev/rules.d/99-sdr.rules

# Add rules for your devices
# RTL-SDR
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"

# HackRF
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="6089", GROUP="plugdev", MODE="0666"

# BladeRF
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="6066", GROUP="plugdev", MODE="0666"

# Airspy
SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="60a1", GROUP="plugdev", MODE="0666"

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

2. Add your user to the plugdev group:
```bash
sudo usermod -a -G plugdev $USER
```

#### macOS
1. Install Docker Desktop
2. Enable USB device sharing in Docker Desktop preferences
3. No additional configuration needed

#### Windows
1. Install Docker Desktop
2. Enable USB device sharing in Docker Desktop preferences
3. Install appropriate USB drivers for your SDR devices

### Running with Docker

1. Build and start the container:
```bash
docker-compose up --build
```

2. Run in detached mode:
```bash
docker-compose up -d
```

3. View logs:
```bash
docker-compose logs -f
```

4. Stop the container:
```bash
docker-compose down
```

### Docker Troubleshooting

1. USB Device Access Issues:
   - Verify udev rules (Linux)
   - Check USB device sharing in Docker Desktop (macOS/Windows)
   - Ensure container has privileged access
   - Check device permissions

2. Container Build Issues:
   - Clear Docker cache: `docker builder prune`
   - Rebuild without cache: `docker-compose build --no-cache`
   - Check system requirements

3. Runtime Issues:
   - Check container logs: `docker-compose logs -f`
   - Verify port mappings
   - Check volume mounts
   - Verify network configuration

4. Performance Issues:
   - Monitor container resources: `docker stats`
   - Check host system resources
   - Verify USB bandwidth allocation
   - Consider using USB 3.0 ports

### Docker Security Considerations

1. Container Security:
   - Use non-root user in container
   - Limit container capabilities
   - Regular security updates
   - Scan images for vulnerabilities

2. Network Security:
   - Use internal Docker network
   - Limit exposed ports
   - Implement proper authentication
   - Use HTTPS for web interface

3. Data Security:
   - Use named volumes for persistent data
   - Implement proper backup strategy
   - Encrypt sensitive data
   - Regular security audits

### Docker Development

1. Development Workflow:
```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up

# Run tests
docker-compose -f docker-compose.dev.yml run --rm ghostnet pytest

# Build production image
docker-compose build
```

2. Development Tools:
   - VS Code Remote Development
   - Docker Desktop
   - Docker CLI
   - Docker Compose

3. Debugging:
   - Attach to container: `docker attach ghostnet`
   - Execute shell: `docker-compose exec ghostnet bash`
   - View logs: `docker-compose logs -f`
   - Monitor resources: `docker stats` 