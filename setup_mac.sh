#!/bin/bash

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install required Homebrew packages
echo "Installing required packages..."
brew install \
    python3 \
    cmake \
    libusb \
    soapysdr \
    rtl-sdr \
    uhd \
    iio \
    airspy \
    hackrf \
    bladerf \
    limesdr

# Install Docker Desktop if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker Desktop..."
    brew install --cask docker
    echo "Please start Docker Desktop and complete the installation process."
    echo "Press any key to continue once Docker Desktop is running..."
    read -n 1
fi

# Create necessary directories
mkdir -p config data

# Create default config if it doesn't exist
if [ ! -f config/ghostnet_config.json ]; then
    echo "Creating default configuration..."
    cat > config/ghostnet_config.json << EOL
{
    "sdr": {
        "type": "auto",
        "sample_rate": 2000000,
        "center_freq": 915000000,
        "gain": 40,
        "bandwidth": 2000000
    },
    "mesh": {
        "port": 5000,
        "host": "0.0.0.0"
    },
    "web": {
        "port": 8000,
        "host": "0.0.0.0"
    }
}
EOL
fi

# Build and start the container
echo "Building and starting GhostNet container..."
docker-compose -f docker-compose.mac.yml up --build -d

# Check if container is running
if [ "$(docker ps -q -f name=ghostnet)" ]; then
    echo "GhostNet is now running!"
    echo "Web interface: http://localhost:8000"
    echo "Mesh networking port: 5000"
else
    echo "Error: Container failed to start. Check logs with: docker-compose -f docker-compose.mac.yml logs"
fi 