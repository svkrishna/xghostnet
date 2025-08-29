#!/bin/bash

# macOS-specific installation script for XGhostNet dependencies

echo "Installing XGhostNet dependencies on macOS..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Please install it first:"
    echo "https://brew.sh/"
    exit 1
fi

# Install system dependencies via Homebrew
echo "Installing system dependencies via Homebrew..."
brew install soapysdr sdrplay hackrf uhd

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install SoapySDR Python bindings
echo "Installing SoapySDR Python bindings..."
pip install soapysdr

# Install additional SDR libraries if needed
echo "Installing additional SDR libraries..."
pip install sdrplay

echo "Installation complete!"
echo ""
echo "Note: Some SDR libraries may require additional setup:"
echo "- For HackRF: brew install hackrf"
echo "- For UHD: brew install uhd"
echo "- For SDRplay: brew install sdrplay"
echo ""
echo "You may need to restart your terminal or reload your shell profile."
