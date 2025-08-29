#!/usr/bin/env python3
"""
Script to install SDR packages that are not available on PyPI.
These packages need to be installed manually or via alternative methods.
"""

import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Installing {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ Successfully installed {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_sdr_packages():
    """Install SDR packages that are not available on PyPI."""
    print("Installing SDR packages that require manual installation...")
    print("=" * 60)
    
    # Check if we're on macOS
    if sys.platform == "darwin":
        print("Detected macOS - using Homebrew for system dependencies")
        
        # Install system dependencies via Homebrew
        brew_packages = [
            ("brew install soapysdr", "SoapySDR (system dependency)"),
            ("brew install sdrplay", "SDRplay (system dependency)"),
            ("brew install hackrf", "HackRF (system dependency)"),
            ("brew install uhd", "UHD (system dependency)"),
        ]
        
        for command, description in brew_packages:
            run_command(command, description)
    
    # Try to install Python packages
    pip_packages = [
        ("pip install soapysdr", "SoapySDR Python bindings"),
        ("pip install hackrf-python", "HackRF Python bindings"),
        ("pip install uhd-python", "UHD Python bindings"),
        ("pip install airspy", "Airspy Python bindings"),
        ("pip install sdrplay", "SDRplay Python bindings"),
    ]
    
    for command, description in pip_packages:
        run_command(command, description)
    
    print("\n" + "=" * 60)
    print("Installation complete!")
    print("\nNote: Some packages may not be available or may require additional setup.")
    print("Check the documentation for each SDR device for specific installation instructions.")

if __name__ == "__main__":
    install_sdr_packages()
