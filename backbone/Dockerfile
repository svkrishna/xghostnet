# Use Ubuntu as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    cmake \
    build-essential \
    libusb-1.0-0-dev \
    libfftw3-dev \
    libhackrf-dev \
    libairspy-dev \
    libairspyhf-dev \
    libbladerf-dev \
    libhackrf-dev \
    liblimesuite-dev \
    libsoapysdr-dev \
    libsdrplay-dev \
    libuhd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY backbone/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app/backbone/src

# Copy application code
COPY . .

# Run the application
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]