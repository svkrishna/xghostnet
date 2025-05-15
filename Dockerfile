# Base stage for building dependencies
FROM python:3.9-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libsoapysdr-dev \
    librtlsdr-dev \
    libuhd-dev \
    libiio-dev \
    libairspy-dev \
    libhackrf-dev \
    libbladerf-dev \
    liblimesuite-dev \
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
    libhackrf0 \
    libbladerf2 \
    liblimesuite20.10-1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash ghostnet

# Create necessary directories
RUN mkdir -p /app/config /app/data && \
    chown -R ghostnet:ghostnet /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/

# Set working directory
WORKDIR /app

# Switch to non-root user
USER ghostnet

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Expose ports
EXPOSE 8000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["python", "src/main.py"] 