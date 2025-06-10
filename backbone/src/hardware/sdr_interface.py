"""
SDR Interface module for GhostNet.
Provides a unified interface for different SDR hardware (HackRF, RTL-SDR, BladeRF, USRP, LimeSDR, Airspy, SDRplay, PlutoSDR).
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any, cast
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
import threading
from queue import Queue
import psutil
import os
from scipy import signal
import requests
import aiohttp
import asyncio
from urllib.parse import urljoin
import json
import adi
from adi.rx import rx_core
from adi.tx import tx_core

# Try importing different SDR libraries
try:
    import pyrtlsdr
    RTL_SDR_AVAILABLE = True
except ImportError:
    RTL_SDR_AVAILABLE = False

# Initialize SoapySDR-related flags
SOAPY_SDR_AVAILABLE = False
SDRPLAY_AVAILABLE = False
AIRSPY_AVAILABLE = False
LIME_SDR_AVAILABLE = False
UHD_AVAILABLE = False

# SoapySDR is optional and only needed for certain devices
try:
    import SoapySDR
    SOAPY_SDR_AVAILABLE = True
except ImportError:
    logger.warning("SoapySDR not available. SDRplay, Airspy, LimeSDR, and USRP devices will not be supported.")

try:
    import adi
    PLUTOSDR_AVAILABLE = False
    try:
        # Import required submodules first
        from adi.rx import rx_core
        from adi.tx import tx_core
        
        # Now we can properly test PlutoSDR availability
        class PlutoTest(rx_core, tx_core):
            def _rx_init_channels(self): pass
            def _rx_buffered_data(self): pass
            def _tx_init_channels(self): pass
            def _tx_data(self, data): pass
            
        _ = PlutoTest("ip:192.168.2.1")
        PLUTOSDR_AVAILABLE = True
    except Exception as e:
        logger.debug(f"PlutoSDR not available: {str(e)}")
except ImportError:
    PLUTOSDR_AVAILABLE = False

@dataclass
class SDRConfig:
    """Configuration parameters for SDR devices."""
    center_freq: float  # Center frequency in Hz
    sample_rate: float  # Sample rate in Hz
    gain: float        # RF gain in dB
    bandwidth: float   # Bandwidth in Hz
    device_args: Dict[str, str]  # Device-specific arguments

@dataclass
class DeviceHealth:
    """Device health monitoring information."""
    temperature: float  # Device temperature in Celsius
    cpu_usage: float   # CPU usage percentage
    memory_usage: float  # Memory usage percentage
    usb_speed: str     # USB speed
    usb_power: float   # USB power consumption in watts
    error_count: int   # Number of errors encountered
    uptime: float      # Device uptime in seconds
    last_calibration: float  # Timestamp of last calibration
    calibration_status: str  # Calibration status
    signal_quality: float    # Signal quality metric (0-100)
    overflows: int          # Number of buffer overflows
    underruns: int         # Number of buffer underruns
    last_error: str        # Last error message

@dataclass
class CalibrationInfo:
    """Device calibration information."""
    dc_offset: Tuple[float, float]  # DC offset (I, Q)
    iq_imbalance: Tuple[float, float]  # IQ imbalance (gain, phase)
    temperature_cal: Dict[float, Dict[str, float]]  # Temperature calibration table
    frequency_cal: Dict[float, Dict[str, float]]  # Frequency calibration table
    last_calibration: float  # Timestamp of last calibration
    calibration_valid: bool  # Whether calibration is valid
    calibration_expiry: float  # Calibration expiry timestamp

class SDRDevice(ABC):
    """Abstract base class for SDR devices."""
    
    def __init__(self):
        """Initialize SDR device."""
        self.config = None
    
    @abstractmethod
    def configure(self, config: SDRConfig) -> bool:
        """Configure the SDR device."""
        pass
    
    @abstractmethod
    def start_streaming(self) -> bool:
        """Start streaming samples from the device."""
        pass
    
    @abstractmethod
    def stop_streaming(self) -> bool:
        """Stop streaming samples."""
        pass
    
    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the device."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, str]:
        """Get device information."""
        pass

    def extract_features(self, samples: np.ndarray) -> np.ndarray:
        """
        Extract features from IQ samples using FFT-based analysis.
        
        Args:
            samples: Complex IQ samples
            
        Returns:
            numpy array of extracted features
        """
        if len(samples) == 0:
            return np.array([])
            
        if self.config is None:
            logger.warning("Device not configured. Using default sample rate of 2 MHz.")
            sample_rate = 2e6
        else:
            sample_rate = self.config.sample_rate
            
        # Parameters for feature extraction
        fft_size = 1024
        hop_size = fft_size // 4  # 75% overlap
        n_mels = 64  # Number of mel bands
        
        # Compute spectrogram
        frequencies, times, Sxx = signal.spectrogram(
            samples,
            fs=sample_rate,
            nperseg=fft_size,
            noverlap=fft_size - hop_size,
            window='hann',
            return_onesided=True
        )
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Extract features
        features = []
        
        # 1. Spectral statistics
        features.extend([
            np.mean(Sxx_db, axis=1),  # Mean power in each frequency bin
            np.std(Sxx_db, axis=1),   # Standard deviation of power
            np.max(Sxx_db, axis=1),   # Maximum power
            np.min(Sxx_db, axis=1)    # Minimum power
        ])
        
        # 2. Time-domain statistics
        features.extend([
            np.mean(np.abs(samples)),     # Mean amplitude
            np.std(np.abs(samples)),      # Amplitude standard deviation
            np.max(np.abs(samples)),      # Peak amplitude
            np.min(np.abs(samples))       # Minimum amplitude
        ])
        
        # 3. Spectral shape features
        # Compute spectral centroid
        centroid = np.sum(frequencies[:, np.newaxis] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        features.append(centroid)
        
        # Compute spectral bandwidth
        bandwidth = np.sqrt(np.sum((frequencies[:, np.newaxis] - centroid)**2 * Sxx, axis=0) / 
                          (np.sum(Sxx, axis=0) + 1e-10))
        features.append(bandwidth)
        
        # 4. Spectral flatness
        flatness = np.exp(np.mean(np.log(Sxx + 1e-10), axis=0)) / (np.mean(Sxx, axis=0) + 1e-10)
        features.append(flatness)
        
        # 5. Spectral rolloff (frequency below which 85% of power is contained)
        cumsum = np.cumsum(Sxx, axis=0)
        rolloff = frequencies[np.argmax(cumsum >= 0.85 * cumsum[-1], axis=0)]
        features.append(rolloff)
        
        # 6. Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(np.real(samples))))
        features.append([zero_crossings / len(samples)])
        
        # 7. Spectral contrast
        # Compute the difference between peaks and valleys in the spectrum
        peaks = np.max(Sxx_db, axis=1)
        valleys = np.min(Sxx_db, axis=1)
        contrast = peaks - valleys
        features.append(contrast)
        
        # Flatten and normalize features
        features = np.concatenate([f.flatten() for f in features])
        features = (features - np.mean(features)) / (np.std(features) + 1e-10)
        
        return features

    def sweep(self, start_freq: float, stop_freq: float, step_hz: float, dwell_time: float) -> Dict[float, np.ndarray]:
        """
        Perform a frequency sweep.
        
        Args:
            start_freq: Starting frequency in Hz
            stop_freq: Stopping frequency in Hz
            step_hz: Frequency step size in Hz
            dwell_time: Time to dwell at each frequency in seconds
            
        Returns:
            Dictionary mapping frequencies to sample arrays
        """
        raise NotImplementedError("Frequency sweep not implemented for this device")

    def run_self_test(self) -> Dict[str, Any]:
        """
        Run self-test diagnostics on the device.
        
        Returns:
            Dictionary containing test results and diagnostics
        """
        raise NotImplementedError("Self-test not implemented for this device")

class RTLSDRDevice(SDRDevice):
    """RTL-SDR device implementation."""
    
    def __init__(self, device_index: int = 0):
        """Initialize RTL-SDR device."""
        if not RTL_SDR_AVAILABLE:
            raise ImportError("pyrtlsdr not available")
        
        self.device_index = device_index
        self.sdr = None
        self.is_streaming = False
        self.sample_queue = Queue(maxsize=1000)
        self._stream_thread = None
        self.error_count = 0
        self.last_error = ""
        
    def configure(self, config: SDRConfig) -> bool:
        """Configure RTL-SDR device."""
        try:
            self.sdr = pyrtlsdr.RtlSdr(device_index=self.device_index)
            self.sdr.center_freq = config.center_freq
            self.sdr.sample_rate = config.sample_rate
            self.sdr.gain = config.gain
            self.sdr.bandwidth = config.bandwidth
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Failed to configure RTL-SDR: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def start_streaming(self) -> bool:
        """Start streaming samples from RTL-SDR."""
        if not self.sdr:
            return False
        
        try:
            self.is_streaming = True
            self._stream_thread = threading.Thread(target=self._stream_samples)
            self._stream_thread.daemon = True
            self._stream_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start RTL-SDR streaming: {str(e)}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming samples."""
        self.is_streaming = False
        if self._stream_thread:
            self._stream_thread.join()
        return True
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the queue."""
        try:
            samples = self.sample_queue.get(timeout=1.0)
            return samples
        except:
            return None
    
    def get_device_info(self) -> Dict[str, str]:
        """Get RTL-SDR device information."""
        if not self.sdr:
            return {}
        
        return {
            'type': 'RTL-SDR',
            'index': str(self.device_index),
            'center_freq': f"{self.sdr.center_freq/1e6:.2f} MHz",
            'sample_rate': f"{self.sdr.sample_rate/1e6:.2f} MS/s",
            'gain': f"{self.sdr.gain:.1f} dB"
        }
    
    def _stream_samples(self):
        """Internal method to stream samples from RTL-SDR."""
        while self.is_streaming:
            try:
                if self.sdr is None:
                    self.error_count += 1
                    self.last_error = "SDR device is not initialized"
                    logger.error(self.last_error)
                    break
                    
                samples = self.sdr.read_samples(1024)
                if not self.sample_queue.full():
                    self.sample_queue.put(samples)
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Error reading samples: {str(e)}"
                logger.error(self.last_error)
                time.sleep(0.1)

    def sweep(self, start_freq: float, stop_freq: float, step_hz: float, dwell_time: float) -> Dict[float, np.ndarray]:
        """
        Perform a frequency sweep using RTL-SDR.
        
        Args:
            start_freq: Starting frequency in Hz
            stop_freq: Stopping frequency in Hz
            step_hz: Frequency step size in Hz
            dwell_time: Time to dwell at each frequency in seconds
            
        Returns:
            Dictionary mapping frequencies to sample arrays
        """
        if not self.sdr:
            self.error_count += 1
            self.last_error = "RTL-SDR device not initialized"
            logger.error(self.last_error)
            return {}
            
        try:
            # Store original configuration
            original_freq = self.sdr.center_freq
            original_sr = self.sdr.sample_rate
            original_gain = self.sdr.gain
            original_bw = self.sdr.bandwidth
            
            # Calculate number of samples to collect at each frequency
            samples_per_freq = int(dwell_time * self.sdr.sample_rate)
            
            # Initialize results dictionary
            sweep_results = {}
            
            # Perform sweep
            for freq in np.arange(start_freq, stop_freq + step_hz, step_hz):
                try:
                    # Tune to frequency
                    self.sdr.center_freq = freq
                    
                    # Wait for PLL to settle
                    time.sleep(0.1)
                    
                    # Collect samples
                    samples = self.sdr.read_samples(samples_per_freq)
                    sweep_results[freq] = samples
                    
                except Exception as e:
                    self.error_count += 1
                    self.last_error = f"Error during sweep at {freq/1e6:.2f} MHz: {str(e)}"
                    logger.error(self.last_error)
                    continue
            
            # Restore original configuration
            self.sdr.center_freq = original_freq
            self.sdr.sample_rate = original_sr
            self.sdr.gain = original_gain
            self.sdr.bandwidth = original_bw
            
            return sweep_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Error during frequency sweep: {str(e)}"
            logger.error(self.last_error)
            return {}

    def run_self_test(self) -> Dict[str, Any]:
        """
        Run self-test diagnostics on RTL-SDR device.
        
        Returns:
            Dictionary containing test results and diagnostics
        """
        test_results = {
            'device_initialized': False,
            'configuration_valid': False,
            'sample_reading_ok': False,
            'buffer_states': {
                'queue_size': 0,
                'queue_full': False,
                'queue_empty': True
            },
            'error_count': self.error_count,
            'last_error': self.last_error,
            'test_duration': 0.0,
            'samples_collected': 0,
            'sample_rate_achieved': 0.0,
            'center_freq': 0.0,
            'gain': 0.0,
            'bandwidth': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Check device initialization
            if not self.sdr:
                self.error_count += 1
                self.last_error = "Device not initialized"
                logger.error(self.last_error)
                return test_results
                
            test_results['device_initialized'] = True
            
            # Check configuration
            if self.config:
                test_results['configuration_valid'] = True
                test_results['center_freq'] = self.sdr.center_freq
                test_results['gain'] = self.sdr.gain
                test_results['bandwidth'] = self.sdr.bandwidth
            
            # Test sample reading
            try:
                # Read a small number of samples
                samples = self.sdr.read_samples(1024)
                if samples is not None and len(samples) > 0:
                    test_results['sample_reading_ok'] = True
                    test_results['samples_collected'] = len(samples)
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Sample reading test failed: {str(e)}"
                logger.error(self.last_error)
            
            # Check buffer states
            test_results['buffer_states'].update({
                'queue_size': self.sample_queue.qsize(),
                'queue_full': self.sample_queue.full(),
                'queue_empty': self.sample_queue.empty()
            })
            
            # Calculate test duration and sample rate
            test_duration = time.time() - start_time
            test_results['test_duration'] = test_duration
            if test_duration > 0:
                test_results['sample_rate_achieved'] = test_results['samples_collected'] / test_duration
            
            return test_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Self-test failed: {str(e)}"
            logger.error(self.last_error)
            return test_results

class HackRFDevice(SDRDevice):
    """HackRF device implementation using SoapySDR."""
    
    def __init__(self, device_index: int = 0):
        """Initialize HackRF device."""
        if not SOAPY_SDR_AVAILABLE:
            raise ImportError("SoapySDR not available")
        
        self.device_index = device_index
        self.sdr = None
        self.is_streaming = False
        self.sample_queue = Queue(maxsize=1000)
        self._stream_thread = None
        self.error_count = 0
        self.last_error = ""
        
    def configure(self, config: SDRConfig) -> bool:
        """Configure HackRF device."""
        try:
            # Create SoapySDR device
            self.sdr = SoapySDR.Device(dict(driver="hackrf", device_index=self.device_index))
            
            # Configure device
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, config.sample_rate)
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, config.center_freq)
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, config.gain)
            self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, config.bandwidth)
            
            # Set up stream
            self.stream = self.sdr.setupStream(
                SoapySDR.SOAPY_SDR_RX,
                SoapySDR.SOAPY_SDR_CF32,
                [0]  # Channel 0
            )
            
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Failed to configure HackRF: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def start_streaming(self) -> bool:
        """Start streaming samples from HackRF."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.activateStream(self.stream)
            self.is_streaming = True
            self._stream_thread = threading.Thread(target=self._stream_samples)
            self._stream_thread.daemon = True
            self._stream_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start HackRF streaming: {str(e)}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming samples."""
        self.is_streaming = False
        if self._stream_thread:
            self._stream_thread.join()
        if self.sdr:
            try:
                self.sdr.deactivateStream(self.stream)
                self.sdr.closeStream(self.stream)
            except Exception as e:
                logger.error(f"Error closing HackRF stream: {str(e)}")
        return True
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the queue."""
        try:
            samples = self.sample_queue.get(timeout=1.0)
            return samples
        except:
            return None
    
    def get_device_info(self) -> Dict[str, str]:
        """Get HackRF device information."""
        if not self.sdr:
            return {}
        
        try:
            return {
                'type': 'HackRF',
                'index': str(self.device_index),
                'center_freq': f"{self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)/1e6:.2f} MHz",
                'sample_rate': f"{self.sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)/1e6:.2f} MS/s",
                'gain': f"{self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0):.1f} dB",
                'bandwidth': f"{self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)/1e6:.2f} MHz"
            }
        except Exception as e:
            logger.error(f"Error getting HackRF info: {str(e)}")
            return {}
    
    def _stream_samples(self):
        """Internal method to stream samples from HackRF."""
        if self.sdr is None:
            self.error_count += 1
            self.last_error = "HackRF device is not initialized"
            logger.error(self.last_error)
            return
            
        buffer = np.zeros(1024, dtype=np.complex64)
        while self.is_streaming:
            try:
                # Read samples
                ret = self.sdr.readStream(
                    self.stream,
                    [buffer],
                    len(buffer),
                    timeoutUs=1000000  # 1 second timeout
                )
                
                if ret.ret > 0:  # Check if samples were read
                    if not self.sample_queue.full():
                        self.sample_queue.put(buffer.copy())
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Error reading HackRF samples: {str(e)}"
                logger.error(self.last_error)
                time.sleep(0.1)

    def run_self_test(self) -> Dict[str, Any]:
        """
        Run self-test diagnostics on HackRF device.
        
        Returns:
            Dictionary containing test results and diagnostics
        """
        test_results = {
            'device_initialized': False,
            'configuration_valid': False,
            'sample_reading_ok': False,
            'buffer_states': {
                'queue_size': 0,
                'queue_full': False,
                'queue_empty': True
            },
            'error_count': self.error_count,
            'last_error': self.last_error,
            'test_duration': 0.0,
            'samples_collected': 0,
            'sample_rate_achieved': 0.0,
            'center_freq': 0.0,
            'gain': 0.0,
            'bandwidth': 0.0,
            'stream_active': False,
            'usb_speed': 'Unknown',
            'temperature': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Check device initialization
            if not self.sdr:
                self.error_count += 1
                self.last_error = "Device not initialized"
                logger.error(self.last_error)
                return test_results
                
            test_results['device_initialized'] = True
            
            # Check configuration
            if self.config:
                test_results['configuration_valid'] = True
                test_results['center_freq'] = self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
                test_results['gain'] = self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
                test_results['bandwidth'] = self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)
            
            # Get USB information
            try:
                test_results['usb_speed'] = self.sdr.getHardwareInfo().get('usb_speed', 'Unknown')
            except:
                pass
            
            # Get temperature
            try:
                test_results['temperature'] = self.sdr.readSensor("temperature")
            except:
                pass
            
            # Test sample reading
            try:
                # Create temporary stream for testing
                stream = self.sdr.setupStream(
                    SoapySDR.SOAPY_SDR_RX,
                    SoapySDR.SOAPY_SDR_CF32,
                    [0]
                )
                
                # Activate stream
                self.sdr.activateStream(stream)
                test_results['stream_active'] = True
                
                # Read samples
                buffer = np.zeros(1024, dtype=np.complex64)
                ret = self.sdr.readStream(stream, [buffer], len(buffer), timeoutUs=1000000)
                
                if ret.ret > 0:
                    test_results['sample_reading_ok'] = True
                    test_results['samples_collected'] = ret.ret
                
                # Clean up stream
                self.sdr.deactivateStream(stream)
                self.sdr.closeStream(stream)
                
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Sample reading test failed: {str(e)}"
                logger.error(self.last_error)
            
            # Check buffer states
            test_results['buffer_states'].update({
                'queue_size': self.sample_queue.qsize(),
                'queue_full': self.sample_queue.full(),
                'queue_empty': self.sample_queue.empty()
            })
            
            # Calculate test duration and sample rate
            test_duration = time.time() - start_time
            test_results['test_duration'] = test_duration
            if test_duration > 0:
                test_results['sample_rate_achieved'] = test_results['samples_collected'] / test_duration
            
            return test_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Self-test failed: {str(e)}"
            logger.error(self.last_error)
            return test_results

    def sweep(self, start_freq: float, stop_freq: float, step_hz: float, dwell_time: float) -> Dict[float, np.ndarray]:
        """
        Perform a frequency sweep using HackRF.
        
        Args:
            start_freq: Starting frequency in Hz
            stop_freq: Stopping frequency in Hz
            step_hz: Frequency step size in Hz
            dwell_time: Time to dwell at each frequency in seconds
            
        Returns:
            Dictionary mapping frequencies to sample arrays
        """
        if not self.sdr:
            self.error_count += 1
            self.last_error = "HackRF device not initialized"
            logger.error(self.last_error)
            return {}
            
        try:
            # Store original configuration
            original_freq = self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
            original_sr = self.sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
            original_gain = self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
            original_bw = self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)
            
            # Calculate number of samples to collect at each frequency
            samples_per_freq = int(dwell_time * original_sr)
            
            # Initialize results dictionary
            sweep_results = {}
            
            # Create temporary stream for sweep
            stream = self.sdr.setupStream(
                SoapySDR.SOAPY_SDR_RX,
                SoapySDR.SOAPY_SDR_CF32,
                [0]
            )
            
            try:
                # Activate stream
                self.sdr.activateStream(stream)
                
                # Perform sweep
                for freq in np.arange(start_freq, stop_freq + step_hz, step_hz):
                    try:
                        # Tune to frequency
                        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
                        
                        # Wait for PLL to settle
                        time.sleep(0.1)
                        
                        # Collect samples
                        buffer = np.zeros(samples_per_freq, dtype=np.complex64)
                        ret = self.sdr.readStream(stream, [buffer], len(buffer), timeoutUs=int(dwell_time * 1e6))
                        
                        if ret.ret > 0:
                            sweep_results[freq] = buffer[:ret.ret]
                        else:
                            self.error_count += 1
                            self.last_error = f"No samples collected at {freq/1e6:.2f} MHz"
                            logger.error(self.last_error)
                            
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Error during sweep at {freq/1e6:.2f} MHz: {str(e)}"
                        logger.error(self.last_error)
                        continue
                
            finally:
                # Clean up stream
                self.sdr.deactivateStream(stream)
                self.sdr.closeStream(stream)
            
            # Restore original configuration
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, original_freq)
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, original_sr)
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, original_gain)
            self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, original_bw)
            
            return sweep_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Error during frequency sweep: {str(e)}"
            logger.error(self.last_error)
            return {}

class BladeRFDevice(SDRDevice):
    """BladeRF device implementation using SoapySDR."""
    
    def __init__(self, device_index: int = 0):
        """Initialize BladeRF device."""
        if not SOAPY_SDR_AVAILABLE:
            raise ImportError("SoapySDR not available")
        
        self.device_index = device_index
        self.sdr = None
        self.is_streaming = False
        self.sample_queue = Queue(maxsize=1000)
        self._stream_thread = None
        self.error_count = 0
        self.last_error = ""
        
    def configure(self, config: SDRConfig) -> bool:
        """Configure BladeRF device."""
        try:
            # Create SoapySDR device
            self.sdr = SoapySDR.Device(dict(driver="bladerf", device_index=self.device_index))
            
            # Configure device
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, config.sample_rate)
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, config.center_freq)
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, config.gain)
            self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, config.bandwidth)
            
            # Set up stream
            self.stream = self.sdr.setupStream(
                SoapySDR.SOAPY_SDR_RX,
                SoapySDR.SOAPY_SDR_CF32,
                [0]  # Channel 0
            )
            
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Failed to configure BladeRF: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def start_streaming(self) -> bool:
        """Start streaming samples from BladeRF."""
        if not self.sdr:
            return False
        
        try:
            self.sdr.activateStream(self.stream)
            self.is_streaming = True
            self._stream_thread = threading.Thread(target=self._stream_samples)
            self._stream_thread.daemon = True
            self._stream_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start BladeRF streaming: {str(e)}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming samples."""
        self.is_streaming = False
        if self._stream_thread:
            self._stream_thread.join()
        if self.sdr:
            try:
                self.sdr.deactivateStream(self.stream)
                self.sdr.closeStream(self.stream)
            except Exception as e:
                logger.error(f"Error closing BladeRF stream: {str(e)}")
        return True
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the queue."""
        try:
            samples = self.sample_queue.get(timeout=1.0)
            return samples
        except:
            return None
    
    def get_device_info(self) -> Dict[str, str]:
        """Get BladeRF device information."""
        if not self.sdr:
            return {}
        
        try:
            return {
                'type': 'BladeRF',
                'index': str(self.device_index),
                'center_freq': f"{self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)/1e6:.2f} MHz",
                'sample_rate': f"{self.sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)/1e6:.2f} MS/s",
                'gain': f"{self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0):.1f} dB",
                'bandwidth': f"{self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)/1e6:.2f} MHz"
            }
        except Exception as e:
            logger.error(f"Error getting BladeRF info: {str(e)}")
            return {}
    
    def _stream_samples(self):
        """Internal method to stream samples from BladeRF."""
        buffer = np.zeros(1024, dtype=np.complex64)
        while self.is_streaming:
            try:
                if self.sdr is None:
                    self.error_count += 1
                    self.last_error = "BladeRF device is not initialized"
                    logger.error(self.last_error)
                    break
                    
                # Read samples
                ret = self.sdr.readStream(
                    self.stream,
                    [buffer],
                    len(buffer),
                    timeoutUs=1000000  # 1 second timeout
                )
                
                if ret.ret > 0:  # Check if samples were read
                    if not self.sample_queue.full():
                        self.sample_queue.put(buffer.copy())
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Error reading BladeRF samples: {str(e)}"
                logger.error(self.last_error)
                time.sleep(0.1)

    def sweep(self, start_freq: float, stop_freq: float, step_hz: float, dwell_time: float) -> Dict[float, np.ndarray]:
        """
        Perform a frequency sweep using BladeRF.
        
        Args:
            start_freq: Starting frequency in Hz
            stop_freq: Stopping frequency in Hz
            step_hz: Frequency step size in Hz
            dwell_time: Time to dwell at each frequency in seconds
            
        Returns:
            Dictionary mapping frequencies to sample arrays
        """
        if not self.sdr:
            self.error_count += 1
            self.last_error = "BladeRF device not initialized"
            logger.error(self.last_error)
            return {}
            
        try:
            # Store original configuration
            original_freq = self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
            original_sr = self.sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
            original_gain = self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
            original_bw = self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)
            
            # Calculate number of samples to collect at each frequency
            samples_per_freq = int(dwell_time * original_sr)
            
            # Initialize results dictionary
            sweep_results = {}
            
            # Create temporary stream for sweep
            stream = self.sdr.setupStream(
                SoapySDR.SOAPY_SDR_RX,
                SoapySDR.SOAPY_SDR_CF32,
                [0]
            )
            
            try:
                # Activate stream
                self.sdr.activateStream(stream)
                
                # Perform sweep
                for freq in np.arange(start_freq, stop_freq + step_hz, step_hz):
                    try:
                        # Tune to frequency
                        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
                        
                        # Wait for PLL to settle
                        time.sleep(0.1)
                        
                        # Collect samples
                        buffer = np.zeros(samples_per_freq, dtype=np.complex64)
                        ret = self.sdr.readStream(stream, [buffer], len(buffer), timeoutUs=int(dwell_time * 1e6))
                        
                        if ret.ret > 0:
                            sweep_results[freq] = buffer[:ret.ret]
                        else:
                            self.error_count += 1
                            self.last_error = f"No samples collected at {freq/1e6:.2f} MHz"
                            logger.error(self.last_error)
                            
                    except Exception as e:
                        self.error_count += 1
                        self.last_error = f"Error during sweep at {freq/1e6:.2f} MHz: {str(e)}"
                        logger.error(self.last_error)
                        continue
                
            finally:
                # Clean up stream
                self.sdr.deactivateStream(stream)
                self.sdr.closeStream(stream)
            
            # Restore original configuration
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, original_freq)
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, original_sr)
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, original_gain)
            self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, original_bw)
            
            return sweep_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Error during frequency sweep: {str(e)}"
            logger.error(self.last_error)
            return {}

    def run_self_test(self) -> Dict[str, Any]:
        """
        Run self-test diagnostics on the device.
        
        Returns:
            Dictionary containing test results and diagnostics
        """
        test_results = {
            'device_initialized': False,
            'configuration_valid': False,
            'sample_reading_ok': False,
            'buffer_states': {
                'queue_size': 0,
                'queue_full': False,
                'queue_empty': True
            },
            'error_count': self.error_count,
            'last_error': self.last_error,
            'test_duration': 0.0,
            'samples_collected': 0,
            'sample_rate_achieved': 0.0,
            'center_freq': 0.0,
            'gain': 0.0,
            'bandwidth': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Check device initialization
            if not self.sdr:
                self.error_count += 1
                self.last_error = "Device not initialized"
                logger.error(self.last_error)
                return test_results
                
            test_results['device_initialized'] = True
            
            # Check configuration
            if self.config:
                test_results['configuration_valid'] = True
                test_results['center_freq'] = self.sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
                test_results['gain'] = self.sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)
                test_results['bandwidth'] = self.sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0)
            
            # Test sample reading
            try:
                # Read a small number of samples
                samples = self.sdr.read_samples(1024)
                if samples is not None and len(samples) > 0:
                    test_results['sample_reading_ok'] = True
                    test_results['samples_collected'] = len(samples)
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Sample reading test failed: {str(e)}"
                logger.error(self.last_error)
            
            # Check buffer states
            test_results['buffer_states'].update({
                'queue_size': self.sample_queue.qsize(),
                'queue_full': self.sample_queue.full(),
                'queue_empty': self.sample_queue.empty()
            })
            
            # Calculate test duration and sample rate
            test_duration = time.time() - start_time
            test_results['test_duration'] = test_duration
            if test_duration > 0:
                test_results['sample_rate_achieved'] = test_results['samples_collected'] / test_duration
            
            return test_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Self-test failed: {str(e)}"
            logger.error(self.last_error)
            return test_results

class PlutoSDRDevice(SDRDevice):
    """PlutoSDR device implementation using adi library."""
    
    def __init__(self, device_index: int = 0):
        """Initialize PlutoSDR device."""
        super().__init__()
        if not PLUTOSDR_AVAILABLE:
            raise ImportError("adi library not available")
        
        self.device_index = device_index
        self.sdr: Optional[Any] = None
        self.is_streaming = False
        self.sample_queue = Queue(maxsize=1000)
        self._stream_thread = None
        self.error_count = 0
        self.last_error = ""
        
    def configure(self, config: SDRConfig) -> bool:
        """Configure PlutoSDR device."""
        try:
            # Create PlutoSDR device
            try:
                # The adi library's implementation is concrete, but type hints are incomplete
                # We use Any to bypass type checking since we know the implementation is valid
                self.sdr = adi.Pluto(uri=f"ip:pluto.local")  # type: ignore
            except (AttributeError, TypeError) as e:
                self.error_count += 1
                self.last_error = f"Failed to create PlutoSDR device: {str(e)}"
                logger.error(self.last_error)
                return False
            
            if not self.sdr:
                self.error_count += 1
                self.last_error = "Failed to create PlutoSDR device"
                raise RuntimeError(self.last_error)
            
            # Configure device
            self.sdr.sample_rate = int(config.sample_rate)
            self.sdr.rx_lo = int(config.center_freq)
            self.sdr.rx_rf_bandwidth = int(config.bandwidth)
            self.sdr.gain_control_mode = 'manual'
            self.sdr.rx_hardwaregain = int(config.gain)
            
            # Store config for feature extraction
            self.config = config
            
            # Set buffer size
            self.sdr.rx_buffer_size = 1024
            
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Failed to configure PlutoSDR: {str(e)}"
            logger.error(self.last_error)
            return False
    
    def start_streaming(self) -> bool:
        """Start streaming samples from PlutoSDR."""
        if not self.sdr:
            return False
        
        try:
            self.is_streaming = True
            self._stream_thread = threading.Thread(target=self._stream_samples)
            self._stream_thread.daemon = True
            self._stream_thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start PlutoSDR streaming: {str(e)}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming samples."""
        self.is_streaming = False
        if self._stream_thread:
            self._stream_thread.join()
        return True
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples from the queue."""
        try:
            samples = self.sample_queue.get(timeout=1.0)
            return samples
        except:
            return None
    
    def get_device_info(self) -> Dict[str, str]:
        """Get PlutoSDR device information."""
        if not self.sdr:
            return {}
        
        try:
            # Convert numeric values to float before division
            rx_lo = float(self.sdr.rx_lo)
            sample_rate = float(self.sdr.sample_rate)
            rx_rf_bandwidth = float(self.sdr.rx_rf_bandwidth)
            
            return {
                'type': 'PlutoSDR',
                'index': str(self.device_index),
                'center_freq': f"{rx_lo/1e6:.2f} MHz",
                'sample_rate': f"{sample_rate/1e6:.2f} MS/s",
                'gain': f"{self.sdr.rx_hardwaregain:.1f} dB",
                'bandwidth': f"{rx_rf_bandwidth/1e6:.2f} MHz",
                'buffer_size': str(self.sdr.rx_buffer_size),
                'gain_control': self.sdr.gain_control_mode
            }
        except Exception as e:
            logger.error(f"Error getting PlutoSDR info: {str(e)}")
            return {}
    
    def _stream_samples(self):
        """Internal method to stream samples from PlutoSDR."""
        if self.sdr is None:
            self.error_count += 1
            self.last_error = "PlutoSDR device is not initialized"
            logger.error(self.last_error)
            return
            
        while self.is_streaming:
            try:
                # Read samples
                samples = self.sdr.rx()
                
                if not self.sample_queue.full():
                    self.sample_queue.put(samples)
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Error reading PlutoSDR samples: {str(e)}"
                logger.error(self.last_error)
                time.sleep(0.1)

    def run_self_test(self) -> Dict[str, Any]:
        """
        Run self-test diagnostics on PlutoSDR device.
        
        Returns:
            Dictionary containing test results and diagnostics
        """
        test_results = {
            'device_initialized': False,
            'configuration_valid': False,
            'sample_reading_ok': False,
            'buffer_states': {
                'queue_size': 0,
                'queue_full': False,
                'queue_empty': True
            },
            'error_count': self.error_count,
            'last_error': self.last_error,
            'test_duration': 0.0,
            'samples_collected': 0,
            'sample_rate_achieved': 0.0,
            'center_freq': 0.0,
            'gain': 0.0,
            'bandwidth': 0.0,
            'temperature': 0.0,
            'quadrature_tracking': False,
            'dc_tracking': False
        }
        
        start_time = time.time()
        
        try:
            # Check device initialization
            if not self.sdr:
                self.error_count += 1
                self.last_error = "Device not initialized"
                logger.error(self.last_error)
                return test_results
                
            test_results['device_initialized'] = True
            
            # Check configuration
            if self.config:
                test_results['configuration_valid'] = True
                test_results['center_freq'] = self.sdr.rx_lo
                test_results['gain'] = self.sdr.rx_hardwaregain
                test_results['bandwidth'] = self.sdr.rx_rf_bandwidth
            
            # Get temperature
            try:
                test_results['temperature'] = self.sdr.temperature
            except:
                pass
            
            # Check tracking features
            try:
                test_results['quadrature_tracking'] = self.sdr.quadrature_tracking_enabled
                test_results['dc_tracking'] = self.sdr.bb_dc_tracking_enabled
            except:
                pass
            
            # Test sample reading
            try:
                # Read a small number of samples
                samples = self.sdr.rx()
                if samples is not None and len(samples) > 0:
                    test_results['sample_reading_ok'] = True
                    test_results['samples_collected'] = len(samples)
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Sample reading test failed: {str(e)}"
                logger.error(self.last_error)
            
            # Check buffer states
            test_results['buffer_states'].update({
                'queue_size': self.sample_queue.qsize(),
                'queue_full': self.sample_queue.full(),
                'queue_empty': self.sample_queue.empty()
            })
            
            # Calculate test duration and sample rate
            test_duration = time.time() - start_time
            test_results['test_duration'] = test_duration
            if test_duration > 0:
                test_results['sample_rate_achieved'] = test_results['samples_collected'] / test_duration
            
            return test_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Self-test failed: {str(e)}"
            logger.error(self.last_error)
            return test_results

    def sweep(self, start_freq: float, stop_freq: float, step_hz: float, dwell_time: float) -> Dict[float, np.ndarray]:
        """
        Perform a frequency sweep using PlutoSDR.
        
        Args:
            start_freq: Starting frequency in Hz
            stop_freq: Stopping frequency in Hz
            step_hz: Frequency step size in Hz
            dwell_time: Time to dwell at each frequency in seconds
            
        Returns:
            Dictionary mapping frequencies to sample arrays
        """
        if not self.sdr:
            self.error_count += 1
            self.last_error = "PlutoSDR device not initialized"
            logger.error(self.last_error)
            return {}
            
        try:
            # Store original configuration
            original_freq = self.sdr.rx_lo
            original_sr = self.sdr.sample_rate
            original_gain = self.sdr.rx_hardwaregain
            original_bw = self.sdr.rx_rf_bandwidth
            
            # Calculate number of samples to collect at each frequency
            samples_per_freq = int(dwell_time * original_sr)
            
            # Initialize results dictionary
            sweep_results = {}
            
            # Perform sweep
            for freq in np.arange(start_freq, stop_freq + step_hz, step_hz):
                try:
                    # Tune to frequency
                    self.sdr.rx_lo = freq
                    
                    # Wait for PLL to settle
                    time.sleep(0.1)
                    
                    # Collect samples
                    samples = self.sdr.rx(samples_per_freq)
                    if samples is not None and len(samples) > 0:
                        sweep_results[freq] = samples
                    else:
                        self.error_count += 1
                        self.last_error = f"No samples collected at {freq/1e6:.2f} MHz"
                        logger.error(self.last_error)
                        
                except Exception as e:
                    self.error_count += 1
                    self.last_error = f"Error during sweep at {freq/1e6:.2f} MHz: {str(e)}"
                    logger.error(self.last_error)
                    continue
            
            # Restore original configuration
            self.sdr.rx_lo = original_freq
            self.sdr.sample_rate = original_sr
            self.sdr.rx_hardwaregain = original_gain
            self.sdr.rx_rf_bandwidth = original_bw
            
            return sweep_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Error during frequency sweep: {str(e)}"
            logger.error(self.last_error)
            return {}

class RemoteSDRDevice(SDRDevice):
    """Remote SDR device implementation that fetches samples over HTTP."""
    
    def __init__(self, base_url: str, device_id: str):
        """Initialize remote SDR device."""
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.device_id = device_id
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_streaming = False
        self.sample_queue = Queue(maxsize=1000)
        self._stream_thread = None
        self.error_count = 0
        self.last_error = ""
        
    def configure(self, config: SDRConfig) -> bool:
        """Configure remote SDR device."""
        try:
            # Store config for feature extraction
            self.config = config
            
            # Configure remote device
            response = requests.post(
                urljoin(self.base_url, f"/api/devices/{self.device_id}/configure"),
                json={
                    "center_freq": config.center_freq,
                    "sample_rate": config.sample_rate,
                    "gain": config.gain,
                    "bandwidth": config.bandwidth,
                    "device_args": config.device_args
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Failed to configure remote SDR device: {str(e)}"
            logger.error(self.last_error)
            return False
    
    async def _fetch_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Fetch samples from remote device asynchronously."""
        if self.session is None:
            self.error_count += 1
            self.last_error = "HTTP session not initialized"
            logger.error(self.last_error)
            return None
            
        try:
            async with self.session.get(
                urljoin(self.base_url, f"/api/devices/{self.device_id}/samples"),
                params={"num_samples": num_samples},
                timeout=10
            ) as response:
                if response.status == 200:
                    # Read binary data
                    data = await response.read()
                    
                    # Convert to numpy array
                    # Assuming data is in complex64 format (8 bytes per sample)
                    samples = np.frombuffer(data, dtype=np.complex64)
                    return samples
                else:
                    self.error_count += 1
                    self.last_error = f"Error fetching samples: {response.status}"
                    logger.error(self.last_error)
                    return None
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Error in async sample fetch: {str(e)}"
            logger.error(self.last_error)
            return None
    
    def _stream_samples(self):
        """Internal method to stream samples from remote device."""
        if self.session is None:
            self.error_count += 1
            self.last_error = "HTTP session not initialized"
            logger.error(self.last_error)
            return
            
        while self.is_streaming:
            try:
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Fetch samples
                samples = loop.run_until_complete(self._fetch_samples(1024))
                
                if samples is not None and not self.sample_queue.full():
                    self.sample_queue.put(samples)
                
                loop.close()
                
                # Small delay to prevent overwhelming the server
                time.sleep(0.01)
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Error in remote sample streaming: {str(e)}"
                logger.error(self.last_error)
                time.sleep(0.1)

    def run_self_test(self) -> Dict[str, Any]:
        """
        Run self-test diagnostics on remote SDR device.
        
        Returns:
            Dictionary containing test results and diagnostics
        """
        test_results = {
            'device_initialized': False,
            'configuration_valid': False,
            'sample_reading_ok': False,
            'buffer_states': {
                'queue_size': 0,
                'queue_full': False,
                'queue_empty': True
            },
            'error_count': self.error_count,
            'last_error': self.last_error,
            'test_duration': 0.0,
            'samples_collected': 0,
            'sample_rate_achieved': 0.0,
            'center_freq': 0.0,
            'gain': 0.0,
            'bandwidth': 0.0,
            'connection_ok': False,
            'stream_endpoint_ok': False,
            'response_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Check device initialization
            if not self.session:
                self.error_count += 1
                self.last_error = "HTTP session not initialized"
                logger.error(self.last_error)
                return test_results
                
            test_results['device_initialized'] = True
            
            # Check configuration
            if self.config:
                test_results['configuration_valid'] = True
                test_results['center_freq'] = self.config.center_freq
                test_results['gain'] = self.config.gain
                test_results['bandwidth'] = self.config.bandwidth
            
            # Test connection
            try:
                # Test basic endpoint
                async def test_connection():
                    if not self.session:
                        return False
                    async with self.session.get(
                        urljoin(self.base_url, f"/api/devices/{self.device_id}/info"),
                        timeout=5
                    ) as response:
                        return response.status == 200
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                connection_ok = loop.run_until_complete(test_connection())
                loop.close()
                
                test_results['connection_ok'] = connection_ok
                
                if connection_ok and self.session:
                    # Test streaming endpoint
                    async def test_stream():
                        if not self.session:
                            return False
                        async with self.session.get(
                            urljoin(self.base_url, f"/api/devices/{self.device_id}/stream"),
                            timeout=5
                        ) as response:
                            if response.status == 200:
                                data = await response.read()
                                return len(data) > 0
                            return False
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    stream_ok = loop.run_until_complete(test_stream())
                    loop.close()
                    
                    test_results['stream_endpoint_ok'] = stream_ok
                    
                    if stream_ok:
                        test_results['sample_reading_ok'] = True
                        test_results['samples_collected'] = 1024  # Assuming we got a full buffer
                
            except Exception as e:
                self.error_count += 1
                self.last_error = f"Connection test failed: {str(e)}"
                logger.error(self.last_error)
            
            # Check buffer states
            test_results['buffer_states'].update({
                'queue_size': self.sample_queue.qsize(),
                'queue_full': self.sample_queue.full(),
                'queue_empty': self.sample_queue.empty()
            })
            
            # Calculate test duration and response time
            test_duration = time.time() - start_time
            test_results['test_duration'] = test_duration
            test_results['response_time'] = test_duration  # Approximate response time
            
            return test_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Self-test failed: {str(e)}"
            logger.error(self.last_error)
            return test_results

    def sweep(self, start_freq: float, stop_freq: float, step_hz: float, dwell_time: float) -> Dict[float, np.ndarray]:
        """
        Perform a frequency sweep using remote SDR device.
        
        Args:
            start_freq: Starting frequency in Hz
            stop_freq: Stopping frequency in Hz
            step_hz: Frequency step size in Hz
            dwell_time: Time to dwell at each frequency in seconds
            
        Returns:
            Dictionary mapping frequencies to sample arrays
        """
        if not self.session:
            self.error_count += 1
            self.last_error = "HTTP session not initialized"
            logger.error(self.last_error)
            return {}
            
        try:
            # Store original configuration
            original_freq = self.config.center_freq if self.config else 0
            original_sr = self.config.sample_rate if self.config else 0
            original_gain = self.config.gain if self.config else 0
            original_bw = self.config.bandwidth if self.config else 0
            
            # Calculate number of samples to collect at each frequency
            samples_per_freq = int(dwell_time * original_sr)
            
            # Initialize results dictionary
            sweep_results = {}
            
            # Perform sweep
            for freq in np.arange(start_freq, stop_freq + step_hz, step_hz):
                try:
                    # Update device configuration
                    async def update_config():
                        if not self.session:
                            return False
                        async with self.session.post(
                            urljoin(self.base_url, f"/api/devices/{self.device_id}/config"),
                            json={
                                'center_freq': freq,
                                'sample_rate': original_sr,
                                'gain': original_gain,
                                'bandwidth': original_bw
                            },
                            timeout=5
                        ) as response:
                            return response.status == 200
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    config_ok = loop.run_until_complete(update_config())
                    loop.close()
                    
                    if not config_ok:
                        self.error_count += 1
                        self.last_error = f"Failed to update configuration at {freq/1e6:.2f} MHz"
                        logger.error(self.last_error)
                        continue
                    
                    # Wait for PLL to settle
                    time.sleep(0.1)
                    
                    # Collect samples
                    async def collect_samples():
                        if not self.session:
                            return None
                        async with self.session.get(
                            urljoin(self.base_url, f"/api/devices/{self.device_id}/stream"),
                            params={'samples': samples_per_freq},
                            timeout=int(dwell_time * 1.5)  # Add 50% margin
                        ) as response:
                            if response.status == 200:
                                data = await response.read()
                                return np.frombuffer(data, dtype=np.complex64)
                            return None
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    samples = loop.run_until_complete(collect_samples())
                    loop.close()
                    
                    if samples is not None and len(samples) > 0:
                        sweep_results[freq] = samples
                    else:
                        self.error_count += 1
                        self.last_error = f"No samples collected at {freq/1e6:.2f} MHz"
                        logger.error(self.last_error)
                        
                except Exception as e:
                    self.error_count += 1
                    self.last_error = f"Error during sweep at {freq/1e6:.2f} MHz: {str(e)}"
                    logger.error(self.last_error)
                    continue
            
            # Restore original configuration
            async def restore_config():
                if not self.session:
                    return False
                async with self.session.post(
                    urljoin(self.base_url, f"/api/devices/{self.device_id}/config"),
                    json={
                        'center_freq': original_freq,
                        'sample_rate': original_sr,
                        'gain': original_gain,
                        'bandwidth': original_bw
                    },
                    timeout=5
                ) as response:
                    return response.status == 200
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(restore_config())
            loop.close()
            
            return sweep_results
            
        except Exception as e:
            self.error_count += 1
            self.last_error = f"Error during frequency sweep: {str(e)}"
            logger.error(self.last_error)
            return {}

class SDRFactory:
    """Factory class for creating SDR device instances."""
    
    SUPPORTED_DEVICES = {
        'rtlsdr': 'RTL-SDR',
        'hackrf': 'HackRF',
        'bladerf': 'BladeRF',
        'usrp': 'USRP',
        'limesdr': 'LimeSDR',
        'airspy': 'Airspy',
        'sdrplay': 'SDRplay',
        'plutosdr': 'PlutoSDR'
    }
    
    # Device-specific parameters
    DEVICE_PARAMS = {
        'airspy': {
            'min_freq': 24e6,
            'max_freq': 1800e6,
            'max_sample_rate': 10e6,
            'max_gain': 21.0,
            'supported_bandwidths': [2.5e6, 5e6, 10e6],
            'supported_modes': ['linearity', 'sensitivity'],
            'has_bias_tee': True,
            'has_lna': True,
            'has_mixer': True,
            'has_vga': True,
            'has_agc': True,
            'has_direct_sampling': False,
            'has_temperature_sensor': True,
            'has_clock_source': False,
            'has_antenna_selection': True,
            'max_channels': 1,
            'supported_formats': ['CF32'],
            'usb_speed': 'USB 2.0',
            'power_consumption': '2.5W',
            'operating_temperature': '-20°C to +70°C'
        },
        'sdrplay': {
            'min_freq': 1e3,
            'max_freq': 2000e6,
            'max_sample_rate': 10e6,
            'max_gain': 59.0,
            'supported_bandwidths': [0.2e6, 0.3e6, 0.6e6, 1.536e6, 5e6, 6e6, 7e6, 8e6],
            'supported_modes': ['linearity', 'sensitivity'],
            'has_bias_tee': True,
            'has_lna': True,
            'has_mixer': True,
            'has_vga': True,
            'has_agc': True,
            'has_direct_sampling': False,
            'has_temperature_sensor': True,
            'has_clock_source': False,
            'has_antenna_selection': True,
            'max_channels': 1,
            'supported_formats': ['CF32'],
            'usb_speed': 'USB 2.0',
            'power_consumption': '2.5W',
            'operating_temperature': '-20°C to +70°C',
            'has_if_gain': True,
            'has_rf_notch': True,
            'has_dab_notch': True,
            'has_am_notch': True,
            'has_fm_notch': True,
            'has_dc_notch': True
        },
        'plutosdr': {
            'min_freq': 325e6,
            'max_freq': 3800e6,
            'max_sample_rate': 61.44e6,
            'max_gain': 73.0,
            'supported_bandwidths': [0.2e6, 0.5e6, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 8e6, 10e6, 12e6, 15e6, 20e6, 25e6, 30e6, 40e6, 50e6, 60e6],
            'supported_modes': ['linearity', 'sensitivity'],
            'has_bias_tee': False,
            'has_lna': True,
            'has_mixer': True,
            'has_vga': True,
            'has_agc': True,
            'has_direct_sampling': False,
            'has_temperature_sensor': True,
            'has_clock_source': True,
            'has_antenna_selection': True,
            'max_channels': 2,
            'supported_formats': ['CF32', 'CS16', 'CS8'],
            'usb_speed': 'USB 2.0',
            'power_consumption': '2.5W',
            'operating_temperature': '-20°C to +70°C',
            'has_quadrature_tracking': True,
            'has_bb_dc_tracking': True,
            'has_rf_dc_tracking': True,
            'has_quadrature_correction': True,
            'has_bb_dc_correction': True,
            'has_rf_dc_correction': True,
            'has_quadrature_phase_correction': True,
            'has_quadrature_gain_correction': True,
            'has_bb_dc_offset_correction': True,
            'has_rf_dc_offset_correction': True
        }
    }
    
    class DeviceError(Exception):
        """Base exception for device-related errors."""
        pass
    
    class DeviceNotFoundError(DeviceError):
        """Exception raised when a device is not found."""
        pass
    
    class DeviceUnavailableError(DeviceError):
        """Exception raised when a device is unavailable."""
        pass
    
    class DeviceCapabilityError(DeviceError):
        """Exception raised when a device capability is not supported."""
        pass
    
    @staticmethod
    def detect_device_capabilities(device_type: str, device_index: int = 0) -> Dict[str, Any]:
        """
        Automatically detect device capabilities.
        
        Args:
            device_type: Type of SDR device
            device_index: Device index
            
        Returns:
            Dictionary containing detected device capabilities
            
        Raises:
            DeviceError: If device capabilities cannot be detected
        """
        try:
            # Get base capabilities from device-specific parameters
            capabilities = SDRFactory.get_device_specific_params(device_type).copy()
            
            if device_type.lower() == 'rtlsdr':
                if not RTL_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("RTL-SDR support not available")
                
                sdr = pyrtlsdr.RtlSdr(device_index=device_index)
                capabilities.update({
                    'detected_tuner': sdr.get_tuner_type() if hasattr(sdr, 'get_tuner_type') else 'Unknown',
                    'detected_gain_stages': sdr.get_tuner_gains() if hasattr(sdr, 'get_tuner_gains') else []
                })
            
            elif device_type.lower() in ['hackrf', 'bladerf', 'usrp', 'limesdr', 'airspy', 'sdrplay', 'plutosdr']:
                if not SOAPY_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("SoapySDR support not available")
                
                devices = SoapySDR.Device.enumerate(dict(driver=device_type.lower()))
                if device_index >= len(devices):
                    raise SDRFactory.DeviceNotFoundError(f"{device_type.upper()} device {device_index} not found")
                
                device = devices[device_index]
                sdr = SoapySDR.Device(dict(driver=device_type.lower(), device_index=device_index))
                
                # Get frequency range
                try:
                    freq_range = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
                    capabilities['min_freq'] = freq_range.minimum()
                    capabilities['max_freq'] = freq_range.maximum()
                except:
                    pass
                
                # Get sample rate range
                try:
                    rate_range = sdr.getSampleRateRange(SoapySDR.SOAPY_SDR_RX, 0)
                    capabilities['max_sample_rate'] = rate_range.maximum()
                except:
                    pass
                
                # Get gain range
                try:
                    gain_range = sdr.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
                    capabilities['max_gain'] = gain_range.maximum()
                except:
                    pass
                
                # Get supported bandwidths
                try:
                    bandwidth_range = sdr.getBandwidthRange(SoapySDR.SOAPY_SDR_RX, 0)
                    capabilities['supported_bandwidths'] = [bw for bw in bandwidth_range]
                except:
                    pass
                
                # Get supported antennas
                try:
                    capabilities['supported_antennas'] = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
                except:
                    pass
                
                # Get supported clock sources
                try:
                    capabilities['supported_clock_sources'] = sdr.listClockSources()
                except:
                    pass
                
                # Get supported formats
                try:
                    capabilities['supported_formats'] = sdr.getStreamFormats(SoapySDR.SOAPY_SDR_RX, 0)
                except:
                    pass
                
                # Get number of channels
                try:
                    capabilities['max_channels'] = len(sdr.listChannels(SoapySDR.SOAPY_SDR_RX))
                except:
                    pass
                
                # Add device-specific features
                if device_type.lower() == 'airspy':
                    try:
                        capabilities.update({
                            'has_bias_tee': sdr.hasBiasTee(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_lna': sdr.hasLNA(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_mixer': sdr.hasMixer(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_vga': sdr.hasVGA(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_agc': sdr.hasAGC(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_temperature_sensor': sdr.hasTemperatureSensor(),
                            'has_antenna_selection': len(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)) > 1
                        })
                    except:
                        pass
                
                elif device_type.lower() == 'sdrplay':
                    try:
                        capabilities.update({
                            'has_bias_tee': sdr.hasBiasTee(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_lna': sdr.hasLNA(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_mixer': sdr.hasMixer(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_vga': sdr.hasVGA(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_agc': sdr.hasAGC(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_temperature_sensor': sdr.hasTemperatureSensor(),
                            'has_antenna_selection': len(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)) > 1,
                            'has_if_gain': sdr.hasIFGain(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_rf_notch': sdr.hasRFNotch(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_dab_notch': sdr.hasDABNotch(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_am_notch': sdr.hasAMNotch(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_fm_notch': sdr.hasFMNotch(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_dc_notch': sdr.hasDCNotch(SoapySDR.SOAPY_SDR_RX, 0)
                        })
                    except:
                        pass
                
                elif device_type.lower() == 'plutosdr':
                    try:
                        capabilities.update({
                            'has_bias_tee': sdr.hasBiasTee(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_lna': sdr.hasLNA(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_mixer': sdr.hasMixer(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_vga': sdr.hasVGA(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_agc': sdr.hasAGC(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_temperature_sensor': sdr.hasTemperatureSensor(),
                            'has_antenna_selection': len(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)) > 1,
                            'has_quadrature_tracking': sdr.hasQuadratureTracking(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_bb_dc_tracking': sdr.hasBBDCTracking(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_rf_dc_tracking': sdr.hasRFDCTracking(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_quadrature_correction': sdr.hasQuadratureCorrection(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_bb_dc_correction': sdr.hasBBDCCorrection(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_rf_dc_correction': sdr.hasRFDCCorrection(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_quadrature_phase_correction': sdr.hasQuadraturePhaseCorrection(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_quadrature_gain_correction': sdr.hasQuadratureGainCorrection(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_bb_dc_offset_correction': sdr.hasBBDCOffsetCorrection(SoapySDR.SOAPY_SDR_RX, 0),
                            'has_rf_dc_offset_correction': sdr.hasRFDCOffsetCorrection(SoapySDR.SOAPY_SDR_RX, 0)
                        })
                    except:
                        pass
            
            return capabilities
            
        except SDRFactory.DeviceError:
            raise
        except Exception as e:
            raise SDRFactory.DeviceError(f"Error detecting device capabilities: {str(e)}")
    
    @staticmethod
    def get_detailed_device_info(device_type: str, device_index: int = 0) -> Dict[str, Any]:
        """
        Get detailed information about a specific device including serial numbers and hardware details.
        
        Args:
            device_type: Type of SDR device
            device_index: Device index
            
        Returns:
            Dictionary containing detailed device information
            
        Raises:
            DeviceError: If device information cannot be retrieved
        """
        try:
            # Detect device capabilities
            capabilities = SDRFactory.detect_device_capabilities(device_type, device_index)
            
            if device_type.lower() == 'rtlsdr':
                if not RTL_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("RTL-SDR support not available")
                
                try:
                    sdr = pyrtlsdr.RtlSdr(device_index=device_index)
                except Exception as e:
                    raise SDRFactory.DeviceNotFoundError(f"RTL-SDR device {device_index} not found: {str(e)}")
                
                return {
                    'type': 'RTL-SDR',
                    'index': device_index,
                    'name': f"RTL-SDR {device_index}",
                    'serial': sdr.get_serial_number() if hasattr(sdr, 'get_serial_number') else f"RTL-SDR-{device_index}",
                    'capabilities': capabilities,
                    'mobile_compatible': True,  # RTL-SDR is mobile compatible
                    'hardware_info': {
                        'tuner_type': sdr.get_tuner_type() if hasattr(sdr, 'get_tuner_type') else 'Unknown',
                        'tuner_gain_stages': sdr.get_tuner_gains() if hasattr(sdr, 'get_tuner_gains') else [],
                        'firmware_version': sdr.get_firmware_version() if hasattr(sdr, 'get_firmware_version') else 'Unknown',
                        'usb_strings': sdr.get_usb_strings() if hasattr(sdr, 'get_usb_strings') else [],
                        'device_manufacturer': sdr.get_device_manufacturer() if hasattr(sdr, 'get_device_manufacturer') else 'Unknown',
                        'device_product': sdr.get_device_product() if hasattr(sdr, 'get_device_product') else 'Unknown',
                        'device_serial': sdr.get_device_serial() if hasattr(sdr, 'get_device_serial') else 'Unknown'
                    },
                    'current_settings': {
                        'center_freq': sdr.center_freq,
                        'sample_rate': sdr.sample_rate,
                        'gain': sdr.gain,
                        'bandwidth': sdr.bandwidth,
                        'freq_correction': sdr.freq_correction if hasattr(sdr, 'freq_correction') else 0,
                        'direct_sampling': sdr.direct_sampling if hasattr(sdr, 'direct_sampling') else 'disabled'
                    }
                }
            
            elif device_type.lower() in ['hackrf', 'bladerf', 'usrp', 'limesdr', 'airspy', 'sdrplay', 'plutosdr']:
                if not SOAPY_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("SoapySDR support not available")
                
                try:
                    devices = SoapySDR.Device.enumerate(dict(driver=device_type.lower()))
                    if device_index >= len(devices):
                        raise SDRFactory.DeviceNotFoundError(f"{device_type.upper()} device {device_index} not found")
                    
                    device = devices[device_index]
                    sdr = SoapySDR.Device(dict(driver=device_type.lower(), device_index=device_index))
                except Exception as e:
                    raise SDRFactory.DeviceUnavailableError(f"Error accessing {device_type.upper()} device: {str(e)}")
                
                # Get all available device information
                info = {
                    'type': device_type.upper(),
                    'index': device_index,
                    'name': f"{device_type.upper()} {device_index}",
                    'serial': device.get('serial', f"{device_type.upper()}-{device_index}"),
                    'capabilities': capabilities,
                    'mobile_compatible': device_type.lower() == 'hackrf',  # Only HackRF is mobile compatible
                    'hardware_info': {
                        'label': device.get('label', ''),
                        'manufacturer': device.get('manufacturer', ''),
                        'product': device.get('product', ''),
                        'version': device.get('version', ''),
                        'usb_bus': device.get('usb_bus', ''),
                        'usb_addr': device.get('usb_addr', ''),
                        'driver': device.get('driver', '')
                    }
                }
                
                # Add device-specific information
                if device_type.lower() == 'hackrf':
                    info['hardware_info'].update({
                        'antenna': device.get('antenna', ''),
                        'clock_source': device.get('clock_source', ''),
                        'clock_rate': device.get('clock_rate', ''),
                        'fpga_version': device.get('fpga_version', ''),
                        'firmware_version': device.get('firmware_version', '')
                    })
                elif device_type.lower() == 'bladerf':
                    info['hardware_info'].update({
                        'fpga': device.get('fpga', ''),
                        'fpga_version': device.get('fpga_version', ''),
                        'firmware_version': device.get('firmware_version', ''),
                        'serial': device.get('serial', ''),
                        'usb_speed': device.get('usb_speed', ''),
                        'expansion': device.get('expansion', '')
                    })
                elif device_type.lower() == 'usrp':
                    info['hardware_info'].update({
                        'mboard': device.get('mboard', ''),
                        'daughterboard': device.get('daughterboard', ''),
                        'gps': device.get('gps', ''),
                        'ref_clock': device.get('ref_clock', ''),
                        'time_sync': device.get('time_sync', '')
                    })
                elif device_type.lower() == 'limesdr':
                    info['hardware_info'].update({
                        'chip': device.get('chip', ''),
                        'firmware_version': device.get('firmware_version', ''),
                        'hardware_version': device.get('hardware_version', ''),
                        'gateware_version': device.get('gateware_version', ''),
                        'gateware_target': device.get('gateware_target', '')
                    })
                elif device_type.lower() == 'airspy':
                    info['hardware_info'].update({
                        'temperature': device.get('temperature', ''),
                        'calibration': device.get('calibration', '')
                    })
                elif device_type.lower() == 'sdrplay':
                    info['hardware_info'].update({
                        'temperature': device.get('temperature', ''),
                        'calibration': device.get('calibration', '')
                    })
                elif device_type.lower() == 'plutosdr':
                    info['hardware_info'].update({
                        'temperature': device.get('temperature', ''),
                        'calibration': device.get('calibration', '')
                    })
                
                # Get current device settings
                try:
                    info['current_settings'] = {
                        'center_freq': sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0),
                        'sample_rate': sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0),
                        'gain': sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0),
                        'bandwidth': sdr.getBandwidth(SoapySDR.SOAPY_SDR_RX, 0),
                        'antenna': sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0),
                        'clock_source': sdr.getClockSource(0) if hasattr(sdr, 'getClockSource') else 'Unknown',
                        'clock_rate': sdr.getMasterClockRate() if hasattr(sdr, 'getMasterClockRate') else 0
                    }
                except Exception as e:
                    logger.error(f"Error getting device settings: {str(e)}")
                    info['current_settings'] = {}
                
                return info
            
            raise SDRFactory.DeviceCapabilityError(f"Unsupported device type: {device_type}")
            
        except SDRFactory.DeviceError:
            raise
        except Exception as e:
            raise SDRFactory.DeviceError(f"Unexpected error getting device info: {str(e)}")
    
    @staticmethod
    def list_all_devices() -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available devices with their detailed information.
        
        Returns:
            Dictionary mapping device types to lists of detailed device information
            
        Raises:
            DeviceError: If there is an error listing devices
        """
        devices = {
            'rtlsdr': [],
            'hackrf': [],
            'bladerf': [],
            'usrp': [],
            'limesdr': [],
            'airspy': [],
            'sdrplay': [],
            'plutosdr': []
        }
        
        try:
            # List RTL-SDR devices
            if RTL_SDR_AVAILABLE:
                try:
                    device_count = pyrtlsdr.RtlSdr.get_device_count()
                    for i in range(device_count):
                        try:
                            info = SDRFactory.get_detailed_device_info('rtlsdr', i)
                            if info:
                                devices['rtlsdr'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting RTL-SDR device {i} info: {str(e)}")
                except Exception as e:
                    logger.error(f"Error listing RTL-SDR devices: {str(e)}")
            
            # List SoapySDR devices
            if SOAPY_SDR_AVAILABLE:
                try:
                    # List HackRF devices
                    hackrf_devices = SoapySDR.Device.enumerate(dict(driver="hackrf"))
                    for i in range(len(hackrf_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('hackrf', i)
                            if info:
                                devices['hackrf'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting HackRF device {i} info: {str(e)}")
                    
                    # List BladeRF devices
                    bladerf_devices = SoapySDR.Device.enumerate(dict(driver="bladerf"))
                    for i in range(len(bladerf_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('bladerf', i)
                            if info:
                                devices['bladerf'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting BladeRF device {i} info: {str(e)}")
                    
                    # List USRP devices
                    usrp_devices = SoapySDR.Device.enumerate(dict(driver="usrp"))
                    for i in range(len(usrp_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('usrp', i)
                            if info:
                                devices['usrp'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting USRP device {i} info: {str(e)}")
                    
                    # List LimeSDR devices
                    limesdr_devices = SoapySDR.Device.enumerate(dict(driver="limesdr"))
                    for i in range(len(limesdr_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('limesdr', i)
                            if info:
                                devices['limesdr'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting LimeSDR device {i} info: {str(e)}")
                    
                    # List Airspy devices
                    airspy_devices = SoapySDR.Device.enumerate(dict(driver="airspy"))
                    for i in range(len(airspy_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('airspy', i)
                            if info:
                                devices['airspy'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting Airspy device {i} info: {str(e)}")
                    
                    # List SDRplay devices
                    sdrplay_devices = SoapySDR.Device.enumerate(dict(driver="sdrplay"))
                    for i in range(len(sdrplay_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('sdrplay', i)
                            if info:
                                devices['sdrplay'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting SDRplay device {i} info: {str(e)}")
                    
                    # List PlutoSDR devices
                    plutosdr_devices = SoapySDR.Device.enumerate(dict(driver="plutosdr"))
                    for i in range(len(plutosdr_devices)):
                        try:
                            info = SDRFactory.get_detailed_device_info('plutosdr', i)
                            if info:
                                devices['plutosdr'].append(info)
                        except SDRFactory.DeviceError as e:
                            logger.error(f"Error getting PlutoSDR device {i} info: {str(e)}")
                except Exception as e:
                    logger.error(f"Error listing SoapySDR devices: {str(e)}")
            
            return devices
            
        except Exception as e:
            raise SDRFactory.DeviceError(f"Error listing devices: {str(e)}")
    
    @staticmethod
    def get_device_health(device_type: str, device_index: int = 0) -> DeviceHealth:
        """
        Get device health monitoring information.
        
        Args:
            device_type: Type of SDR device
            device_index: Device index
            
        Returns:
            DeviceHealth object containing health monitoring information
            
        Raises:
            DeviceError: If health information cannot be retrieved
        """
        try:
            if device_type.lower() == 'rtlsdr':
                if not RTL_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("RTL-SDR support not available")
                
                sdr = pyrtlsdr.RtlSdr(device_index=device_index)
                process = psutil.Process(os.getpid())
                
                return DeviceHealth(
                    temperature=0.0,  # RTL-SDR doesn't have temperature sensor
                    cpu_usage=process.cpu_percent(),
                    memory_usage=process.memory_percent(),
                    usb_speed='USB 2.0',
                    usb_power=2.5,  # Typical power consumption
                    error_count=sdr.error_count if hasattr(sdr, 'error_count') else 0,
                    uptime=time.time() - process.create_time(),
                    last_calibration=0.0,
                    calibration_status='Not supported',
                    signal_quality=100.0,  # Would need to implement signal quality metric
                    overflows=0,
                    underruns=0,
                    last_error=sdr.last_error if hasattr(sdr, 'last_error') else ""
                )
            
            elif device_type.lower() in ['hackrf', 'bladerf', 'usrp', 'limesdr', 'airspy', 'sdrplay', 'plutosdr']:
                if not SOAPY_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("SoapySDR support not available")
                
                devices = SoapySDR.Device.enumerate(dict(driver=device_type.lower()))
                if device_index >= len(devices):
                    raise SDRFactory.DeviceNotFoundError(f"{device_type.upper()} device {device_index} not found")
                
                device = devices[device_index]
                sdr = SoapySDR.Device(dict(driver=device_type.lower(), device_index=device_index))
                process = psutil.Process(os.getpid())
                
                # Get temperature if available
                try:
                    temperature = sdr.readSensor("temperature")
                except:
                    temperature = 0.0
                
                # Get USB information
                try:
                    usb_speed = device.get('usb_speed', 'Unknown')
                    usb_power = float(device.get('usb_power', 0.0))
                except:
                    usb_speed = 'Unknown'
                    usb_power = 0.0
                
                # Get calibration information
                try:
                    last_calibration = float(device.get('last_calibration', 0.0))
                    calibration_status = device.get('calibration_status', 'Unknown')
                except:
                    last_calibration = 0.0
                    calibration_status = 'Unknown'
                
                # Get signal quality (placeholder implementation)
                signal_quality = 100.0  # Would need to implement actual signal quality metric
                
                # Get buffer statistics
                try:
                    overflows = int(device.get('overflows', 0))
                    underruns = int(device.get('underruns', 0))
                except:
                    overflows = 0
                    underruns = 0
                
                return DeviceHealth(
                    temperature=temperature,
                    cpu_usage=process.cpu_percent(),
                    memory_usage=process.memory_percent(),
                    usb_speed=usb_speed,
                    usb_power=usb_power,
                    error_count=sdr.error_count if hasattr(sdr, 'error_count') else 0,
                    uptime=time.time() - process.create_time(),
                    last_calibration=last_calibration,
                    calibration_status=calibration_status,
                    signal_quality=signal_quality,
                    overflows=overflows,
                    underruns=underruns,
                    last_error=sdr.last_error if hasattr(sdr, 'last_error') else ""
                )
            
            raise SDRFactory.DeviceCapabilityError(f"Unsupported device type: {device_type}")
            
        except SDRFactory.DeviceError:
            raise
        except Exception as e:
            raise SDRFactory.DeviceError(f"Error getting device health: {str(e)}")
    
    @staticmethod
    def get_calibration_info(device_type: str, device_index: int = 0) -> CalibrationInfo:
        """
        Get device calibration information.
        
        Args:
            device_type: Type of SDR device
            device_index: Device index
            
        Returns:
            CalibrationInfo object containing calibration information
            
        Raises:
            DeviceError: If calibration information cannot be retrieved
        """
        try:
            if device_type.lower() == 'rtlsdr':
                if not RTL_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("RTL-SDR support not available")
                
                sdr = pyrtlsdr.RtlSdr(device_index=device_index)
                
                return CalibrationInfo(
                    dc_offset=(0.0, 0.0),  # RTL-SDR doesn't support calibration
                    iq_imbalance=(1.0, 0.0),
                    temperature_cal={},
                    frequency_cal={},
                    last_calibration=0.0,
                    calibration_valid=False,
                    calibration_expiry=0.0
                )
            
            elif device_type.lower() in ['hackrf', 'bladerf', 'usrp', 'limesdr', 'airspy', 'sdrplay', 'plutosdr']:
                if not SOAPY_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("SoapySDR support not available")
                
                devices = SoapySDR.Device.enumerate(dict(driver=device_type.lower()))
                if device_index >= len(devices):
                    raise SDRFactory.DeviceNotFoundError(f"{device_type.upper()} device {device_index} not found")
                
                device = devices[device_index]
                sdr = SoapySDR.Device(dict(driver=device_type.lower(), device_index=device_index))
                
                # Get DC offset
                try:
                    dc_offset_i = sdr.getDCOffset(SoapySDR.SOAPY_SDR_RX, 0)[0]
                    dc_offset_q = sdr.getDCOffset(SoapySDR.SOAPY_SDR_RX, 0)[1]
                except:
                    dc_offset_i = 0.0
                    dc_offset_q = 0.0
                
                # Get IQ imbalance
                try:
                    iq_gain = sdr.getIQBalance(SoapySDR.SOAPY_SDR_RX, 0)[0]
                    iq_phase = sdr.getIQBalance(SoapySDR.SOAPY_SDR_RX, 0)[1]
                except:
                    iq_gain = 1.0
                    iq_phase = 0.0
                
                # Get temperature calibration
                try:
                    temperature_cal = sdr.getTemperatureCalibration()
                except:
                    temperature_cal = {}
                
                # Get frequency calibration
                try:
                    frequency_cal = sdr.getFrequencyCalibration()
                except:
                    frequency_cal = {}
                
                # Get calibration status
                try:
                    last_calibration = float(device.get('last_calibration', 0.0))
                    calibration_valid = device.get('calibration_valid', False)
                    calibration_expiry = float(device.get('calibration_expiry', 0.0))
                except:
                    last_calibration = 0.0
                    calibration_valid = False
                    calibration_expiry = 0.0
                
                return CalibrationInfo(
                    dc_offset=(dc_offset_i, dc_offset_q),
                    iq_imbalance=(iq_gain, iq_phase),
                    temperature_cal=temperature_cal,
                    frequency_cal=frequency_cal,
                    last_calibration=last_calibration,
                    calibration_valid=calibration_valid,
                    calibration_expiry=calibration_expiry
                )
            
            raise SDRFactory.DeviceCapabilityError(f"Unsupported device type: {device_type}")
            
        except SDRFactory.DeviceError:
            raise
        except Exception as e:
            raise SDRFactory.DeviceError(f"Error getting calibration info: {str(e)}")
    
    @staticmethod
    def perform_calibration(device_type: str, device_index: int = 0) -> bool:
        """
        Perform device calibration.
        
        Args:
            device_type: Type of SDR device
            device_index: Device index
            
        Returns:
            True if calibration was successful
            
        Raises:
            DeviceError: If calibration cannot be performed
        """
        try:
            if device_type.lower() == 'rtlsdr':
                if not RTL_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("RTL-SDR support not available")
                
                # RTL-SDR doesn't support calibration
                return False
            
            elif device_type.lower() in ['hackrf', 'bladerf', 'usrp', 'limesdr', 'airspy', 'sdrplay', 'plutosdr']:
                if not SOAPY_SDR_AVAILABLE:
                    raise SDRFactory.DeviceUnavailableError("SoapySDR support not available")
                
                devices = SoapySDR.Device.enumerate(dict(driver=device_type.lower()))
                if device_index >= len(devices):
                    raise SDRFactory.DeviceNotFoundError(f"{device_type.upper()} device {device_index} not found")
                
                device = devices[device_index]
                sdr = SoapySDR.Device(dict(driver=device_type.lower(), device_index=device_index))
                
                # Perform calibration
                try:
                    # DC offset calibration
                    sdr.setDCOffset(SoapySDR.SOAPY_SDR_RX, 0, (0.0, 0.0))
                    
                    # IQ imbalance calibration
                    sdr.setIQBalance(SoapySDR.SOAPY_SDR_RX, 0, (1.0, 0.0))
                    
                    # Temperature calibration
                    sdr.calibrateTemperature()
                    
                    # Frequency calibration
                    sdr.calibrateFrequency()
                    
                    # Update calibration timestamp
                    device['last_calibration'] = time.time()
                    device['calibration_valid'] = True
                    device['calibration_expiry'] = time.time() + 24 * 3600  # 24 hours
                    
                    return True
                except Exception as e:
                    logger.error(f"Error during calibration: {str(e)}")
                    return False
            
            raise SDRFactory.DeviceCapabilityError(f"Unsupported device type: {device_type}")
            
        except SDRFactory.DeviceError:
            raise
        except Exception as e:
            raise SDRFactory.DeviceError(f"Error performing calibration: {str(e)}")

    @staticmethod
    def get_device_specific_params(device_type: str) -> Dict[str, Any]:
        """
        Get device-specific parameters.
        
        Args:
            device_type: Type of SDR device
            
        Returns:
            Dictionary containing device-specific parameters
            
        Raises:
            DeviceCapabilityError: If device type is not supported
        """
        if device_type.lower() not in SDRFactory.DEVICE_PARAMS:
            raise SDRFactory.DeviceCapabilityError(f"Unsupported device type: {device_type}")
        
        return SDRFactory.DEVICE_PARAMS[device_type.lower()]

    @staticmethod
    def create_device(device_type: str, device_index: int = 0) -> Optional[SDRDevice]:
        """
        Create an instance of an SDR device.
        
        Args:
            device_type: Type of SDR device
            device_index: Device index
            
        Returns:
            SDRDevice instance or None if device type is not supported
        """
        if device_type.lower() == 'rtlsdr':
            return RTLSDRDevice(device_index)
        elif device_type.lower() == 'hackrf':
            return HackRFDevice(device_index)
        elif device_type.lower() == 'bladerf':
            return BladeRFDevice(device_index)
        elif device_type.lower() == 'plutosdr':
            if not PLUTOSDR_AVAILABLE:
                raise SDRFactory.DeviceUnavailableError("PlutoSDR support not available")
            return PlutoSDRDevice(device_index)
        elif device_type.lower() == 'usrp':
            if not SOAPY_SDR_AVAILABLE:
                raise SDRFactory.DeviceUnavailableError("USRP support not available")
            raise NotImplementedError("USRP device support not implemented yet")
        elif device_type.lower() == 'airspy':
            if not SOAPY_SDR_AVAILABLE:
                raise SDRFactory.DeviceUnavailableError("Airspy support not available")
            raise NotImplementedError("Airspy device support not implemented yet")
        elif device_type.lower() == 'sdrplay':
            if not SOAPY_SDR_AVAILABLE:
                raise SDRFactory.DeviceUnavailableError("SDRplay support not available")
            raise NotImplementedError("SDRplay device support not implemented yet")
        else:
            logger.warning(f"Device type '{device_type}' not supported yet.")
            return None

class SDRManager:
    """Manager class for handling multiple SDR devices."""
    
    def __init__(self):
        """Initialize SDR manager."""
        self.devices: Dict[str, SDRDevice] = {}
        self.configs: Dict[str, SDRConfig] = {}
    
    def add_device(self, device_id: str, device_type: str, config: SDRConfig) -> bool:
        """
        Add a new SDR device.
        
        Args:
            device_id: Unique identifier for the device
            device_type: Type of SDR device
            config: Device configuration
            
        Returns:
            True if device was added successfully
        """
        device = SDRFactory.create_device(device_type)
        if not device:
            return False
        
        if not device.configure(config):
            return False
        
        self.devices[device_id] = device
        self.configs[device_id] = config
        return True
    
    def remove_device(self, device_id: str) -> bool:
        """Remove an SDR device."""
        if device_id in self.devices:
            device = self.devices[device_id]
            device.stop_streaming()
            del self.devices[device_id]
            del self.configs[device_id]
            return True
        return False
    
    def start_all_devices(self) -> bool:
        """Start streaming from all devices."""
        success = True
        for device_id, device in self.devices.items():
            if not device.start_streaming():
                logger.error(f"Failed to start device {device_id}")
                success = False
        return success
    
    def stop_all_devices(self) -> bool:
        """Stop streaming from all devices."""
        success = True
        for device_id, device in self.devices.items():
            if not device.stop_streaming():
                logger.error(f"Failed to stop device {device_id}")
                success = False
        return success
    
    def get_device_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all devices."""
        return {
            device_id: device.get_device_info()
            for device_id, device in self.devices.items()
        } 