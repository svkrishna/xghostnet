"""
Core signal processing module for GhostNet.
Handles RF signal acquisition, processing, and feature extraction.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalProcessor:
    def __init__(self, sample_rate: float = 2e6, fft_size: int = 1024):
        """
        Initialize the signal processor with basic parameters.
        
        Args:
            sample_rate: Sampling rate in Hz
            fft_size: Size of FFT window for spectral analysis
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_resolution = sample_rate / fft_size
        
    def compute_spectrogram(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram of the input signal.
        
        Args:
            samples: Raw IQ samples
            
        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        frequencies, times, Sxx = signal.spectrogram(
            samples,
            fs=self.sample_rate,
            nperseg=self.fft_size,
            noverlap=self.fft_size // 2,
            window='hann'
        )
        return frequencies, times, Sxx
    
    def extract_features(self, samples: np.ndarray) -> Dict[str, float]:
        """
        Extract key features from the signal for classification.
        
        Args:
            samples: Raw IQ samples
            
        Returns:
            Dictionary of extracted features
        """
        # Compute power spectrum
        _, _, Sxx = self.compute_spectrogram(samples)
        
        # Calculate spectral entropy
        spectral_entropy = self._compute_spectral_entropy(Sxx)
        
        # Calculate signal bandwidth
        bandwidth = self._estimate_bandwidth(Sxx)
        
        # Calculate burst timing features
        burst_features = self._analyze_bursts(samples)
        
        # Combine all features
        features = {
            'spectral_entropy': spectral_entropy,
            'bandwidth': bandwidth,
            'burst_duration_mean': burst_features['duration_mean'],
            'burst_interval_mean': burst_features['interval_mean'],
            'peak_power': np.max(np.abs(samples)),
            'rms_power': np.sqrt(np.mean(np.abs(samples)**2))
        }
        
        return features
    
    def _compute_spectral_entropy(self, Sxx: np.ndarray) -> float:
        """Calculate spectral entropy of the signal."""
        # Normalize the power spectrum
        p = Sxx / np.sum(Sxx)
        # Calculate entropy
        return -np.sum(p * np.log2(p + 1e-10))
    
    def _estimate_bandwidth(self, Sxx: np.ndarray) -> float:
        """Estimate the signal bandwidth."""
        # Find the frequency range containing 90% of the signal power
        power = np.sum(Sxx, axis=1)
        cum_power = np.cumsum(power)
        total_power = cum_power[-1]
        
        # Find indices containing 90% of power
        idx_90 = np.where(cum_power >= 0.9 * total_power)[0][0]
        return idx_90 * self.freq_resolution
    
    def _analyze_bursts(self, samples: np.ndarray) -> Dict[str, float]:
        """Analyze burst timing characteristics."""
        # Simple threshold-based burst detection
        threshold = np.mean(np.abs(samples)) + 2 * np.std(np.abs(samples))
        burst_mask = np.abs(samples) > threshold
        
        # Find burst boundaries
        burst_starts = np.where(np.diff(burst_mask.astype(int)) == 1)[0]
        burst_ends = np.where(np.diff(burst_mask.astype(int)) == -1)[0]
        
        if len(burst_starts) == 0 or len(burst_ends) == 0:
            return {'duration_mean': 0.0, 'interval_mean': 0.0}
        
        # Calculate burst durations
        durations = (burst_ends - burst_starts) / self.sample_rate
        
        # Calculate intervals between bursts
        intervals = (burst_starts[1:] - burst_ends[:-1]) / self.sample_rate
        
        return {
            'duration_mean': float(np.mean(durations)),
            'interval_mean': float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        }
    
    def detect_doppler_shift(self, samples: np.ndarray, time_window: float = 0.1) -> Optional[float]:
        """
        Detect Doppler shift in the signal.
        
        Args:
            samples: Raw IQ samples
            time_window: Time window for analysis in seconds
            
        Returns:
            Estimated Doppler shift in Hz, or None if not detected
        """
        window_samples = int(time_window * self.sample_rate)
        if len(samples) < window_samples:
            return None
            
        # Use phase difference to estimate frequency shift
        phase_diff = np.diff(np.unwrap(np.angle(samples)))
        freq_shift = float(np.mean(phase_diff) * self.sample_rate / (2 * np.pi))
        
        return freq_shift
    
    def classify_signal(self, features: dict) -> str:
        """
        Classify the signal based on extracted features.
        
        Args:
            features: Dictionary of signal features
            
        Returns:
            Signal classification ('drone', 'vehicle', 'human', or 'unknown')
        """
        # This is a placeholder for the actual classification logic
        # Will be replaced with ML model inference
        return 'unknown' 