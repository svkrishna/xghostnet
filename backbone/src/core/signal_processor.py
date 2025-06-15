"""
Core signal processing module for GhostNet.
Handles RF signal acquisition, processing, and feature extraction.
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union, Any, Mapping
from dataclasses import dataclass
import pywt
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpectralFeatures:
    """Spectral features of a signal."""
    centroid: float
    bandwidth: float
    flatness: float
    rolloff: float
    kurtosis: float
    skewness: float
    entropy: float
    peaks: List[Tuple[float, float]]  # (frequency, magnitude)
    harmonics: List[Tuple[float, float]]  # (frequency, magnitude)

@dataclass
class TemporalFeatures:
    """Temporal features of a signal."""
    zero_crossing_rate: float
    rms: float
    crest_factor: float
    energy: float
    entropy: float
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    autocorrelation: np.ndarray

@dataclass
class ModulationFeatures:
    """Modulation-related features."""
    amplitude_mean: float
    amplitude_std: float
    phase_mean: float
    phase_std: float
    frequency_deviation: float
    modulation_index: float
    constellation_points: np.ndarray
    eye_diagram: np.ndarray
    symbol_rate: float

@dataclass
class StatisticalFeatures:
    """Statistical features of a signal."""
    mean: float
    std: float
    median: float
    percentiles: Mapping[str, float]
    range: float
    iqr: float
    skewness: float
    kurtosis: float
    entropy: float

class SignalProcessor:
    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048):
        """Initialize signal processor.
        
        Args:
            sample_rate: Sample rate in Hz
            fft_size: Size of FFT for spectral analysis
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = signal.windows.hann(fft_size)
        
    def extract_spectral_features(self, samples: np.ndarray) -> SpectralFeatures:
        """Extract spectral features from signal.
        
        Args:
            samples: Input signal samples
            
        Returns:
            SpectralFeatures object containing spectral features
        """
        # Apply window and compute FFT
        windowed = samples[:self.fft_size] * self.window
        spectrum = rfft(windowed)
        freqs = rfftfreq(self.fft_size, 1/self.sample_rate)
        magnitude = np.abs(np.asarray(spectrum))
        
        # Calculate spectral centroid
        magnitude_sum = np.sum(magnitude)
        if magnitude_sum > 0:
            centroid = float(np.sum(freqs * magnitude) / magnitude_sum)
        else:
            centroid = 0.0
        
        # Calculate bandwidth
        if magnitude_sum > 0:
            bandwidth = float(np.sqrt(np.sum((freqs - centroid)**2 * magnitude) / magnitude_sum))
        else:
            bandwidth = 0.0
        
        # Calculate spectral flatness
        if magnitude_sum > 0:
            flatness = float(np.exp(np.mean(np.log(magnitude + 1e-10))) / np.mean(magnitude))
        else:
            flatness = 0.0
        
        # Calculate spectral rolloff
        rolloff_threshold = 0.85 * np.sum(magnitude)
        rolloff_idx = np.where(np.cumsum(magnitude) >= rolloff_threshold)[0]
        rolloff = float(freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1])
        
        # Calculate spectral kurtosis and skewness
        kurtosis = float(stats.kurtosis(magnitude))
        skewness = float(stats.skew(magnitude))
        
        # Calculate spectral entropy
        normalized_magnitude = magnitude / np.sum(magnitude)
        entropy = float(-np.sum(normalized_magnitude * np.log2(normalized_magnitude + 1e-10)))
        
        # Find spectral peaks
        peaks, _ = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
        peak_freqs = freqs[peaks]
        peak_mags = magnitude[peaks]
        peaks = [(float(f), float(m)) for f, m in zip(peak_freqs, peak_mags)]
        
        # Find harmonics
        fundamental_freq = peak_freqs[np.argmax(peak_mags)]
        harmonics = []
        for i in range(2, 6):  # Look for up to 5th harmonic
            harmonic_freq = fundamental_freq * i
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            if magnitude[idx] > np.max(magnitude) * 0.1:
                harmonics.append((float(freqs[idx]), float(magnitude[idx])))
                
        return SpectralFeatures(
            centroid=centroid,
            bandwidth=bandwidth,
            flatness=flatness,
            rolloff=rolloff,
            kurtosis=kurtosis,
            skewness=skewness,
            entropy=entropy,
            peaks=peaks,
            harmonics=harmonics
        )
        
    def extract_temporal_features(self, samples: np.ndarray) -> TemporalFeatures:
        """Extract temporal features from signal.
        
        Args:
            samples: Input signal samples
            
        Returns:
            TemporalFeatures object containing temporal features
        """
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(samples)))
        zero_crossing_rate = float(zero_crossings / (len(samples) - 1))
        
        # Calculate RMS value
        rms = float(np.sqrt(np.mean(samples**2)))
        
        # Calculate crest factor
        crest_factor = float(np.max(np.abs(samples)) / rms)
        
        # Calculate energy
        energy = float(np.sum(samples**2))
        
        # Calculate entropy
        hist, _ = np.histogram(samples, bins=50, density=True)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        # Calculate basic statistics
        mean = float(np.mean(samples))
        variance = float(np.var(samples))
        skewness = float(stats.skew(samples))
        kurtosis = float(stats.kurtosis(samples))
        
        # Calculate autocorrelation
        autocorr = signal.correlate(samples, samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        return TemporalFeatures(
            zero_crossing_rate=zero_crossing_rate,
            rms=rms,
            crest_factor=crest_factor,
            energy=energy,
            entropy=entropy,
            mean=mean,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            autocorrelation=autocorr
        )
        
    def extract_modulation_features(self, samples: np.ndarray) -> ModulationFeatures:
        """Extract modulation-related features.
        
        Args:
            samples: Input signal samples
            
        Returns:
            ModulationFeatures object containing modulation features
        """
        # Calculate amplitude statistics
        amplitude = np.abs(samples)
        amplitude_mean = float(np.mean(amplitude))
        amplitude_std = float(np.std(amplitude))
        
        # Calculate phase statistics
        phase = np.angle(samples)
        phase_mean = float(np.mean(phase))
        phase_std = float(np.std(phase))
        
        # Calculate frequency deviation
        freq_dev = float(np.std(np.diff(phase)) * self.sample_rate / (2 * np.pi))
        
        # Calculate modulation index
        mod_index = float(freq_dev / (self.sample_rate / 2))
        
        # Generate constellation diagram
        normalized = samples / np.max(np.abs(samples))
        constellation = normalized[::100]  # Sample every 100th point
        
        # Generate eye diagram
        symbol_period = int(self.sample_rate / 1000)  # Assuming 1 kbps
        eye_diagram = np.zeros((symbol_period, len(samples) // symbol_period))
        for i in range(len(samples) // symbol_period):
            eye_diagram[:, i] = samples[i*symbol_period:(i+1)*symbol_period]
            
        # Estimate symbol rate
        symbol_rate = float(self.sample_rate / symbol_period)
        
        return ModulationFeatures(
            amplitude_mean=amplitude_mean,
            amplitude_std=amplitude_std,
            phase_mean=phase_mean,
            phase_std=phase_std,
            frequency_deviation=freq_dev,
            modulation_index=mod_index,
            constellation_points=constellation,
            eye_diagram=eye_diagram,
            symbol_rate=symbol_rate
        )
        
    def extract_statistical_features(self, samples: np.ndarray) -> StatisticalFeatures:
        """Extract statistical features from signal.
        
        Args:
            samples: Input signal samples
            
        Returns:
            StatisticalFeatures object containing statistical features
        """
        # Calculate basic statistics
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        median = float(np.median(samples))
        
        # Calculate percentiles
        percentiles = {
            '25': float(np.percentile(samples, 25)),
            '75': float(np.percentile(samples, 75)),
            '90': float(np.percentile(samples, 90)),
            '95': float(np.percentile(samples, 95)),
            '99': float(np.percentile(samples, 99))
        }
        
        # Calculate range and IQR
        range_val = float(np.max(samples) - np.min(samples))
        iqr = float(percentiles['75'] - percentiles['25'])
        
        # Calculate skewness and kurtosis
        skewness = float(stats.skew(samples))
        kurtosis = float(stats.kurtosis(samples))
        
        # Calculate entropy
        hist, _ = np.histogram(samples, bins=50, density=True)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        return StatisticalFeatures(
            mean=mean,
            std=std,
            median=median,
            percentiles=percentiles,
            range=range_val,
            iqr=iqr,
            skewness=skewness,
            kurtosis=kurtosis,
            entropy=entropy
        )
        
    def extract_all_features(self, samples: np.ndarray) -> Dict[str, Any]:
        """Extract all available features from signal.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Dictionary containing all extracted features
        """
        spectral = self.extract_spectral_features(samples)
        temporal = self.extract_temporal_features(samples)
        modulation = self.extract_modulation_features(samples)
        statistical = self.extract_statistical_features(samples)
        
        return {
            'spectral': spectral,
            'temporal': temporal,
            'modulation': modulation,
            'statistical': statistical
        }
        
    def detect_signal_characteristics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Detect signal characteristics and modulation type.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Dictionary containing detected characteristics
        """
        # Extract features
        features = self.extract_all_features(samples)
        
        # Detect modulation type based on features
        mod_features = features['modulation']
        spectral_features = features['spectral']
        
        # Initialize characteristics
        characteristics = {
            'modulation_type': 'unknown',
            'confidence': 0.0,
            'bandwidth': spectral_features.bandwidth,
            'symbol_rate': mod_features.symbol_rate,
            'is_digital': False,
            'is_analog': False,
            'is_constant_envelope': False,
            'is_linear_modulation': False
        }
        
        # Detect modulation type
        if mod_features.modulation_index > 0.5:
            characteristics['modulation_type'] = 'fm'
            characteristics['is_analog'] = True
            characteristics['confidence'] = 0.8
        elif spectral_features.bandwidth < 1000:
            characteristics['modulation_type'] = 'am'
            characteristics['is_analog'] = True
            characteristics['confidence'] = 0.7
        elif mod_features.amplitude_std < 0.1:
            characteristics['modulation_type'] = 'psk'
            characteristics['is_digital'] = True
            characteristics['is_constant_envelope'] = True
            characteristics['confidence'] = 0.9
        elif mod_features.amplitude_std > 0.5:
            characteristics['modulation_type'] = 'ask'
            characteristics['is_digital'] = True
            characteristics['is_linear_modulation'] = True
            characteristics['confidence'] = 0.8
            
        return characteristics
    
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
        return idx_90 * self.sample_rate / self.fft_size
    
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