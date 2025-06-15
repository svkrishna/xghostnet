"""
Advanced signal detection algorithms for GhostNet.
Implements various signal detection methods including energy detection,
matched filtering, cyclostationary detection, and machine learning-based detection.
"""

import numpy as np
from scipy import signal
from scipy.stats import chi2, kurtosis, entropy, skew
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pywt
from .signal_types import ModulationType, SignalFeatures, ClassificationResult, DetectionResult, SignalMetrics
from scipy.fft import rfft, rfftfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSignalDetector:
    def __init__(self, sample_rate: float = 2e6, fft_size: int = 1024):
        """
        Initialize the advanced signal detector.
        
        Args:
            sample_rate: Sampling rate in Hz
            fft_size: Size of FFT window for spectral analysis
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_resolution = sample_rate / fft_size
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        
    def energy_detection(self, 
                        samples: np.ndarray, 
                        threshold_db: float = -20,
                        window_size: int = 1024) -> DetectionResult:
        """
        Perform energy detection on the signal.
        
        Args:
            samples: Input signal samples
            threshold_db: Detection threshold in dB
            window_size: Size of the detection window
            
        Returns:
            DetectionResult object containing detection results
        """
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold_db / 10)
        
        # Calculate signal energy
        energy = np.abs(samples) ** 2
        
        # Apply moving average
        energy_smooth = np.convolve(energy, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
        
        # Calculate SNR
        noise_floor = np.mean(energy_smooth[energy_smooth < threshold_linear])
        signal_power = np.mean(energy_smooth[energy_smooth >= threshold_linear])
        snr = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0
        
        # Detect signal presence
        detected = bool(np.any(energy_smooth >= threshold_linear))
        
        # Calculate confidence based on SNR
        confidence = min(1.0, max(0.0, (snr + 20) / 40))  # Normalize to [0,1]
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=snr,
            features={
                'energy_mean': float(np.mean(energy)),
                'energy_std': float(np.std(energy)),
                'peak_energy': float(np.max(energy))
            },
            metadata={
                'threshold_db': threshold_db,
                'window_size': window_size
            }
        )
        
    def matched_filter_detection(self,
                               samples: np.ndarray,
                               template: np.ndarray,
                               threshold: float = 0.7) -> DetectionResult:
        """
        Perform matched filter detection using a known signal template.
        
        Args:
            samples: Input signal samples
            template: Known signal template
            threshold: Detection threshold (0-1)
            
        Returns:
            DetectionResult object containing detection results
        """
        # Normalize template
        template = template / np.linalg.norm(template)
        
        # Perform matched filtering
        correlation = np.correlate(samples, template, mode='valid')
        correlation = correlation / np.max(np.abs(correlation))
        
        # Detect peaks
        peaks, _ = signal.find_peaks(np.abs(correlation), 
                                   height=threshold,
                                   distance=len(template)//2)
        
        detected = bool(len(peaks) > 0)
        confidence = float(np.max(np.abs(correlation))) if detected else 0.0
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=self._calculate_snr(samples),
            features={
                'correlation_max': float(np.max(np.abs(correlation))),
                'correlation_mean': float(np.mean(np.abs(correlation))),
                'peak_count': len(peaks)
            },
            metadata={
                'template_length': len(template),
                'threshold': threshold
            }
        )
        
    def cyclostationary_detection(self,
                                samples: np.ndarray,
                                alpha: float = 0.0,
                                threshold: float = 0.1) -> DetectionResult:
        """
        Perform cyclostationary detection to identify periodic features.
        
        Args:
            samples: Input signal samples
            alpha: Cyclic frequency to detect
            threshold: Detection threshold
            
        Returns:
            DetectionResult object containing detection results
        """
        # Calculate cyclic autocorrelation
        N = len(samples)
        t = np.arange(N)
        cyclic_corr = np.zeros(N, dtype=complex)
        
        for tau in range(N):
            cyclic_corr[tau] = np.mean(samples * np.conj(samples) * 
                                     np.exp(-1j * 2 * np.pi * alpha * t))
            
        # Calculate cyclic spectrum
        cyclic_spectrum = np.fft.fft(cyclic_corr)
        
        # Detect peaks in cyclic spectrum
        peaks, _ = signal.find_peaks(np.abs(cyclic_spectrum),
                                   height=threshold,
                                   distance=10)
        
        detected = bool(len(peaks) > 0)
        confidence = float(np.max(np.abs(cyclic_spectrum))) if detected else 0.0
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=self._calculate_snr(samples),
            features={
                'cyclic_spectrum_max': float(np.max(np.abs(cyclic_spectrum))),
                'cyclic_spectrum_mean': float(np.mean(np.abs(cyclic_spectrum))),
                'peak_count': len(peaks)
            },
            metadata={
                'alpha': alpha,
                'threshold': threshold
            }
        )
        
    def _estimate_frequency_bandwidth(self, samples: np.ndarray) -> Tuple[float, float]:
        """Estimate signal frequency and bandwidth."""
        # Compute FFT
        samples_array = np.array(samples, dtype=np.float64)
        spectrum = np.fft.rfft(samples_array)
        freqs = np.fft.rfftfreq(len(samples_array), 1/self.sample_rate)
        magnitude = np.abs(spectrum)
        
        # Basic spectral features
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * magnitude) / np.sum(magnitude))
        
        # Temporal features
        zero_crossings = np.sum(np.diff(np.signbit(samples_array)))
        zero_crossing_rate = zero_crossings / len(samples_array)
        rms = np.sqrt(np.mean(samples_array**2))
        crest_factor = np.max(np.abs(samples_array)) / rms if rms > 0 else 0
        
        # Modulation detection
        modulation = self._detect_modulation_simple(samples_array)
        
        # SNR estimation
        noise_floor = np.percentile(magnitude, 10)
        signal_peak = np.max(magnitude)
        snr = 20 * np.log10(signal_peak / noise_floor) if noise_floor > 0 else 0
        
        return spectral_centroid, bandwidth
        
    def _calculate_snr(self, samples: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Estimate noise floor using median
        noise_floor = np.median(np.abs(samples))
        
        # Estimate signal power
        signal_power = np.mean(np.abs(samples) ** 2)
        
        # Calculate SNR
        snr = 10 * np.log10(signal_power / (noise_floor ** 2)) if noise_floor > 0 else 0
        
        return snr 

    def cfar_detection(self,
                      samples: np.ndarray,
                      guard_cells: int = 4,
                      training_cells: int = 16,
                      pfa: float = 1e-6) -> DetectionResult:
        """
        Perform Constant False Alarm Rate (CFAR) detection.
        
        Args:
            samples: Input signal samples
            guard_cells: Number of guard cells
            training_cells: Number of training cells
            pfa: Probability of false alarm
            
        Returns:
            DetectionResult object containing detection results
        """
        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(samples)) ** 2
        
        # Initialize detection array
        detections = np.zeros_like(spectrum, dtype=bool)
        
        # Perform CFAR detection
        for i in range(len(spectrum)):
            # Define training and guard regions
            train_left = max(0, i - guard_cells - training_cells)
            train_right = min(len(spectrum), i + guard_cells + training_cells)
            
            # Calculate noise level from training cells
            noise_level = np.mean(spectrum[train_left:train_right])
            
            # Calculate threshold
            threshold = noise_level * (-np.log(pfa))
            
            # Perform detection
            detections[i] = spectrum[i] > threshold
        
        detected = bool(np.any(detections))
        confidence = float(np.max(spectrum[detections]) / np.mean(spectrum)) if detected else 0.0
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=self._calculate_snr(samples),
            features={
                'cfar_threshold': float(threshold),
                'noise_level': float(noise_level),
                'detection_count': int(np.sum(detections))
            },
            metadata={
                'guard_cells': guard_cells,
                'training_cells': training_cells,
                'pfa': pfa
            }
        )
        
    def wavelet_detection(self,
                         samples: np.ndarray,
                         wavelet: str = 'morl',
                         scales: Optional[np.ndarray] = None) -> DetectionResult:
        """
        Perform wavelet-based signal detection.
        
        Args:
            samples: Input signal samples
            wavelet: Wavelet type
            scales: Wavelet scales to use
            
        Returns:
            DetectionResult object containing detection results
        """
        if scales is None:
            scales = np.arange(1, 128)
            
        # Perform continuous wavelet transform
        coeffs, freqs = pywt.cwt(samples, scales, wavelet)
        
        # Calculate energy at each scale
        energy = np.abs(coeffs) ** 2
        
        # Detect peaks in wavelet domain
        peaks = []
        for i in range(len(scales)):
            peaks.extend(signal.find_peaks(energy[i], height=np.mean(energy[i]) + 2*np.std(energy[i]))[0])
            
        detected = bool(len(peaks) > 0)
        confidence = float(np.max(energy) / np.mean(energy)) if detected else 0.0
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=self._calculate_snr(samples),
            features={
                'wavelet_energy_max': float(np.max(energy)),
                'wavelet_energy_mean': float(np.mean(energy)),
                'peak_count': len(peaks)
            },
            metadata={
                'wavelet': wavelet,
                'scales': scales.tolist()
            }
        )
        
    def kurtosis_detection(self,
                          samples: np.ndarray,
                          window_size: int = 1024,
                          threshold: float = 3.0) -> DetectionResult:
        """
        Perform kurtosis-based signal detection.
        
        Args:
            samples: Input signal samples
            window_size: Size of analysis window
            threshold: Detection threshold
            
        Returns:
            DetectionResult object containing detection results
        """
        # Calculate kurtosis in sliding windows
        kurt = np.array([kurtosis(np.abs(samples[i:i+window_size]))
                        for i in range(0, len(samples)-window_size, window_size//2)])
        
        # Detect peaks in kurtosis
        peaks, _ = signal.find_peaks(kurt, height=threshold)
        
        detected = bool(len(peaks) > 0)
        confidence = float(np.max(kurt) / threshold) if detected else 0.0
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=self._calculate_snr(samples),
            features={
                'kurtosis_max': float(np.max(kurt)),
                'kurtosis_mean': float(np.mean(kurt)),
                'peak_count': len(peaks)
            },
            metadata={
                'window_size': window_size,
                'threshold': threshold
            }
        )
        
    def entropy_detection(self,
                         samples: np.ndarray,
                         window_size: int = 1024,
                         threshold: float = 0.5) -> DetectionResult:
        """
        Perform entropy-based signal detection.
        
        Args:
            samples: Input signal samples
            window_size: Size of analysis window
            threshold: Detection threshold
            
        Returns:
            DetectionResult object containing detection results
        """
        # Calculate spectral entropy in sliding windows
        entropies = []
        for i in range(0, len(samples)-window_size, window_size//2):
            window = samples[i:i+window_size]
            spectrum = np.abs(np.fft.fft(window)) ** 2
            p = spectrum / np.sum(spectrum)
            entropies.append(entropy(p))
            
        entropies = np.array(entropies)
        
        # Detect regions with low entropy (indicating signal presence)
        detected = bool(np.any(entropies < threshold))
        confidence = float(1 - np.min(entropies)) if detected else 0.0
        
        # Estimate frequency and bandwidth
        if detected:
            freq, bw = self._estimate_frequency_bandwidth(samples)
        else:
            freq, bw = 0.0, 0.0
            
        return DetectionResult(
            detected=detected,
            confidence=confidence,
            frequency=freq,
            bandwidth=bw,
            snr=self._calculate_snr(samples),
            features={
                'entropy_min': float(np.min(entropies)),
                'entropy_mean': float(np.mean(entropies)),
                'entropy_std': float(np.std(entropies))
            },
            metadata={
                'window_size': window_size,
                'threshold': threshold
            }
        )
        
    def denoise_signal(self,
                      samples: np.ndarray,
                      method: str = 'wavelet',
                      **kwargs) -> np.ndarray:
        """
        Denoise the input signal.
        
        Args:
            samples: Input signal samples
            method: Denoising method ('wavelet' or 'spectral')
            **kwargs: Additional parameters for denoising
            
        Returns:
            Denoised signal samples
        """
        if method == 'wavelet':
            # Wavelet denoising
            coeffs = pywt.wavedec(samples, 'db4', level=4)
            threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(samples)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, 'db4')
        else:
            # Spectral denoising
            spectrum = np.fft.fft(samples)
            threshold = np.std(np.abs(spectrum)) * np.sqrt(2 * np.log(len(samples)))
            spectrum[np.abs(spectrum) < threshold] = 0
            return np.fft.ifft(spectrum)
            
    def extract_features(self, samples: np.ndarray) -> SignalFeatures:
        """Extract comprehensive signal features for classification."""
        # Convert to numpy array
        samples_array = np.array(samples, dtype=np.float64)
        
        # Compute FFT
        spectrum = np.fft.rfft(samples_array)
        freqs = np.fft.rfftfreq(len(samples_array), 1/self.sample_rate)
        magnitude = np.abs(spectrum)
        
        # Spectral features
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * magnitude) / np.sum(magnitude))
        spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)
        
        # Temporal features
        zero_crossings = np.sum(np.diff(np.signbit(samples_array)))
        zero_crossing_rate = zero_crossings / len(samples_array)
        rms = np.sqrt(np.mean(samples_array**2))
        crest_factor = np.max(np.abs(samples_array)) / rms if rms > 0 else 0
        
        # SNR estimation
        noise_floor = np.percentile(magnitude, 10)
        signal_peak = np.max(magnitude)
        snr = 20 * np.log10(signal_peak / noise_floor) if noise_floor > 0 else 0
        
        # Entropy calculation
        entropy = -np.sum((magnitude/np.sum(magnitude)) * np.log2(magnitude/np.sum(magnitude) + 1e-10))
        
        # Modulation detection
        modulation = self._detect_modulation_simple(samples_array)
        
        return SignalFeatures(
            frequency=float(spectral_centroid),
            bandwidth=float(bandwidth),
            modulation=modulation,
            snr=float(snr),
            spectral_centroid=float(spectral_centroid),
            spectral_flatness=float(spectral_flatness),
            zero_crossing_rate=float(zero_crossing_rate),
            crest_factor=float(crest_factor),
            rms=float(rms),
            entropy=float(entropy)
        )
        
    def _detect_modulation_simple(self, samples: np.ndarray) -> ModulationType:
        """
        Simple modulation detection from signal samples.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Detected modulation type
        """
        # Basic modulation detection logic
        if len(samples) < 1000:
            return ModulationType.UNKNOWN
            
        # Check for AM
        envelope = np.abs(samples)
        if np.std(envelope) / np.mean(envelope) > 0.5:
            return ModulationType.AM
            
        # Check for FM
        phase = np.unwrap(np.angle(samples))
        if np.std(np.diff(phase)) > 0.5:
            return ModulationType.FM
            
        # Check for PSK
        constellation = samples[::100]  # Sample every 100th point
        if len(np.unique(np.round(np.angle(constellation) / (np.pi/4)))) <= 4:
            return ModulationType.PSK
            
        return ModulationType.UNKNOWN
        
    def recognize_modulation(self, samples: np.ndarray) -> str:
        """
        Recognize the modulation type of the signal.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Detected modulation type
        """
        # Use the simpler modulation detection method
        modulation = self._detect_modulation_simple(samples)
        return modulation.value
        
    def calculate_signal_quality(self, samples: np.ndarray) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate various signal quality metrics.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Dictionary of quality metrics
        """
        # Calculate SNR
        snr = self._calculate_snr(samples)
        
        # Calculate EVM (Error Vector Magnitude)
        ideal_symbols = self._estimate_ideal_symbols(samples)
        evm = np.sqrt(np.mean(np.abs(samples - ideal_symbols)**2)) / np.sqrt(np.mean(np.abs(samples)**2))
        
        # Calculate BER estimate
        ber = self._estimate_ber(samples)
        
        # Calculate constellation metrics
        constellation_metrics = self._calculate_constellation_metrics(samples)
        
        return {
            'snr': snr,
            'evm': float(evm),
            'ber': float(ber),
            'constellation_metrics': constellation_metrics
        }
        
    def _estimate_ideal_symbols(self, samples: np.ndarray) -> np.ndarray:
        """Estimate ideal symbol locations."""
        # This is a simplified implementation
        # In practice, this would depend on the modulation scheme
        return np.round(samples / np.mean(np.abs(samples))) * np.mean(np.abs(samples))
        
    def _estimate_ber(self, samples: np.ndarray) -> float:
        """Estimate bit error rate."""
        # This is a simplified implementation
        # In practice, this would require knowledge of the transmitted bits
        snr = self._calculate_snr(samples)
        return 0.5 * (1 - np.sqrt(snr / (1 + snr)))
        
    def _calculate_constellation_metrics(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate constellation diagram metrics."""
        # Normalize samples
        normalized = samples / np.mean(np.abs(samples))
        
        # Calculate metrics
        return {
            'constellation_density': float(np.mean(np.abs(normalized))),
            'constellation_spread': float(np.std(np.abs(normalized))),
            'constellation_skewness': float(skew(np.abs(normalized))),
            'constellation_kurtosis': float(kurtosis(np.abs(normalized)))
        }

    def _detect_modulation(self, samples: np.ndarray, 
                         spectral_flatness: float, 
                         bandwidth: float) -> Tuple[ModulationType, float]:
        """
        Detect modulation type from signal samples.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Detected modulation type
        """
        # Basic modulation detection logic
        if len(samples) < 1000:
            return ModulationType.UNKNOWN, 0.0
            
        # Check for AM
        envelope = np.abs(samples)
        if np.std(envelope) / np.mean(envelope) > 0.5:
            return ModulationType.AM, 0.8
            
        # Check for FM
        phase = np.unwrap(np.angle(samples))
        if np.std(np.diff(phase)) > 0.5:
            return ModulationType.FM, 0.7
            
        # Check for PSK
        constellation = samples[::100]  # Sample every 100th point
        if len(np.unique(np.round(np.angle(constellation) / (np.pi/4)))) <= 4:
            return ModulationType.PSK, 0.8
            
        return ModulationType.UNKNOWN, 0.0

class SignalDetector:
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        """Initialize signal detector with real-time processing capabilities.
        
        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Size of processing buffer
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size, dtype=np.float64)
        self.buffer_index = 0
        self.noise_floor = -90.0  # Initial noise floor estimate in dB
        self.snr_threshold = 10.0  # SNR threshold for signal detection
        
        # Initialize Kalman filter
        self.kalman_filter = {
            'x': 0.0,  # State estimate
            'P': 1.0,  # Error covariance
            'Q': 0.001,  # Process noise covariance
            'R': 0.1    # Measurement noise covariance
        }
        
        # Initialize adaptive filter
        self.adaptive_filter = {
            'mu': 0.01,  # Step size
            'filter_length': 64,
            'weights': np.zeros(64, dtype=np.float64)
        }
        
        # Initialize modulation detection parameters
        self.modulation_thresholds = {
            'magnitude_std': 0.1,
            'phase_std': 0.5,
            'spectral_flatness': 0.8,
            'bandwidth_threshold': 1000
        }
        
    def process_samples(self, samples: np.ndarray) -> Tuple[bool, float, SignalMetrics]:
        """Process incoming samples in real-time.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Tuple of (signal_detected, snr, metrics)
        """
        # Update buffer
        self._update_buffer(samples)
        
        # Apply noise filtering
        filtered_samples = self._apply_noise_filtering(self.buffer)
        
        # Calculate signal metrics
        metrics = self.get_signal_quality(filtered_samples)
        
        # Update noise floor estimate
        self._update_noise_floor(filtered_samples)
        
        # Detect signal presence
        signal_detected = metrics.snr > self.snr_threshold
        
        return signal_detected, metrics.snr, metrics
        
    def _update_buffer(self, samples: np.ndarray):
        """Update circular buffer with new samples."""
        n_samples = len(samples)
        if n_samples >= self.buffer_size:
            # If we have more samples than buffer size, keep the most recent ones
            self.buffer = samples[-self.buffer_size:]
            self.buffer_index = 0
        else:
            # Otherwise, shift buffer and add new samples
            self.buffer = np.roll(self.buffer, -n_samples)
            self.buffer[self.buffer_size - n_samples:] = samples
            self.buffer_index = (self.buffer_index + n_samples) % self.buffer_size
            
    def _apply_noise_filtering(self, samples: np.ndarray) -> np.ndarray:
        """Apply multiple noise filtering techniques."""
        # Apply Kalman filter
        filtered = self._apply_kalman_filter(samples)
        
        # Apply adaptive filtering
        filtered = self._apply_adaptive_filter(filtered)
        
        # Apply spectral subtraction
        filtered = self._apply_spectral_subtraction(filtered)
        
        return filtered
        
    def _apply_kalman_filter(self, samples: np.ndarray) -> np.ndarray:
        """Apply Kalman filtering for noise reduction."""
        filtered = np.zeros_like(samples)
        
        for i, sample in enumerate(samples):
            # Predict
            x_pred = self.kalman_filter['x']
            P_pred = self.kalman_filter['P'] + self.kalman_filter['Q']
            
            # Update
            K = P_pred / (P_pred + self.kalman_filter['R'])  # Kalman gain
            self.kalman_filter['x'] = x_pred + K * (sample - x_pred)
            self.kalman_filter['P'] = (1 - K) * P_pred
            
            filtered[i] = self.kalman_filter['x']
            
        return filtered
        
    def _apply_adaptive_filter(self, samples: np.ndarray) -> np.ndarray:
        """Apply adaptive filtering for noise cancellation."""
        filtered = np.zeros_like(samples)
        weights = self.adaptive_filter['weights']
        mu = self.adaptive_filter['mu']
        
        for i in range(len(samples)):
            if i >= len(weights):
                # Apply filter
                filtered[i] = np.sum(weights * samples[i-len(weights):i])
                
                # Update weights using LMS algorithm
                error = samples[i] - filtered[i]
                weights += mu * error * samples[i-len(weights):i]
                
        return filtered
        
    def _apply_spectral_subtraction(self, samples: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction for noise reduction."""
        # Compute FFT
        fft = np.fft.rfft(samples)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Estimate noise spectrum
        noise_spectrum = np.mean(magnitude[:len(magnitude)//4])
        
        # Subtract noise spectrum
        magnitude = np.maximum(magnitude - noise_spectrum, 0)
        
        # Reconstruct signal
        filtered_fft = magnitude * np.exp(1j * phase)
        filtered = np.fft.irfft(filtered_fft)
        
        return filtered
        
    def _update_noise_floor(self, samples: np.ndarray):
        """Update noise floor estimate using exponential averaging."""
        current_noise = 10 * np.log10(np.mean(samples**2))
        alpha = 0.1  # Smoothing factor
        self.noise_floor = alpha * current_noise + (1 - alpha) * self.noise_floor
        
    def get_signal_quality(self, samples: np.ndarray) -> SignalMetrics:
        """Calculate comprehensive signal quality metrics."""
        # Calculate basic metrics
        signal_power = np.mean(samples**2)
        signal_db = 10 * np.log10(signal_power)
        snr = signal_db - self.noise_floor
        
        # Calculate EVM
        evm = np.sqrt(np.mean((samples - np.mean(samples))**2)) / np.sqrt(signal_power)
        
        # Calculate crest factor
        crest_factor = np.max(np.abs(samples)) / np.sqrt(signal_power)
        
        # Calculate RMS value
        rms = np.sqrt(signal_power)
        
        # Calculate spectral features
        spectrum = np.fft.rfft(samples)
        freq = np.fft.rfftfreq(len(samples), 1/self.sample_rate)
        
        # Calculate bandwidth
        magnitude = np.abs(spectrum)
        threshold = 0.1 * np.max(magnitude)
        bandwidth = np.sum(magnitude > threshold) * (freq[1] - freq[0])
        
        # Calculate spectral flatness
        spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / np.mean(magnitude)
        
        # Detect modulation type
        modulation_type, confidence = self._detect_modulation(samples, spectral_flatness, bandwidth)
        
        return SignalMetrics(
            snr=float(snr),
            signal_strength=float(signal_db),
            evm=float(evm),
            crest_factor=float(crest_factor),
            rms=float(rms),
            noise_floor=float(self.noise_floor),
            bandwidth=float(bandwidth),
            spectral_flatness=float(spectral_flatness),
            modulation_type=modulation_type,
            confidence=float(confidence)
        )
        
    def _detect_modulation(self, samples: np.ndarray, 
                         spectral_flatness: float, 
                         bandwidth: float) -> Tuple[ModulationType, float]:
        """Detect modulation type with confidence score."""
        # Calculate signal statistics
        magnitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Calculate features
        magnitude_std = np.std(magnitude)
        phase_std = np.std(phase)
        magnitude_skew = skew(magnitude)
        phase_skew = skew(phase)
        
        # Initialize confidence scores
        confidences = {
            ModulationType.PSK: 0.0,
            ModulationType.ASK: 0.0,
            ModulationType.FSK: 0.0,
            ModulationType.FM: 0.0,
            ModulationType.AM: 0.0,
            ModulationType.QAM: 0.0,
            ModulationType.OFDM: 0.0
        }
        
        # Calculate confidence scores for each modulation type
        if magnitude_std < self.modulation_thresholds['magnitude_std'] and phase_std > self.modulation_thresholds['phase_std']:
            confidences[ModulationType.PSK] = 0.8
        elif magnitude_std > self.modulation_thresholds['magnitude_std'] and phase_std < self.modulation_thresholds['phase_std']:
            confidences[ModulationType.ASK] = 0.8
        elif magnitude_std < self.modulation_thresholds['magnitude_std'] and phase_std < self.modulation_thresholds['phase_std']:
            confidences[ModulationType.FSK] = 0.8
            
        if spectral_flatness > self.modulation_thresholds['spectral_flatness']:
            confidences[ModulationType.FM] = 0.7
        elif bandwidth < self.modulation_thresholds['bandwidth_threshold']:
            confidences[ModulationType.AM] = 0.7
            
        # Select modulation type with highest confidence
        modulation_type = max(confidences.items(), key=lambda x: x[1])
        
        return modulation_type[0], modulation_type[1]
        
    def set_thresholds(self, snr_threshold: Optional[float] = None,
                      magnitude_std: Optional[float] = None,
                      phase_std: Optional[float] = None,
                      spectral_flatness: Optional[float] = None,
                      bandwidth_threshold: Optional[float] = None):
        """Update detection thresholds."""
        if snr_threshold is not None:
            self.snr_threshold = snr_threshold
        if magnitude_std is not None:
            self.modulation_thresholds['magnitude_std'] = magnitude_std
        if phase_std is not None:
            self.modulation_thresholds['phase_std'] = phase_std
        if spectral_flatness is not None:
            self.modulation_thresholds['spectral_flatness'] = spectral_flatness
        if bandwidth_threshold is not None:
            self.modulation_thresholds['bandwidth_threshold'] = bandwidth_threshold 