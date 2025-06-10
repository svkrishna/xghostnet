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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Results from signal detection algorithms."""
    detected: bool
    confidence: float
    frequency: float
    bandwidth: float
    snr: float
    features: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class SignalFeatures:
    """Extracted signal features for classification."""
    spectral_features: Dict[str, float]
    temporal_features: Dict[str, float]
    statistical_features: Dict[str, float]
    wavelet_features: Dict[str, float]
    modulation_features: Dict[str, float]

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
        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(samples)) ** 2
        freqs = np.fft.fftfreq(len(samples), 1/self.sample_rate)
        
        # Find peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]
        
        # Estimate bandwidth at -3dB point
        peak_power = spectrum[peak_idx]
        half_power = peak_power / 2
        
        # Find -3dB points
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > half_power:
            left_idx -= 1
            
        right_idx = peak_idx
        while right_idx < len(spectrum)-1 and spectrum[right_idx] > half_power:
            right_idx += 1
            
        bandwidth = freqs[right_idx] - freqs[left_idx]
        
        return abs(peak_freq), abs(bandwidth)
        
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
        """
        Extract comprehensive signal features.
        
        Args:
            samples: Input signal samples
            
        Returns:
            SignalFeatures object containing extracted features
        """
        # Spectral features
        spectrum = np.abs(np.fft.fft(samples)) ** 2
        freqs = np.fft.fftfreq(len(samples), 1/self.sample_rate)
        
        spectral_features = {
            'spectral_centroid': float(np.sum(freqs * spectrum) / np.sum(spectrum)),
            'spectral_bandwidth': float(np.sqrt(np.sum((freqs - np.sum(freqs * spectrum) / np.sum(spectrum))**2 * spectrum) / np.sum(spectrum))),
            'spectral_flatness': float(np.exp(np.mean(np.log(spectrum + 1e-10))) / np.mean(spectrum)),
            'spectral_rolloff': float(freqs[np.where(np.cumsum(spectrum) >= 0.85 * np.sum(spectrum))[0][0]])
        }
        
        # Temporal features
        temporal_features = {
            'zero_crossing_rate': float(np.sum(np.diff(np.signbit(samples))) / len(samples)),
            'rms': float(np.sqrt(np.mean(np.abs(samples)**2))),
            'peak_to_peak': float(np.max(np.abs(samples)) - np.min(np.abs(samples))),
            'crest_factor': float(np.max(np.abs(samples)) / np.sqrt(np.mean(np.abs(samples)**2)))
        }
        
        # Statistical features
        statistical_features = {
            'kurtosis': float(kurtosis(np.abs(samples))),
            'skewness': float(skew(np.abs(samples))),
            'entropy': float(entropy(np.abs(samples))),
            'variance': float(np.var(samples))
        }
        
        # Wavelet features
        coeffs = pywt.wavedec(samples, 'db4', level=4)
        wavelet_features = {
            'wavelet_energy': float(np.sum(np.abs(coeffs[0])**2)),
            'wavelet_entropy': float(entropy(np.abs(coeffs[0]))),
            'wavelet_mean': float(np.mean(np.abs(coeffs[0]))),
            'wavelet_std': float(np.std(np.abs(coeffs[0])))
        }
        
        # Modulation features
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase) * self.sample_rate / (2 * np.pi)
        
        modulation_features = {
            'frequency_deviation': float(np.std(freq)),
            'phase_deviation': float(np.std(phase)),
            'amplitude_mean': float(np.mean(np.abs(samples))),
            'amplitude_std': float(np.std(np.abs(samples)))
        }
        
        return SignalFeatures(
            spectral_features=spectral_features,
            temporal_features=temporal_features,
            statistical_features=statistical_features,
            wavelet_features=wavelet_features,
            modulation_features=modulation_features
        )
        
    def recognize_modulation(self, samples: np.ndarray) -> str:
        """
        Recognize the modulation type of the signal.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Detected modulation type
        """
        features = self.extract_features(samples)
        
        # Combine all features into a single array
        feature_vector = np.concatenate([
            list(features.spectral_features.values()),
            list(features.temporal_features.values()),
            list(features.statistical_features.values()),
            list(features.wavelet_features.values()),
            list(features.modulation_features.values())
        ])
        
        # Normalize features
        feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Predict modulation type
        # Note: This requires a trained classifier
        # For now, return a placeholder
        return "unknown"
        
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