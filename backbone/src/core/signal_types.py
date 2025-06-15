"""
Common signal-related types and data structures for GhostNet.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import numpy as np

class ModulationType(Enum):
    """Supported modulation types."""
    AM = "AM"
    FM = "FM"
    ASK = "ASK"
    PSK = "PSK"
    FSK = "FSK"
    QAM = "QAM"
    OFDM = "OFDM"
    UNKNOWN = "UNKNOWN"

@dataclass
class SignalFeatures:
    """Structured signal features for classification."""
    frequency: float
    bandwidth: float
    modulation: ModulationType
    snr: float
    spectral_centroid: float
    spectral_flatness: float
    zero_crossing_rate: float
    crest_factor: float
    rms: float
    entropy: float
    constellation_points: Optional[np.ndarray] = None
    eye_diagram: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ClassificationResult:
    """Results from signal classification."""
    predicted_class: str
    confidence: float
    matched_rule_id: Optional[str]
    features: SignalFeatures
    probabilities: Dict[str, float]
    metadata: Dict[str, Any]

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
class SignalMetrics:
    """Signal quality metrics."""
    snr: float
    signal_strength: float
    evm: float
    crest_factor: float
    rms: float
    noise_floor: float
    bandwidth: float
    spectral_flatness: float
    modulation_type: ModulationType
    confidence: float 