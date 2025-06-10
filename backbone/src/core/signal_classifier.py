"""
Machine learning-based signal classification for GhostNet.
Implements various ML models for signal classification and modulation recognition.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, cast, TypedDict, TypeVar, cast, overload, Set
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from scipy import signal
from scipy.stats import moment, kurtosis, skew
import pywt
from scipy.signal.wavelets import cwt, morlet2
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import hilbert, stft, istft
import math
import librosa
from numpy.typing import NDArray
import scipy
import soundfile as sf
from datetime import datetime
from abc import ABC, abstractmethod
import threading
import time
import queue
from datetime import datetime, timedelta
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import socket
import zeroconf
from zeroconf import ServiceBrowser, ServiceListener, ServiceInfo
import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import base64

# Import SDR modules
try:
    from rtlsdr import RtlSdr  # type: ignore
except ImportError:
    RtlSdr = None  # type: ignore
    logging.warning("RTL-SDR module not found. RTL-SDR functionality will be disabled.")

try:
    import uhd  # type: ignore
except ImportError:
    uhd = None  # type: ignore
    logging.warning("USRP module not found. USRP functionality will be disabled.")

try:
    import SoapySDR  # type: ignore
except ImportError:
    SoapySDR = None  # type: ignore
    logging.warning("SoapySDR module not found. LimeSDR functionality will be disabled.")

try:
    import airspy  # type: ignore
except ImportError:
    airspy = None  # type: ignore
    logging.warning("Airspy module not found. Airspy functionality will be disabled.")

try:
    import sdrplay  # type: ignore
except ImportError:
    sdrplay = None  # type: ignore
    logging.warning("SDRplay module not found. SDRplay functionality will be disabled.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class SignalDataset(Dataset):
    """Dataset class for signal data."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetModel(nn.Module):
    """ResNet architecture for signal classification."""
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Calculate final feature size
        final_size = input_size // 32
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * final_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TransformerModel(nn.Module):
    """Transformer architecture for signal classification."""
    def __init__(self, input_size: int, num_classes: int, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 6, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension
        x = x.unsqueeze(-1)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.output_proj(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer architecture."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.get_buffer('pe')[:x.size(1)]
        return self.dropout(x)

class CNNModel(nn.Module):
    """Convolutional Neural Network for signal classification."""
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (input_size // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    """Long Short-Term Memory network for signal classification."""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs = q.size(0)
        
        # Linear projections and reshape
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(context)

class AttentionBlock(nn.Module):
    """Attention block with residual connection and layer normalization."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class AttentionModel(nn.Module):
    """Attention-based model for signal classification."""
    def __init__(self, input_size: int, num_classes: int, d_model: int = 256, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension
        x = x.unsqueeze(-1)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply attention blocks
        for block in self.attention_blocks:
            x = block(x)
            
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.output_proj(x)
        return x

@dataclass
class ClassificationResult:
    """Results from signal classification."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, float]
    metadata: Dict[str, Any]

class ClassificationMetrics(TypedDict):
    accuracy: float
    macro_avg_precision: float
    macro_avg_recall: float
    macro_avg_f1: float
    best_params: Optional[Dict[str, Any]]
    train_loss: Optional[float]
    optimal_lr: Optional[float]

class AdaptiveFilter:
    """Adaptive filtering for noise reduction."""
    def __init__(self, filter_length: int = 64, mu: float = 0.01):
        self.filter_length = filter_length
        self.mu = mu  # Step size
        self.weights = np.zeros(filter_length)
        
    def lms_filter(self, signal: NDArray[np.float64], reference: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply LMS adaptive filtering.
        
        Args:
            signal: Input signal with noise
            reference: Reference signal or noise estimate
            
        Returns:
            Filtered signal
        """
        filtered = np.zeros_like(signal)
        for i in range(len(signal)):
            if i < self.filter_length:
                filtered[i] = signal[i]
                continue
                
            # Get current window
            window = signal[i-self.filter_length:i]
            
            # Calculate output
            output = np.dot(self.weights, window)
            filtered[i] = output
            
            # Calculate error
            error = reference[i] - output
            
            # Update weights
            self.weights += self.mu * error * window
            
        return filtered
        
    def rls_filter(self, signal: NDArray[np.float64], reference: NDArray[np.float64], 
                  lambda_: float = 0.99, delta: float = 0.01) -> NDArray[np.float64]:
        """
        Apply RLS adaptive filtering.
        
        Args:
            signal: Input signal with noise
            reference: Reference signal or noise estimate
            lambda_: Forgetting factor
            delta: Initialization constant
            
        Returns:
            Filtered signal
        """
        filtered = np.zeros_like(signal)
        P = np.eye(self.filter_length) / delta  # Inverse correlation matrix
        
        for i in range(len(signal)):
            if i < self.filter_length:
                filtered[i] = signal[i]
                continue
                
            # Get current window
            window = signal[i-self.filter_length:i]
            
            # Calculate Kalman gain
            k = np.dot(P, window) / (lambda_ + np.dot(window, np.dot(P, window)))
            
            # Calculate output
            output = np.dot(self.weights, window)
            filtered[i] = output
            
            # Calculate error
            error = reference[i] - output
            
            # Update weights
            self.weights += k * error
            
            # Update inverse correlation matrix
            P = (P - np.outer(k, np.dot(window, P))) / lambda_
            
        return filtered

class NoiseReducer:
    """Noise reduction using multiple techniques."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def kalman_filter(self, signal: Union[NDArray[np.float64], NDArray[np.complex128]], 
                     process_noise: float = 0.001,
                     measurement_noise: float = 0.1) -> NDArray[np.float64]:
        """
        Apply Kalman filtering for noise reduction.
        
        Args:
            signal: Input signal (real or complex)
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
            
        Returns:
            Filtered signal
        """
        # Convert to complex if real
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
            
        # Cast to complex128 to satisfy type checker
        signal_complex = cast(NDArray[np.complex128], signal)
        
        # Initialize prediction and error arrays
        n = len(signal_complex)
        prediction = np.zeros(n, dtype=np.complex128)
        error = np.zeros(n, dtype=np.float64)
        
        # Initial state
        prediction[0] = signal_complex[0]
        error[0] = 1.0
        
        # Kalman filter iteration
        for i in range(1, n):
            # Prediction step
            prediction[i] = prediction[i-1]
            
            # Update step
            kalman_gain = error[i-1] / (error[i-1] + measurement_noise)
            prediction[i] = prediction[i] + kalman_gain * (signal_complex[i] - prediction[i])
            error[i] = (1 - kalman_gain) * error[i-1] + process_noise
            
        return np.abs(prediction).astype(np.float64)
        
    def spectral_subtraction(self, signal: Union[NDArray[np.float64], NDArray[np.complex128]], 
                           noise_estimate: NDArray[np.float64],
                           alpha: float = 2.0,
                           beta: float = 0.1) -> NDArray[np.float64]:
        """
        Apply spectral subtraction for noise reduction.
        
        Args:
            signal: Input signal (real or complex)
            noise_estimate: Noise estimate
            alpha: Oversubtraction factor
            beta: Spectral floor
            
        Returns:
            Denoised signal
        """
        # Convert to complex if real
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
            
        # Cast to complex128 to satisfy type checker
        signal_complex = cast(NDArray[np.complex128], signal)
        
        # Compute STFT
        f, t, Zxx = stft(signal_complex, fs=self.sample_rate, nperseg=256)
        
        # Compute noise floor
        noise_floor = np.mean(np.abs(Zxx), axis=1)
        
        # Apply spectral subtraction
        Zxx_denoised = np.zeros_like(Zxx)
        for i in range(Zxx.shape[1]):
            Zxx_denoised[:, i] = Zxx[:, i] - alpha * noise_floor[:, np.newaxis]
            Zxx_denoised[:, i] = np.maximum(Zxx_denoised[:, i], beta * noise_floor[:, np.newaxis])
            
        # Reconstruct signal
        _, denoised = istft(Zxx_denoised, fs=self.sample_rate)
        return np.abs(denoised).astype(np.float64)
        
    def wiener_filter(self, signal: Union[NDArray[np.float64], NDArray[np.complex128]], 
                     noise_estimate: NDArray[np.float64],
                     snr_estimate: float = 10.0) -> NDArray[np.float64]:
        """
        Apply Wiener filtering for noise reduction.
        
        Args:
            signal: Input signal (real or complex)
            noise_estimate: Noise estimate
            snr_estimate: Signal-to-noise ratio estimate
            
        Returns:
            Filtered signal
        """
        # Convert to complex if real
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
            
        # Cast to complex128 to satisfy type checker
        signal_complex = cast(NDArray[np.complex128], signal)
        
        # Compute STFT
        f, t, Zxx = stft(signal_complex, fs=self.sample_rate, nperseg=256)
        
        # Compute Wiener gain
        noise_power = np.mean(np.abs(Zxx), axis=1)
        signal_power = np.abs(Zxx) ** 2
        wiener_gain = signal_power / (signal_power + noise_power / snr_estimate)
        
        # Apply filter
        Zxx_filtered = Zxx * wiener_gain
        
        # Reconstruct signal
        _, filtered = istft(Zxx_filtered, fs=self.sample_rate)
        return np.abs(filtered).astype(np.float64)

class ModulationDetector:
    """Detects and handles different modulation schemes."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    @overload
    def detect_modulation(self, signal: NDArray[np.float64]) -> str: ...
        
    @overload
    def detect_modulation(self, signal: NDArray[np.complex128]) -> str: ...
        
    def detect_modulation(self, signal: Union[NDArray[np.float64], NDArray[np.complex128]]) -> str:
        """
        Detect the modulation scheme of the signal.
        
        Args:
            signal: Input signal (real or complex)
            
        Returns:
            Detected modulation type ('AM', 'FM', 'PSK', 'QAM', 'FSK', 'ASK', 'OFDM')
        """
        # Convert to complex if real
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
            
        # Compute signal statistics
        amplitude = np.abs(signal)
        phase = np.unwrap(np.angle(signal))
        freq = np.diff(phase) / (2 * np.pi)
        
        # Compute features
        amp_var = np.var(amplitude)
        phase_var = np.var(phase)
        freq_var = np.var(freq)
        
        # Detect modulation type
        if amp_var > 0.1 and phase_var < 0.1:
            return 'AM'
        elif amp_var < 0.1 and freq_var > 0.1:
            return 'FM'
        elif amp_var < 0.1 and phase_var > 0.1:
            return 'PSK'
        elif amp_var > 0.1 and phase_var > 0.1:
            return 'QAM'
        elif freq_var > 0.1 and np.std(np.diff(freq)) > 0.1:
            return 'FSK'
        elif amp_var > 0.1 and phase_var < 0.1:
            return 'ASK'
        else:
            return 'OFDM'
            
    @overload
    def demodulate(self, signal: NDArray[np.float64], modulation_type: str) -> NDArray[np.float64]: ...
        
    @overload
    def demodulate(self, signal: NDArray[np.complex128], modulation_type: str) -> NDArray[np.float64]: ...
        
    def demodulate(self, signal: Union[NDArray[np.float64], NDArray[np.complex128]], 
                  modulation_type: str) -> NDArray[np.float64]:
        """
        Demodulate the signal based on modulation type.
        
        Args:
            signal: Input signal (real or complex)
            modulation_type: Type of modulation
            
        Returns:
            Demodulated signal
        """
        # Convert to complex if real
        if not np.iscomplexobj(signal):
            signal = signal.astype(np.complex128)
            
        # Cast to complex128 to satisfy type checker
        signal_complex = cast(NDArray[np.complex128], signal)
            
        if modulation_type == 'AM':
            return self._demodulate_am(signal_complex)
        elif modulation_type == 'FM':
            return self._demodulate_fm(signal_complex)
        elif modulation_type == 'PSK':
            return self._demodulate_psk(signal_complex)
        elif modulation_type == 'QAM':
            return self._demodulate_qam(signal_complex)
        elif modulation_type == 'FSK':
            return self._demodulate_fsk(signal_complex)
        elif modulation_type == 'ASK':
            return self._demodulate_ask(signal_complex)
        elif modulation_type == 'OFDM':
            return self._demodulate_ofdm(signal_complex)
        else:
            raise ValueError(f"Unsupported modulation type: {modulation_type}")
            
    def _demodulate_am(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate AM signal."""
        # Envelope detection using FFT
        signal_real = signal.real.astype(np.float64)
        n = len(signal_real)
        fft_signal = np.fft.fft(signal_real)
        fft_signal[n//2:] = 0  # Zero out negative frequencies
        analytic_signal = np.fft.ifft(fft_signal)
        amplitude_envelope = np.abs(analytic_signal)
        return amplitude_envelope.astype(np.float64)
        
    def _demodulate_fm(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate FM signal."""
        # Phase differentiation
        phase = np.unwrap(np.angle(signal))
        freq = np.diff(phase) / (2 * np.pi)
        return freq.astype(np.float64)
        
    def _demodulate_psk(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate PSK signal."""
        # Phase detection
        phase = np.angle(signal)
        # Quantize to nearest symbol
        symbols = np.round(phase / (np.pi/2)) * (np.pi/2)
        return symbols.astype(np.float64)
        
    def _demodulate_qam(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate QAM signal."""
        # Separate I and Q components
        i_component = np.real(signal)
        q_component = np.imag(signal)
        # Quantize to nearest symbol
        i_symbols = np.round(i_component)
        q_symbols = np.round(q_component)
        # Return magnitude of complex symbols
        return np.abs(i_symbols + 1j * q_symbols).astype(np.float64)
        
    def _demodulate_fsk(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate FSK signal."""
        # Frequency detection
        phase = np.unwrap(np.angle(signal))
        freq = np.diff(phase) / (2 * np.pi)
        # Quantize to nearest frequency
        return np.round(freq).astype(np.float64)
        
    def _demodulate_ask(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate ASK signal."""
        # Amplitude detection
        amplitude = np.abs(signal)
        # Quantize to nearest symbol
        return np.round(amplitude).astype(np.float64)
        
    def _demodulate_ofdm(self, signal: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Demodulate OFDM signal."""
        # FFT to get subcarriers
        fft_size = 1024
        ofdm_symbols = np.fft.fft(signal.reshape(-1, fft_size))
        # Return magnitude of subcarriers
        return np.abs(ofdm_symbols).astype(np.float64)

class SignalRecorder:
    """Handles signal recording and playback capabilities."""
    
    def __init__(self, sample_rate: int = 44100, output_dir: str = "recordings"):
        """
        Initialize signal recorder.
        
        Args:
            sample_rate: Sample rate in Hz
            output_dir: Directory to save recordings
        """
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def record_signal(self, signal: Union[NDArray[np.float64], NDArray[np.complex128]], 
                     filename: Optional[str] = None) -> str:
        """
        Record signal to WAV file.
        
        Args:
            signal: Input signal (real or complex)
            filename: Optional filename, defaults to timestamp
            
        Returns:
            Path to saved recording
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"signal_{timestamp}.wav"
            
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to real if complex
        if np.iscomplexobj(signal):
            signal = np.abs(signal)
            
        # Normalize signal
        signal = signal / np.max(np.abs(signal))
        
        # Save to WAV file
        sf.write(filepath, signal, self.sample_rate)
        return filepath
        
    def play_signal(self, filepath: str) -> None:
        """
        Play recorded signal.
        
        Args:
            filepath: Path to WAV file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Recording not found: {filepath}")
            
        # Read and play signal
        signal, sr = sf.read(filepath)
        sf.play(signal, sr)
        sf.wait()  # Wait until playback is finished
        
    def get_recordings(self) -> List[str]:
        """
        Get list of available recordings.
        
        Returns:
            List of recording filepaths
        """
        return [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) 
                if f.endswith('.wav')]

@dataclass
class SDRConfig:
    """Configuration for SDR device."""
    center_freq: float
    sample_rate: float
    gain: float
    bandwidth: float
    device_args: Dict[str, Any]

    def __init__(self, center_freq: float, sample_rate: float, gain: float, bandwidth: float, device_args: Dict[str, Any]):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.bandwidth = bandwidth
        self.device_args = device_args

class SDRDevice(ABC):
    """Abstract base class for SDR devices."""
    
    @abstractmethod
    def initialize(self, config: SDRConfig) -> bool:
        """Initialize the SDR device."""
        pass
        
    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        """Read samples from the SDR device."""
        pass
        
    @abstractmethod
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        """Write samples to the SDR device."""
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Close the SDR device."""
        pass

class RTLSDREmulator(SDRDevice):
    """Emulator for RTL-SDR devices."""
    
    def __init__(self):
        self.sdr = None
        self.optimal_settings = {
            'sample_rate': 2.4e6,  # Optimal sample rate for RTL-SDR
            'gain': 40,            # Default gain
            'bandwidth': 2.4e6,    # Optimal bandwidth
            'agc_mode': True,      # Enable AGC
            'direct_sampling': False,  # Disable direct sampling
            'bias_tee': False      # Disable bias tee
        }
        
    def initialize(self, config: SDRConfig) -> bool:
        try:
            if RtlSdr is None:  # type: ignore
                logging.error("RTL-SDR module not available")
                return False
            self.sdr = RtlSdr()  # type: ignore
            
            # Apply optimal settings
            self.sdr.center_freq = self.optimal_settings['center_freq']
            self.sdr.sample_rate = self.optimal_settings['sample_rate']
            self.sdr.gain = self.optimal_settings['gain']
            self.sdr.bandwidth = self.optimal_settings['bandwidth']
            self.sdr.set_agc_mode(self.optimal_settings['agc_mode'])
            self.sdr.set_direct_sampling(self.optimal_settings['direct_sampling'])
            self.sdr.set_bias_tee(self.optimal_settings['bias_tee'])
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize RTL-SDR: {str(e)}")
            return False
            
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        if self.sdr is None:
            logging.error("RTL-SDR not initialized")
            return None
        try:
            samples = self.sdr.read_samples(num_samples)
            return np.array(samples, dtype=np.complex128)
        except Exception as e:
            logging.error(f"Failed to read samples from RTL-SDR: {str(e)}")
            return None
            
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        logging.warning("RTL-SDR does not support writing samples")
        return False
        
    def close(self) -> None:
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception as e:
                logging.error(f"Failed to close RTL-SDR: {str(e)}")
            finally:
                self.sdr = None

class HackRFEmulator(SDRDevice):
    """Emulator for HackRF devices."""
    
    def __init__(self):
        self.sdr = None
        self.optimal_settings = {
            'sample_rate': 20e6,   # Optimal sample rate for HackRF
            'gain': 40,            # Default gain
            'bandwidth': 20e6,     # Optimal bandwidth
            'lna_gain': 40,        # LNA gain
            'vga_gain': 40,        # VGA gain
            'amp_enable': True,    # Enable amplifier
            'antenna_enable': True # Enable antenna
        }
        
    def initialize(self, config: SDRConfig) -> bool:
        try:
            if hackrf is None:  # type: ignore
                logging.error("HackRF module not available")
                return False
            self.sdr = hackrf.HackRF()  # type: ignore
            
            # Apply optimal settings
            self.sdr.sample_rate = self.optimal_settings['sample_rate']
            self.sdr.center_freq = config.center_freq
            self.sdr.gain = self.optimal_settings['gain']
            self.sdr.bandwidth = self.optimal_settings['bandwidth']
            self.sdr.lna_gain = self.optimal_settings['lna_gain']
            self.sdr.vga_gain = self.optimal_settings['vga_gain']
            self.sdr.amp_enable = self.optimal_settings['amp_enable']
            self.sdr.antenna_enable = self.optimal_settings['antenna_enable']
            
            # Optimize buffer size
            self.sdr.buffer_size = 1024 * 1024  # 1MB buffer
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize HackRF: {str(e)}")
            return False
            
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        if self.sdr is None:
            logging.error("HackRF not initialized")
            return None
        try:
            samples = self.sdr.read_samples(num_samples)
            return np.array(samples, dtype=np.complex128)
        except Exception as e:
            logging.error(f"Failed to read samples from HackRF: {str(e)}")
            return None
            
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        if self.sdr is None:
            logging.error("HackRF not initialized")
            return False
        try:
            return self.sdr.write_samples(samples)
        except Exception as e:
            logging.error(f"Failed to write samples to HackRF: {str(e)}")
            return False
            
    def close(self) -> None:
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception as e:
                logging.error(f"Failed to close HackRF: {str(e)}")
            finally:
                self.sdr = None

class USRPEmulator(SDRDevice):
    """Emulator for USRP devices."""
    
    def __init__(self):
        self.sdr = None
        self.optimal_settings = {
            'sample_rate': 50e6,   # Optimal sample rate for USRP
            'gain': 50,            # Default gain
            'bandwidth': 50e6,     # Optimal bandwidth
            'clock_source': 'internal',  # Use internal clock
            'time_source': 'internal',   # Use internal time source
            'master_clock_rate': 200e6,  # Master clock rate
            'antenna': 'TX/RX'     # Default antenna
        }
        
    def initialize(self, config: SDRConfig) -> bool:
        try:
            if uhd is None:  # type: ignore
                logging.error("USRP module not available")
                return False
            self.sdr = uhd.usrp.MultiUSRP()  # type: ignore
            
            # Apply optimal settings
            self.sdr.set_clock_source(self.optimal_settings['clock_source'])
            self.sdr.set_time_source(self.optimal_settings['time_source'])
            self.sdr.set_master_clock_rate(self.optimal_settings['master_clock_rate'])
            self.sdr.set_antenna(self.optimal_settings['antenna'])
            self.sdr.set_sample_rate(self.optimal_settings['sample_rate'])
            self.sdr.set_center_freq(config.center_freq)
            self.sdr.set_gain(self.optimal_settings['gain'])
            self.sdr.set_bandwidth(self.optimal_settings['bandwidth'])
            
            # Optimize buffer size
            self.sdr.set_rx_buffer_size(1024 * 1024)  # 1MB buffer
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize USRP: {str(e)}")
            return False
            
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        if self.sdr is None:
            logging.error("USRP not initialized")
            return None
        try:
            samples = self.sdr.recv_num_samps(num_samples)
            return np.array(samples, dtype=np.complex128)
        except Exception as e:
            logging.error(f"Failed to read samples from USRP: {str(e)}")
            return None
            
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        if self.sdr is None:
            logging.error("USRP not initialized")
            return False
        try:
            return self.sdr.send_waveform(samples)
        except Exception as e:
            logging.error(f"Failed to write samples to USRP: {str(e)}")
            return False
            
    def close(self) -> None:
        if self.sdr is not None:
            try:
                del self.sdr
            except Exception as e:
                logging.error(f"Failed to close USRP: {str(e)}")
            finally:
                self.sdr = None

class LimeSDREmulator(SDRDevice):
    """Emulator for LimeSDR devices."""
    
    def __init__(self):
        self.sdr = None
        self.optimal_settings = {
            'sample_rate': 30.72e6,  # Optimal sample rate for LimeSDR
            'gain': 60,              # Default gain
            'bandwidth': 30.72e6,    # Optimal bandwidth
            'lna_gain': 30,          # LNA gain
            'tia_gain': 3,           # TIA gain
            'pga_gain': 30,          # PGA gain
            'calibrate': True,       # Enable calibration
            'antenna': 'LNAW'        # Default antenna
        }
        
    def initialize(self, config: SDRConfig) -> bool:
        try:
            if SoapySDR is None:  # type: ignore
                logging.error("LimeSDR module not available")
                return False
            self.sdr = SoapySDR.Device(dict(driver="lime"))  # type: ignore
            
            # Apply optimal settings
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.optimal_settings['sample_rate'])  # type: ignore
            self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, config.center_freq)  # type: ignore
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.optimal_settings['gain'])  # type: ignore
            self.sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, self.optimal_settings['bandwidth'])  # type: ignore
            self.sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, self.optimal_settings['antenna'])  # type: ignore
            
            # Set gains
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'LNA', self.optimal_settings['lna_gain'])  # type: ignore
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'TIA', self.optimal_settings['tia_gain'])  # type: ignore
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'PGA', self.optimal_settings['pga_gain'])  # type: ignore
            
            # Calibrate if enabled
            if self.optimal_settings['calibrate']:
                self.sdr.calibrate(SoapySDR.SOAPY_SDR_RX, 0)  # type: ignore
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize LimeSDR: {str(e)}")
            return False
            
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        if self.sdr is None:
            logging.error("LimeSDR not initialized")
            return None
        try:
            samples = np.empty(num_samples, dtype=np.complex128)
            self.sdr.readStream(self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32), [samples], num_samples)  # type: ignore
            return samples
        except Exception as e:
            logging.error(f"Failed to read samples from LimeSDR: {str(e)}")
            return None
            
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        if self.sdr is None:
            logging.error("LimeSDR not initialized")
            return False
        try:
            self.sdr.writeStream(self.sdr.setupStream(SoapySDR.SOAPY_SDR_TX, SoapySDR.SOAPY_SDR_CF32), [samples], len(samples))  # type: ignore
            return True
        except Exception as e:
            logging.error(f"Failed to write samples to LimeSDR: {str(e)}")
            return False
            
    def close(self) -> None:
        if self.sdr is not None:
            try:
                del self.sdr
            except Exception as e:
                logging.error(f"Failed to close LimeSDR: {str(e)}")
            finally:
                self.sdr = None

class AirspyEmulator(SDRDevice):
    """Emulator for Airspy devices."""
    
    def __init__(self):
        self.sdr = None
        self.optimal_settings = {
            'sample_rate': 10e6,    # Optimal sample rate for Airspy
            'gain': 15,             # Default gain
            'bandwidth': 10e6,      # Optimal bandwidth
            'linearity': 10,        # Linearity setting
            'sensitivity': 10,      # Sensitivity setting
            'bias_tee': False,      # Disable bias tee
            'packing': True         # Enable packing
        }
        
    def initialize(self, config: SDRConfig) -> bool:
        try:
            if airspy is None:  # type: ignore
                logging.error("Airspy module not available")
                return False
            self.sdr = airspy.Airspy()  # type: ignore
            
            # Apply optimal settings
            self.sdr.set_sample_rate(self.optimal_settings['sample_rate'])
            self.sdr.set_center_freq(config.center_freq)
            self.sdr.set_gain(self.optimal_settings['gain'])
            self.sdr.set_linearity(self.optimal_settings['linearity'])
            self.sdr.set_sensitivity(self.optimal_settings['sensitivity'])
            self.sdr.set_bias_tee(self.optimal_settings['bias_tee'])
            self.sdr.set_packing(self.optimal_settings['packing'])
            
            # Optimize buffer size
            self.sdr.set_buffer_size(262144)  # 256KB buffer
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize Airspy: {str(e)}")
            return False
            
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        if self.sdr is None:
            logging.error("Airspy not initialized")
            return None
        try:
            samples = self.sdr.read_samples(num_samples)
            return np.array(samples, dtype=np.complex128)
        except Exception as e:
            logging.error(f"Failed to read samples from Airspy: {str(e)}")
            return None
            
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        logging.warning("Airspy does not support writing samples")
        return False
        
    def close(self) -> None:
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception as e:
                logging.error(f"Failed to close Airspy: {str(e)}")
            finally:
                self.sdr = None

class SDRplayEmulator(SDRDevice):
    """Emulator for SDRplay devices."""
    
    def __init__(self):
        self.sdr = None
        self.optimal_settings = {
            'sample_rate': 8e6,     # Optimal sample rate for SDRplay
            'gain': 50,             # Default gain
            'bandwidth': 8e6,       # Optimal bandwidth
            'lna_state': 4,         # LNA state
            'if_gain': 20,          # IF gain
            'bias_tee': False,      # Disable bias tee
            'agc_setpoint': -30,    # AGC setpoint
            'agc_enable': True      # Enable AGC
        }
        
    def initialize(self, config: SDRConfig) -> bool:
        try:
            if sdrplay is None:  # type: ignore
                logging.error("SDRplay module not available")
                return False
            self.sdr = sdrplay.SDRplay()  # type: ignore
            
            # Apply optimal settings
            self.sdr.set_sample_rate(self.optimal_settings['sample_rate'])
            self.sdr.set_center_freq(config.center_freq)
            self.sdr.set_gain(self.optimal_settings['gain'])
            self.sdr.set_bandwidth(self.optimal_settings['bandwidth'])
            self.sdr.set_lna_state(self.optimal_settings['lna_state'])
            self.sdr.set_if_gain(self.optimal_settings['if_gain'])
            self.sdr.set_bias_tee(self.optimal_settings['bias_tee'])
            self.sdr.set_agc_setpoint(self.optimal_settings['agc_setpoint'])
            self.sdr.set_agc_enable(self.optimal_settings['agc_enable'])
            
            # Optimize buffer size
            self.sdr.set_buffer_size(262144)  # 256KB buffer
            
            return True
        except Exception as e:
            logging.error(f"Failed to initialize SDRplay: {str(e)}")
            return False
            
    def read_samples(self, num_samples: int) -> Optional[NDArray[np.complex128]]:
        if self.sdr is None:
            logging.error("SDRplay not initialized")
            return None
        try:
            samples = self.sdr.read_samples(num_samples)
            return np.array(samples, dtype=np.complex128)
        except Exception as e:
            logging.error(f"Failed to read samples from SDRplay: {str(e)}")
            return None
            
    def write_samples(self, samples: NDArray[np.complex128]) -> bool:
        logging.warning("SDRplay does not support writing samples")
        return False
        
    def close(self) -> None:
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception as e:
                logging.error(f"Failed to close SDRplay: {str(e)}")
            finally:
                self.sdr = None

class DeviceArray:
    """Manages an array of synchronized SDR devices."""
    
    def __init__(self, name: str):
        self.name = name
        self.devices: Dict[str, SDRDevice] = {}
        self.sync_queue = queue.Queue()
        self.running = False
        self.sync_thread = None
        self.sample_buffers: Dict[str, List[NDArray[np.complex128]]] = {}
        self.last_timestamp: Dict[str, float] = {}
        self.sync_interval = 1.0  # seconds
        self.max_buffer_size = 1000  # samples per device
        
    def add_device(self, device_id: str, device: SDRDevice) -> bool:
        """Add a device to the array."""
        if device_id in self.devices:
            logging.error(f"Device {device_id} already in array {self.name}")
            return False
        self.devices[device_id] = device
        self.sample_buffers[device_id] = []
        self.last_timestamp[device_id] = 0.0
        return True
        
    def remove_device(self, device_id: str) -> None:
        """Remove a device from the array."""
        if device_id in self.devices:
            del self.devices[device_id]
            del self.sample_buffers[device_id]
            del self.last_timestamp[device_id]
            
    def start(self) -> bool:
        """Start the device array synchronization."""
        if self.running:
            return True
            
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        return True
        
    def stop(self) -> None:
        """Stop the device array synchronization."""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join()
            
    def _sync_loop(self) -> None:
        """Main synchronization loop."""
        while self.running:
            try:
                # Collect samples from all devices
                samples = {}
                timestamps = {}
                
                for device_id, device in self.devices.items():
                    try:
                        # Read samples with timestamp
                        device_samples = device.read_samples(1024)
                        if device_samples is not None:
                            samples[device_id] = device_samples
                            timestamps[device_id] = time.time()
                    except Exception as e:
                        logging.error(f"Error reading from device {device_id}: {str(e)}")
                        
                # Synchronize timestamps
                if samples:
                    self._synchronize_samples(samples, timestamps)
                    
                time.sleep(self.sync_interval)
                
            except Exception as e:
                logging.error(f"Error in sync loop: {str(e)}")
                time.sleep(1.0)
                
    def _synchronize_samples(self, samples: Dict[str, NDArray[np.complex128]], 
                           timestamps: Dict[str, float]) -> None:
        """Synchronize samples from multiple devices."""
        try:
            # Find the reference timestamp (earliest)
            ref_timestamp = min(timestamps.values())
            
            # Adjust samples based on timing differences
            for device_id, device_samples in samples.items():
                time_diff = timestamps[device_id] - ref_timestamp
                
                # Apply time correction if needed
                if abs(time_diff) > 0.001:  # 1ms threshold
                    # Interpolate samples to correct timing
                    corrected_samples = self._interpolate_samples(device_samples, time_diff)
                    self.sample_buffers[device_id].append(corrected_samples)
                else:
                    self.sample_buffers[device_id].append(device_samples)
                    
                # Trim buffer if too large
                if len(self.sample_buffers[device_id]) > self.max_buffer_size:
                    self.sample_buffers[device_id] = self.sample_buffers[device_id][-self.max_buffer_size:]
                    
        except Exception as e:
            logging.error(f"Error synchronizing samples: {str(e)}")
            
    def _interpolate_samples(self, samples: NDArray[np.complex128], 
                           time_diff: float) -> NDArray[np.complex128]:
        """Interpolate samples to correct timing differences."""
        try:
            # Calculate number of samples to shift
            sample_diff = int(time_diff * len(samples))
            
            if sample_diff == 0:
                return samples
                
            # Create interpolation points
            x_old = np.arange(len(samples))
            x_new = np.arange(len(samples)) + sample_diff
            
            # Interpolate real and imaginary parts separately
            real_interp = np.interp(x_new, x_old, samples.real)
            imag_interp = np.interp(x_new, x_old, samples.imag)
            
            return real_interp + 1j * imag_interp
            
        except Exception as e:
            logging.error(f"Error interpolating samples: {str(e)}")
            return samples
            
    def get_synchronized_samples(self, num_samples: int) -> Dict[str, NDArray[np.complex128]]:
        """Get synchronized samples from all devices."""
        result = {}
        try:
            for device_id, buffer in self.sample_buffers.items():
                if buffer:
                    # Concatenate available samples
                    samples = np.concatenate(buffer)
                    # Take requested number of samples
                    result[device_id] = samples[:num_samples]
                    # Keep remaining samples
                    self.sample_buffers[device_id] = [samples[num_samples:]]
        except Exception as e:
            logging.error(f"Error getting synchronized samples: {str(e)}")
        return result

class SDRManager:
    """Manager for SDR devices."""
    
    def __init__(self):
        self.devices: Dict[str, SDRDevice] = {}
        self.device_arrays: Dict[str, DeviceArray] = {}
        
    def add_device(self, device_id: str, device_type: str, config: SDRConfig) -> bool:
        """Add a new SDR device."""
        if device_id in self.devices:
            logging.error(f"Device {device_id} already exists")
            return False
            
        device: Optional[SDRDevice] = None
        if device_type == 'rtlsdr':
            device = RTLSDREmulator()
        elif device_type == 'hackrf':
            device = HackRFEmulator()
        elif device_type == 'usrp':
            device = USRPEmulator()
        elif device_type == 'limesdr':
            device = LimeSDREmulator()
        elif device_type == 'airspy':
            device = AirspyEmulator()
        elif device_type == 'sdrplay':
            device = SDRplayEmulator()
        else:
            logging.error(f"Unknown device type: {device_type}")
            return False
            
        if device and device.initialize(config):
            self.devices[device_id] = device
            return True
        return False

    def remove_device(self, device_id: str) -> None:
        """Remove an SDR device."""
        if device_id in self.devices:
            self.devices[device_id].close()
            del self.devices[device_id]

    def read_samples(self, device_id: str, num_samples: int) -> Optional[NDArray[np.complex128]]:
        """Read samples from a specific device."""
        if device_id not in self.devices:
            logging.error(f"Device not found: {device_id}")
            return None
        return self.devices[device_id].read_samples(num_samples)

    def write_samples(self, device_id: str, samples: NDArray[np.complex128]) -> bool:
        """Write samples to a specific device."""
        if device_id not in self.devices:
            logging.error(f"Device not found: {device_id}")
            return False
        return self.devices[device_id].write_samples(samples)
        
    def create_device_array(self, array_name: str, device_ids: List[str]) -> bool:
        """Create a new device array with specified devices."""
        if array_name in self.device_arrays:
            logging.error(f"Device array {array_name} already exists")
            return False
            
        # Verify all devices exist
        for device_id in device_ids:
            if device_id not in self.devices:
                logging.error(f"Device {device_id} not found")
                return False
                
        # Create and initialize array
        array = DeviceArray(array_name)
        for device_id in device_ids:
            if not array.add_device(device_id, self.devices[device_id]):
                return False
                
        self.device_arrays[array_name] = array
        return array.start()
        
    def get_synchronized_samples(self, array_name: str, num_samples: int) -> Dict[str, NDArray[np.complex128]]:
        """Get synchronized samples from a device array."""
        if array_name not in self.device_arrays:
            logging.error(f"Device array {array_name} not found")
            return {}
        return self.device_arrays[array_name].get_synchronized_samples(num_samples)
        
    def remove_device_array(self, array_name: str) -> None:
        """Remove a device array."""
        if array_name in self.device_arrays:
            self.device_arrays[array_name].stop()
            del self.device_arrays[array_name]
            
    def close_all(self) -> None:
        """Close all devices and arrays."""
        for array in self.device_arrays.values():
            array.stop()
        self.device_arrays.clear()
        
        for device in self.devices.values():
            device.close()
        self.devices.clear()

class SignalClassifier:
    def __init__(self, model_type: str = 'rf', model_path: Optional[str] = None,
                 ensemble_size: int = 3):
        """
        Initialize the signal classifier.
        
        Args:
            model_type: Type of model to use ('rf', 'svm', 'gb', 'mlp', 'cnn', 'lstm', 
                        'resnet', 'transformer', 'attention')
            model_path: Path to pre-trained model file
            ensemble_size: Number of models in ensemble
        """
        self.model_type = model_type
        self.ensemble_size = ensemble_size
        self.models = []
        self.scalers = []
        
        if model_type == 'ensemble':
            # Create ensemble of different model types
            model_types = ['rf', 'gb', 'mlp', 'cnn', 'lstm', 'resnet', 'transformer', 'attention']
            for _ in range(ensemble_size):
                model_type = np.random.choice(model_types)
                self.models.append(self._create_model(model_type))
                self.scalers.append(StandardScaler())
        else:
            self.model = self._create_model(model_type)
            self.scaler = StandardScaler()
            
        self.feature_names = self._get_feature_names()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_scaler = GradScaler()  # For mixed precision training
        
        # Initialize noise reduction components
        self.adaptive_filter = AdaptiveFilter()
        self.noise_reducer = NoiseReducer()
        
        self.modulation_detector = ModulationDetector()
        
        self.recorder = SignalRecorder()
        
        self.sdr_manager = SDRManager()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
        # Initialize API server
        self.api = SignalClassifierAPI(self)
        
    def _create_model(self, model_type: str) -> Any:
        """Create a new model instance."""
        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'svm':
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        elif model_type == 'cnn':
            return CNNModel(input_size=1024, num_classes=10)
        elif model_type == 'lstm':
            return LSTMModel(input_size=1024, hidden_size=128, num_classes=10)
        elif model_type == 'resnet':
            return ResNetModel(input_size=1024, num_classes=10)
        elif model_type == 'transformer':
            return TransformerModel(input_size=1024, num_classes=10)
        elif model_type == 'attention':
            return AttentionModel(input_size=1024, num_classes=10)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names used by the classifier."""
        return [
            # Spectral features
            'spectral_centroid',
            'spectral_bandwidth',
            'spectral_flatness',
            'spectral_rolloff',
            'spectral_skewness',
            'spectral_kurtosis',
            'spectral_entropy',
            'spectral_flux',
            
            # Temporal features
            'zero_crossing_rate',
            'rms',
            'peak_to_peak',
            'crest_factor',
            'form_factor',
            'clearance_factor',
            'impulse_factor',
            'shape_factor',
            
            # Statistical features
            'kurtosis',
            'skewness',
            'entropy',
            'variance',
            'mean',
            'median',
            'iqr',
            'range',
            
            # Wavelet features
            'wavelet_energy',
            'wavelet_entropy',
            'wavelet_mean',
            'wavelet_std',
            'wavelet_skewness',
            'wavelet_kurtosis',
            'wavelet_peak',
            'wavelet_valley',
            
            # Modulation features
            'frequency_deviation',
            'phase_deviation',
            'amplitude_mean',
            'amplitude_std',
            'modulation_index',
            'bandwidth_occupancy',
            'peak_frequency',
            'spectral_peak_ratio',
            
            # Higher-order statistics
            'cumulant_2',
            'cumulant_3',
            'cumulant_4',
            'moment_2',
            'moment_3',
            'moment_4',
            'moment_5',
            'moment_6',
            
            # Additional spectral features
            'spectral_bandwidth_ratio',
            'spectral_centroid_std',
            'spectral_flatness_std',
            'spectral_rolloff_std',
            'spectral_flux_std',
            'spectral_crest',
            'spectral_spread',
            'spectral_decrease',
            
            # Additional temporal features
            'temporal_crest',
            'temporal_spread',
            'temporal_decrease',
            'temporal_entropy',
            'temporal_flatness',
            'temporal_flux',
            'temporal_centroid',
            'temporal_rolloff',
            
            # Time-frequency features
            'wavelet_scale_1',
            'wavelet_scale_2',
            'wavelet_scale_3',
            'wavelet_scale_4',
            'wavelet_scale_5',
            'wavelet_scale_6',
            'wavelet_scale_7',
            'wavelet_scale_8',
            
            # Cepstral features
            'cepstral_centroid',
            'cepstral_flux',
            'cepstral_rolloff',
            'cepstral_flatness',
            'cepstral_crest',
            'cepstral_spread',
            'cepstral_decrease',
            'cepstral_entropy',
            
            # Hilbert transform features
            'hilbert_phase_mean',
            'hilbert_phase_std',
            'hilbert_amplitude_mean',
            'hilbert_amplitude_std',
            'hilbert_frequency_mean',
            'hilbert_frequency_std',
            'hilbert_instantaneous_phase',
            'hilbert_instantaneous_frequency',
            
            # Higher-order spectral features
            'bispectrum_mean',
            'bispectrum_std',
            'bispectrum_skewness',
            'bispectrum_kurtosis',
            'trispectrum_mean',
            'trispectrum_std',
            'trispectrum_skewness',
            'trispectrum_kurtosis',
            
            # Additional features
            'mfcc_1',
            'mfcc_2',
            'mfcc_3',
            'mfcc_4',
            'mfcc_5',
            'mfcc_6',
            'mfcc_7',
            'mfcc_8',
            'mfcc_9',
            'mfcc_10',
            'mfcc_11',
            'mfcc_12',
            'mfcc_13',
            
            'chroma_1',
            'chroma_2',
            'chroma_3',
            'chroma_4',
            'chroma_5',
            'chroma_6',
            'chroma_7',
            'chroma_8',
            'chroma_9',
            'chroma_10',
            'chroma_11',
            'chroma_12',
            
            'spectral_contrast_1',
            'spectral_contrast_2',
            'spectral_contrast_3',
            'spectral_contrast_4',
            'spectral_contrast_5',
            'spectral_contrast_6',
            'spectral_contrast_7',
            
            'tonnetz_1',
            'tonnetz_2',
            'tonnetz_3',
            'tonnetz_4',
            'tonnetz_5',
            'tonnetz_6'
        ]
        
    def extract_features(self, samples: NDArray[np.float64]) -> Dict[str, float]:
        """
        Extract features from signal samples with modulation-specific features.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Dictionary of extracted features
        """
        # Get base features
        features = self._extract_base_features(samples)
        
        # Add modulation-specific features
        modulation_type = self.modulation_detector.detect_modulation(samples)
        
        if modulation_type == 'AM':
            # AM-specific features
            features.update({
                'modulation_index': float(np.mean(np.abs(samples)) / np.std(np.abs(samples))),
                'carrier_power': float(np.mean(np.abs(samples)**2)),
                'sideband_power': float(np.var(np.abs(samples)))
            })
        elif modulation_type == 'FM':
            # FM-specific features
            phase = np.unwrap(np.angle(samples))
            freq = np.diff(phase) / (2 * np.pi)
            features.update({
                'frequency_deviation': float(np.std(freq)),
                'modulation_index': float(np.std(freq) / np.mean(np.abs(freq))),
                'phase_variance': float(np.var(phase))
            })
        elif modulation_type in ['PSK', 'QAM']:
            # PSK/QAM-specific features
            features.update({
                'constellation_variance': float(np.var(np.abs(samples))),
                'phase_variance': float(np.var(np.angle(samples))),
                'symbol_rate': float(1 / np.mean(np.diff(np.where(np.abs(np.diff(np.angle(samples))) > np.pi/4)[0])))
            })
        elif modulation_type == 'FSK':
            # FSK-specific features
            phase = np.unwrap(np.angle(samples))
            freq = np.diff(phase) / (2 * np.pi)
            features.update({
                'frequency_separation': float(np.std(freq)),
                'symbol_rate': float(1 / np.mean(np.diff(np.where(np.abs(np.diff(freq)) > np.std(freq)/2)[0]))),
                'frequency_stability': float(np.std(np.diff(freq)))
            })
        elif modulation_type == 'OFDM':
            # OFDM-specific features
            fft_size = 1024
            ofdm_symbols = np.fft.fft(samples.reshape(-1, fft_size))
            features.update({
                'subcarrier_power': float(np.mean(np.abs(ofdm_symbols)**2)),
                'cyclic_prefix_ratio': float(len(samples) / (len(samples) - fft_size)),
                'subcarrier_spacing': float(self.modulation_detector.sample_rate / fft_size)
            })
            
        return features
        
    def _extract_base_features(self, samples: NDArray[np.float64]) -> Dict[str, float]:
        """
        Extract base features from signal samples.
        
        Args:
            samples: Input signal samples
            
        Returns:
            Dictionary of extracted base features
        """
        # Compute power spectrum
        spectrum = np.abs(np.fft.fft(samples)) ** 2
        freqs = np.fft.fftfreq(len(samples))
        
        # Spectral features
        spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
        spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * spectrum) / np.sum(spectrum))
        spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / np.mean(spectrum)
        spectral_rolloff = freqs[np.where(np.cumsum(spectrum) >= 0.85 * np.sum(spectrum))[0][0]]
        spectral_skewness = skew(spectrum)
        spectral_kurtosis = kurtosis(spectrum)
        spectral_entropy = -np.sum(spectrum * np.log2(spectrum + 1e-10))
        spectral_flux = np.sum(np.abs(np.diff(spectrum)))
        
        # Temporal features
        zero_crossing_rate = np.sum(np.diff(np.signbit(samples))) / len(samples)
        rms = np.sqrt(np.mean(np.abs(samples)**2))
        peak_to_peak = np.max(np.abs(samples)) - np.min(np.abs(samples))
        crest_factor = np.max(np.abs(samples)) / rms
        form_factor = rms / np.mean(np.abs(samples))
        clearance_factor = np.max(np.abs(samples)) / np.sqrt(np.mean(np.abs(samples)**2))
        impulse_factor = np.max(np.abs(samples)) / np.mean(np.abs(samples))
        shape_factor = rms / np.mean(np.abs(samples))
        
        # Statistical features
        kurt = kurtosis(samples)
        skewness = skew(samples)
        entropy = -np.sum(spectrum * np.log2(spectrum + 1e-10))
        variance = np.var(samples)
        mean = np.mean(samples)
        median = np.median(samples)
        iqr = np.percentile(samples, 75) - np.percentile(samples, 25)
        range_val = np.max(samples) - np.min(samples)
        
        # Wavelet features
        coeffs = pywt.wavedec(samples, 'db4', level=4)
        wavelet_energy = np.sum(np.abs(coeffs[0])**2)
        wavelet_entropy = -np.sum(np.abs(coeffs[0])**2 * np.log2(np.abs(coeffs[0])**2 + 1e-10))
        wavelet_mean = np.mean(np.abs(coeffs[0]))
        wavelet_std = np.std(np.abs(coeffs[0]))
        wavelet_skewness = skew(np.abs(coeffs[0]))
        wavelet_kurtosis = kurtosis(np.abs(coeffs[0]))
        wavelet_peak = np.max(np.abs(coeffs[0]))
        wavelet_valley = np.min(np.abs(coeffs[0]))
        
        # Modulation features
        phase = np.unwrap(np.angle(samples))
        freq = np.diff(phase) / (2 * np.pi)
        frequency_deviation = np.std(freq)
        phase_deviation = np.std(phase)
        amplitude_mean = np.mean(np.abs(samples))
        amplitude_std = np.std(np.abs(samples))
        modulation_index = frequency_deviation / np.mean(np.abs(freq))
        bandwidth_occupancy = np.sum(spectrum > 0.1 * np.max(spectrum)) / len(spectrum)
        peak_frequency = freqs[np.argmax(spectrum)]
        spectral_peak_ratio = np.max(spectrum) / np.mean(spectrum)
        
        # Higher-order statistics
        cumulant_2 = moment(samples, 2)
        cumulant_3 = moment(samples, 3)
        cumulant_4 = moment(samples, 4)
        moment_2 = moment(samples, 2)
        moment_3 = moment(samples, 3)
        moment_4 = moment(samples, 4)
        moment_5 = moment(samples, 5)
        moment_6 = moment(samples, 6)
        
        # Additional spectral features
        spectral_bandwidth_ratio = spectral_bandwidth / (np.max(freqs) - np.min(freqs))
        spectral_centroid_std = np.std(freqs * spectrum) / np.sum(spectrum)
        spectral_flatness_std = np.std(np.log(spectrum + 1e-10))
        spectral_rolloff_std = np.std(np.cumsum(spectrum) / np.sum(spectrum))
        spectral_flux_std = np.std(np.abs(np.diff(spectrum)))
        spectral_crest = np.max(spectrum) / np.mean(spectrum)
        spectral_spread = np.sqrt(np.sum((freqs - spectral_centroid)**2 * spectrum) / np.sum(spectrum))
        spectral_decrease = np.sum((spectrum[1:] - spectrum[0]) / np.arange(1, len(spectrum))) / np.sum(spectrum[1:])
        
        # Additional temporal features
        temporal_crest = np.max(np.abs(samples)) / np.mean(np.abs(samples))
        temporal_spread = np.std(samples)
        temporal_decrease = np.sum((samples[1:] - samples[0]) / np.arange(1, len(samples))) / np.sum(samples[1:])
        temporal_entropy = -np.sum(np.abs(samples) * np.log2(np.abs(samples) + 1e-10))
        temporal_flatness = np.exp(np.mean(np.log(np.abs(samples) + 1e-10))) / np.mean(np.abs(samples))
        temporal_flux = np.sum(np.abs(np.diff(samples)))
        temporal_centroid = np.sum(np.arange(len(samples)) * np.abs(samples)) / np.sum(np.abs(samples))
        temporal_rolloff = np.percentile(np.abs(samples), 85)
        
        # Time-frequency features using CWT
        widths = np.arange(1, 65)
        cwtmatr = scipy.signal.wavelets.cwt(samples, morlet2, widths)
        wavelet_scales = [np.mean(np.abs(cwtmatr[i])) for i in range(8)]
        
        # Cepstral features
        cepstrum = np.abs(np.fft.ifft(np.log(spectrum + 1e-10)))
        cepstral_centroid = np.sum(np.arange(len(cepstrum)) * cepstrum) / np.sum(cepstrum)
        cepstral_flux = np.sum(np.abs(np.diff(cepstrum)))
        cepstral_rolloff = np.percentile(cepstrum, 85)
        cepstral_flatness = np.exp(np.mean(np.log(cepstrum + 1e-10))) / np.mean(cepstrum)
        cepstral_crest = np.max(cepstrum) / np.mean(cepstrum)
        cepstral_spread = np.std(cepstrum)
        cepstral_decrease = np.sum((cepstrum[1:] - cepstrum[0]) / np.arange(1, len(cepstrum))) / np.sum(cepstrum[1:])
        cepstral_entropy = -np.sum(cepstrum * np.log2(cepstrum + 1e-10))
        
        # Hilbert transform features
        analytic_signal = cast(NDArray[np.complex128], hilbert(samples))
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
        
        hilbert_phase_mean = np.mean(instantaneous_phase)
        hilbert_phase_std = np.std(instantaneous_phase)
        hilbert_amplitude_mean = np.mean(amplitude_envelope)
        hilbert_amplitude_std = np.std(amplitude_envelope)
        hilbert_frequency_mean = np.mean(instantaneous_frequency)
        hilbert_frequency_std = np.std(instantaneous_frequency)
        hilbert_instantaneous_phase = np.mean(np.abs(np.diff(instantaneous_phase)))
        hilbert_instantaneous_frequency = np.mean(np.abs(np.diff(instantaneous_frequency)))
        
        # Higher-order spectral features
        bispectrum = np.abs(np.fft.fft2(samples.reshape(-1, 1) * samples.reshape(1, -1)))
        trispectrum = np.abs(np.fft.fft2(samples.reshape(-1, 1, 1) * samples.reshape(1, -1, 1) * samples.reshape(1, 1, -1)))
        
        bispectrum_mean = np.mean(bispectrum)
        bispectrum_std = np.std(bispectrum)
        bispectrum_skewness = skew(bispectrum.flatten())
        bispectrum_kurtosis = kurtosis(bispectrum.flatten())
        
        trispectrum_mean = np.mean(trispectrum)
        trispectrum_std = np.std(trispectrum)
        trispectrum_skewness = skew(trispectrum.flatten())
        trispectrum_kurtosis = kurtosis(trispectrum.flatten())
        
        # Additional features
        # MFCC features
        mfcc = librosa.feature.mfcc(y=samples, sr=44100, n_mfcc=13)
        mfcc_features = {f'mfcc_{i+1}': float(np.mean(mfcc[i])) for i in range(13)}
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=samples, sr=44100)
        chroma_features = {f'chroma_{i+1}': float(np.mean(chroma[i])) for i in range(12)}
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=samples, sr=44100)
        contrast_features = {f'spectral_contrast_{i+1}': float(np.mean(contrast[i])) for i in range(7)}
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=samples, sr=44100)
        tonnetz_features = {f'tonnetz_{i+1}': float(np.mean(tonnetz[i])) for i in range(6)}
        
        # Combine all features
        features = {
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_flatness': float(spectral_flatness),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_skewness': float(spectral_skewness),
            'spectral_kurtosis': float(spectral_kurtosis),
            'spectral_entropy': float(spectral_entropy),
            'spectral_flux': float(spectral_flux),
            'zero_crossing_rate': float(zero_crossing_rate),
            'rms': float(rms),
            'peak_to_peak': float(peak_to_peak),
            'crest_factor': float(crest_factor),
            'form_factor': float(form_factor),
            'clearance_factor': float(clearance_factor),
            'impulse_factor': float(impulse_factor),
            'shape_factor': float(shape_factor),
            'kurtosis': float(kurt),
            'skewness': float(skewness),
            'entropy': float(entropy),
            'variance': float(variance),
            'mean': float(mean),
            'median': float(median),
            'iqr': float(iqr),
            'range': float(range_val),
            'wavelet_energy': float(wavelet_energy),
            'wavelet_entropy': float(wavelet_entropy),
            'wavelet_mean': float(wavelet_mean),
            'wavelet_std': float(wavelet_std),
            'wavelet_skewness': float(wavelet_skewness),
            'wavelet_kurtosis': float(wavelet_kurtosis),
            'wavelet_peak': float(wavelet_peak),
            'wavelet_valley': float(wavelet_valley),
            'frequency_deviation': float(frequency_deviation),
            'phase_deviation': float(phase_deviation),
            'amplitude_mean': float(amplitude_mean),
            'amplitude_std': float(amplitude_std),
            'modulation_index': float(modulation_index),
            'bandwidth_occupancy': float(bandwidth_occupancy),
            'peak_frequency': float(peak_frequency),
            'spectral_peak_ratio': float(spectral_peak_ratio),
            'cumulant_2': float(cumulant_2),
            'cumulant_3': float(cumulant_3),
            'cumulant_4': float(cumulant_4),
            'moment_2': float(moment_2),
            'moment_3': float(moment_3),
            'moment_4': float(moment_4),
            'moment_5': float(moment_5),
            'moment_6': float(moment_6),
            'spectral_bandwidth_ratio': float(spectral_bandwidth_ratio),
            'spectral_centroid_std': float(spectral_centroid_std),
            'spectral_flatness_std': float(spectral_flatness_std),
            'spectral_rolloff_std': float(spectral_rolloff_std),
            'spectral_flux_std': float(spectral_flux_std),
            'spectral_crest': float(spectral_crest),
            'spectral_spread': float(spectral_spread),
            'spectral_decrease': float(spectral_decrease),
            'temporal_crest': float(temporal_crest),
            'temporal_spread': float(temporal_spread),
            'temporal_decrease': float(temporal_decrease),
            'temporal_entropy': float(temporal_entropy),
            'temporal_flatness': float(temporal_flatness),
            'temporal_flux': float(temporal_flux),
            'temporal_centroid': float(temporal_centroid),
            'temporal_rolloff': float(temporal_rolloff),
            **{f'wavelet_scale_{i+1}': float(scale) for i, scale in enumerate(wavelet_scales)},
            'cepstral_centroid': float(cepstral_centroid),
            'cepstral_flux': float(cepstral_flux),
            'cepstral_rolloff': float(cepstral_rolloff),
            'cepstral_flatness': float(cepstral_flatness),
            'cepstral_crest': float(cepstral_crest),
            'cepstral_spread': float(cepstral_spread),
            'cepstral_decrease': float(cepstral_decrease),
            'cepstral_entropy': float(cepstral_entropy),
            'hilbert_phase_mean': float(hilbert_phase_mean),
            'hilbert_phase_std': float(hilbert_phase_std),
            'hilbert_amplitude_mean': float(hilbert_amplitude_mean),
            'hilbert_amplitude_std': float(hilbert_amplitude_std),
            'hilbert_frequency_mean': float(hilbert_frequency_mean),
            'hilbert_frequency_std': float(hilbert_frequency_std),
            'hilbert_instantaneous_phase': float(hilbert_instantaneous_phase),
            'hilbert_instantaneous_frequency': float(hilbert_instantaneous_frequency),
            'bispectrum_mean': float(bispectrum_mean),
            'bispectrum_std': float(bispectrum_std),
            'bispectrum_skewness': float(bispectrum_skewness),
            'bispectrum_kurtosis': float(bispectrum_kurtosis),
            'trispectrum_mean': float(trispectrum_mean),
            'trispectrum_std': float(trispectrum_std),
            'trispectrum_skewness': float(trispectrum_skewness),
            'trispectrum_kurtosis': float(trispectrum_kurtosis),
            **mfcc_features,
            **chroma_features,
            **contrast_features,
            **tonnetz_features
        }
        
        return features
        
    def preprocess_signal(self, samples: Union[NDArray[np.float64], NDArray[np.complex128]], 
                         noise_reference: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        """
        Preprocess signal with noise reduction.
        
        Args:
            samples: Input signal samples (real or complex)
            noise_reference: Optional reference noise signal
            
        Returns:
            Preprocessed signal
        """
        # Convert to complex if real
        if not np.iscomplexobj(samples):
            samples = samples.astype(np.complex128)
            
        # Cast to complex128 to satisfy type checker
        samples_complex = cast(NDArray[np.complex128], samples)
            
        # Apply Kalman filtering
        filtered = self.noise_reducer.kalman_filter(samples_complex)
        
        if noise_reference is not None:
            # Apply adaptive filtering
            filtered = self.adaptive_filter.lms_filter(filtered, noise_reference)
            
            # Apply spectral subtraction
            filtered = self.noise_reducer.spectral_subtraction(filtered, noise_reference)
            
        return filtered.astype(np.float64)
        
    def classify_signal(self, samples: Union[NDArray[np.float64], NDArray[np.complex128]], 
                       noise_reference: Optional[NDArray[np.float64]] = None,
                       detect_modulation: bool = True) -> ClassificationResult:
        """
        Classify the input signal with optional noise reduction and modulation detection.
        
        Args:
            samples: Input signal samples (real or complex)
            noise_reference: Optional reference noise signal
            detect_modulation: Whether to detect and handle modulation
            
        Returns:
            ClassificationResult object containing classification results
        """
        # Preprocess signal
        if noise_reference is not None:
            samples = self.preprocess_signal(samples, noise_reference)
            
        # Detect and handle modulation if requested
        if detect_modulation:
            modulation_type = self.modulation_detector.detect_modulation(samples)
            demodulated = self.modulation_detector.demodulate(samples, modulation_type)
            samples = demodulated
            
        if self.model_type in ['cnn', 'lstm']:
            return self._classify_deep_learning(samples)
        else:
            return self._classify_sklearn(samples)
            
    def _classify_sklearn(self, samples: np.ndarray) -> ClassificationResult:
        """Classify using scikit-learn models."""
        # Extract features
        features = self.extract_features(samples)
        
        # Convert features to array
        feature_vector = np.array([features[name] for name in self.feature_names])
        
        # Scale features
        feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Get predicted class
        predicted_class = self.model.classes_[np.argmax(probabilities)]
        
        # Calculate confidence
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(self.model.classes_, probabilities)
        }
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            features=features,
            metadata={
                'model_type': self.model_type,
                'feature_count': len(self.feature_names)
            }
        )
        
    def _classify_deep_learning(self, samples: np.ndarray) -> ClassificationResult:
        """Classify using deep learning models."""
        # Prepare input
        if len(samples) != 1024:  # Resize to expected input size
            samples = cast(NDArray[np.float64], signal.resample(samples, 1024))
        x = torch.FloatTensor(samples).unsqueeze(0).to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        # Get predicted class
        predicted_class = str(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            str(i): float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        # Extract features for metadata
        features = self.extract_features(samples)
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            features=features,
            metadata={
                'model_type': self.model_type,
                'feature_count': len(self.feature_names)
            }
        )
        
    def train(self,
             X: np.ndarray,
             y: np.ndarray,
             test_size: float = 0.2,
             random_state: int = 42,
             n_splits: int = 5) -> ClassificationMetrics:
        """
        Train the classifier with cross-validation and hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            random_state: Random seed
            n_splits: Number of cross-validation folds
            
        Returns:
            Dictionary of training metrics
        """
        if self.model_type in ['cnn', 'lstm']:
            return self._train_deep_learning(X, y, test_size, random_state)
        else:
            return self._train_sklearn(X, y, test_size, random_state, n_splits)
            
    def _train_sklearn(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      test_size: float,
                      random_state: int,
                      n_splits: int) -> ClassificationMetrics:
        """Train scikit-learn models with cross-validation and hyperparameter tuning."""
        # Define parameter grids for different models
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1],
                'kernel': ['rbf', 'poly']
            },
            'gb': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Perform grid search with cross-validation
        if self.model_type in param_grids:
            grid_search = GridSearchCV(
                self.model,
                param_grids[self.model_type],
                cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)
            
        # Evaluate model
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_dict = cast(Dict[str, Any], report)
        model_metrics = cast(ClassificationMetrics, {
            'accuracy': float(report_dict['accuracy']),
            'macro_avg_precision': float(report_dict['macro avg']['precision']),
            'macro_avg_recall': float(report_dict['macro avg']['recall']),
            'macro_avg_f1': float(report_dict['macro avg']['f1-score']),
            'best_params': grid_search.best_params_ if self.model_type in param_grids else None,
            'train_loss': None,
            'optimal_lr': None
        })
        
        return model_metrics
        
    def _train_deep_learning(self,
                            X: NDArray[np.float64],
                            y: NDArray[np.int64],
                            test_size: float,
                            random_state: int,
                            gradient_accumulation_steps: int = 4) -> ClassificationMetrics:
        """Train deep learning models with advanced techniques."""
        # Prepare data
        if X.shape[1] != 1024:  # Resize to expected input size
            X = np.array([signal.resample(x, 1024) for x in X])
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create datasets and dataloaders
        train_dataset = SignalDataset(X_train, y_train)
        test_dataset = SignalDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Find optimal learning rate
        lrs, losses = self.find_learning_rate(train_loader)
        optimal_lr = lrs[np.argmin(losses)]
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=optimal_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimal_lr,
            epochs=50,
            steps_per_epoch=len(train_loader)
        )
        
        # Training loop with mixed precision and gradient accumulation
        best_accuracy = 0
        for epoch in range(50):
            self.model.train()
            train_loss = 0
            optimizer.zero_grad()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.grad_scaler.scale(loss).backward()
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                    optimizer.zero_grad()
                    
                scheduler.step()
                train_loss += loss.item() * gradient_accumulation_steps
                
            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            accuracy = 100 * correct / total
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                
        return cast(ClassificationMetrics, {
            'accuracy': float(best_accuracy / 100),
            'macro_avg_precision': 0.0,  # Not available for deep learning
            'macro_avg_recall': 0.0,     # Not available for deep learning
            'macro_avg_f1': 0.0,         # Not available for deep learning
            'best_params': None,
            'train_loss': float(train_loss / len(train_loader)),
            'optimal_lr': float(optimal_lr)
        })
        
    def find_learning_rate(self, train_loader: DataLoader, init_value: float = 1e-8, 
                          final_value: float = 10., beta: float = 0.98) -> Tuple[List[float], List[float]]:
        """
        Find optimal learning rate using learning rate finder.
        
        Args:
            train_loader: Training data loader
            init_value: Initial learning rate
            final_value: Final learning rate
            beta: Smoothing factor
            
        Returns:
            Tuple of (learning rates, losses)
        """
        num = len(train_loader) - 1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        
        for data in train_loader:
            batch_num += 1
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                
            # Backward pass with gradient scaling
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
            
            avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
                
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
                
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
            
        return log_lrs, losses
        
    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model_type in ['cnn', 'lstm']:
            model_data = {
                'model_state': self.model.state_dict(),
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            torch.save(model_data, path)
        else:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            joblib.dump(model_data, path)
        
    def load_model(self, path: str):
        """Load a trained model from disk."""
        if self.model_type in ['cnn', 'lstm']:
            model_data = torch.load(path)
            self.model.load_state_dict(model_data['model_state'])
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
        else:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores if available."""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return {
                name: float(imp)
                for name, imp in zip(self.feature_names, importance)
            }
        else:
            return {}

    def train_ensemble(self,
                      X: NDArray[np.float64],
                      y: NDArray[np.int64],
                      test_size: float = 0.2,
                      random_state: int = 42) -> Dict[str, float]:
        """
        Train an ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            random_state: Random seed
            
        Returns:
            Dictionary of training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train each model in the ensemble
        metrics: List[ClassificationMetrics] = []
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            logger.info(f"Training model {i+1}/{self.ensemble_size}")
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if isinstance(model, (CNNModel, LSTMModel, ResNetModel, TransformerModel, AttentionModel)):
                model_metrics = self._train_deep_learning(
                    X_train_scaled, y_train, test_size, random_state
                )
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_dict = cast(Dict[str, Any], report)
                model_metrics = cast(ClassificationMetrics, {
                    'accuracy': float(report_dict['accuracy']),
                    'macro_avg_precision': float(report_dict['macro avg']['precision']),
                    'macro_avg_recall': float(report_dict['macro avg']['recall']),
                    'macro_avg_f1': float(report_dict['macro avg']['f1-score']),
                    'best_params': None,
                    'train_loss': None,
                    'optimal_lr': None
                })
                
            metrics.append(model_metrics)
            
        # Calculate ensemble metrics
        ensemble_metrics = {
            'mean_accuracy': float(np.mean([m['accuracy'] for m in metrics])),
            'std_accuracy': float(np.std([m['accuracy'] for m in metrics])),
            'best_accuracy': float(np.max([m['accuracy'] for m in metrics])),
            'worst_accuracy': float(np.min([m['accuracy'] for m in metrics]))
        }
        
        return ensemble_metrics

    def record_signal(self, samples: Union[NDArray[np.float64], NDArray[np.complex128]], 
                     filename: Optional[str] = None) -> str:
        """
        Record signal to WAV file.
        
        Args:
            samples: Input signal samples (real or complex)
            filename: Optional filename
            
        Returns:
            Path to saved recording
        """
        return self.recorder.record_signal(samples, filename)
        
    def play_signal(self, filepath: str) -> None:
        """
        Play recorded signal.
        
        Args:
            filepath: Path to WAV file
        """
        self.recorder.play_signal(filepath)
        
    def get_recordings(self) -> List[str]:
        """
        Get list of available recordings.
        
        Returns:
            List of recording filepaths
        """
        return self.recorder.get_recordings() 

    def add_sdr_device(self, device_id: str, device_type: str, config: SDRConfig) -> bool:
        """
        Add an SDR device.
        
        Args:
            device_id: Unique identifier for the device
            device_type: Type of SDR device ('rtlsdr', 'hackrf', 'usrp')
            config: Device configuration
            
        Returns:
            True if device was added successfully
        """
        return self.sdr_manager.add_device(device_id, device_type, config)
        
    def remove_sdr_device(self, device_id: str) -> None:
        """Remove an SDR device."""
        self.sdr_manager.remove_device(device_id)
        
    def read_sdr_samples(self, device_id: str, num_samples: int) -> Optional[NDArray[np.complex128]]:
        """Read samples from a specific SDR device."""
        return self.sdr_manager.read_samples(device_id, num_samples)
        
    def write_sdr_samples(self, device_id: str, samples: NDArray[np.complex128]) -> bool:
        """Write samples to a specific SDR device."""
        return self.sdr_manager.write_samples(device_id, samples)
        
    def close_sdr_devices(self) -> None:
        """Close all SDR devices."""
        self.sdr_manager.close_all()

    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API server."""
        self.api.start(host, port)

class WebSocketManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self, sdr_manager: SDRManager):
        self.active_connections: Set[WebSocket] = set()
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        self.sdr_manager = sdr_manager
        
    async def connect(self, websocket: WebSocket) -> None:
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections.add(websocket)
        
    def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket client."""
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: str) -> None:
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)
                
    async def stream_samples(self, device_id: str, num_samples: int = 1024) -> None:
        """Stream samples from a device to all connected clients."""
        while True:
            try:
                samples = self.sdr_manager.read_samples(device_id, num_samples)
                if samples is not None:
                    # Convert samples to JSON-serializable format
                    data = {
                        'device_id': device_id,
                        'timestamp': time.time(),
                        'samples': samples.tolist()
                    }
                    await self.broadcast(json.dumps(data))
                await asyncio.sleep(0.1)  # 100ms delay
            except Exception as e:
                logging.error(f"Error streaming samples: {str(e)}")
                await asyncio.sleep(1.0)
                
    def start_streaming(self, device_id: str) -> None:
        """Start streaming from a device."""
        if device_id not in self.streaming_tasks:
            self.streaming_tasks[device_id] = asyncio.create_task(
                self.stream_samples(device_id)
            )
            
    def stop_streaming(self, device_id: str) -> None:
        """Stop streaming from a device."""
        if device_id in self.streaming_tasks:
            self.streaming_tasks[device_id].cancel()
            del self.streaming_tasks[device_id]

class User(BaseModel):
    """User model for authentication."""
    username: str
    password_hash: str
    role: str
    device_id: Optional[str] = None

class Token(BaseModel):
    """Token model for authentication."""
    access_token: str
    token_type: str

class SecurityManager:
    """Manages security and encryption."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.users: Dict[str, User] = {}
        self.encryption_keys: Dict[str, bytes] = {}
        
    def create_user(self, username: str, password: str, role: str, device_id: Optional[str] = None) -> bool:
        """Create a new user."""
        if username in self.users:
            return False
            
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        self.users[username] = User(
            username=username,
            password_hash=password_hash.decode(),
            role=role,
            device_id=device_id
        )
        return True
        
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = self.users.get(username)
        if not user:
            return None
            
        if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            return None
            
        return user
        
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        expires_delta = timedelta(minutes=30)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": user.username,
            "role": user.role,
            "device_id": user.device_id,
            "exp": expire
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None
            
    def generate_encryption_key(self, device_id: str) -> bytes:
        """Generate encryption key for device."""
        if device_id not in self.encryption_keys:
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.secret_key.encode()))
            self.encryption_keys[device_id] = key
        return self.encryption_keys[device_id]
        
    def encrypt_data(self, data: bytes, device_id: str) -> bytes:
        """Encrypt data for device."""
        key = self.generate_encryption_key(device_id)
        f = Fernet(key)
        return f.encrypt(data)
        
    def decrypt_data(self, encrypted_data: bytes, device_id: str) -> bytes:
        """Decrypt data from device."""
        key = self.generate_encryption_key(device_id)
        f = Fernet(key)
        return f.decrypt(encrypted_data)

class SignalClassifierAPI:
    """REST API and WebSocket server for SignalClassifier."""
    
    def __init__(self, classifier: 'SignalClassifier', secret_key: str = "your-secret-key-here"):
        self.classifier = classifier
        self.app = FastAPI(title="Signal Classifier API")
        self.ws_manager = WebSocketManager(self.classifier.sdr_manager)
        self.device_manager = NetworkDeviceManager()
        self.security_manager = SecurityManager(secret_key)
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.post("/token")
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            """Login endpoint."""
            user = self.security_manager.authenticate_user(form_data.username, form_data.password)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            access_token = self.security_manager.create_access_token(user)
            return Token(access_token=access_token, token_type="bearer")
            
        @self.app.post("/users")
        async def create_user(username: str, password: str, role: str, device_id: Optional[str] = None):
            """Create new user."""
            success = self.security_manager.create_user(username, password, role, device_id)
            if not success:
                raise HTTPException(status_code=400, detail="Username already exists")
            return {"success": True}
            
        @self.app.post("/discovery/start")
        async def start_discovery(token: str = Depends(self.security_manager.oauth2_scheme)):
            """Start device discovery."""
            payload = self.security_manager.verify_token(token)
            if not payload or payload["role"] not in ["admin", "operator"]:
                raise HTTPException(status_code=403, detail="Not authorized")
                
            self.device_manager.start_discovery()
            return {"success": True}
            
        @self.app.post("/discovery/stop")
        async def stop_discovery(token: str = Depends(self.security_manager.oauth2_scheme)):
            """Stop device discovery."""
            payload = self.security_manager.verify_token(token)
            if not payload or payload["role"] not in ["admin", "operator"]:
                raise HTTPException(status_code=403, detail="Not authorized")
                
            self.device_manager.stop_discovery()
            return {"success": True}
            
        @self.app.get("/discovery/devices")
        async def get_discovered_devices(token: str = Depends(self.security_manager.oauth2_scheme)):
            """Get list of discovered devices."""
            payload = self.security_manager.verify_token(token)
            if not payload:
                raise HTTPException(status_code=401, detail="Invalid token")
                
            return self.device_manager.get_discovered_devices()
            
        @self.app.post("/discovery/register")
        async def register_device_discovery(
            name: str,
            port: int,
            properties: Dict[str, str],
            token: str = Depends(self.security_manager.oauth2_scheme)
        ):
            """Register this device for discovery."""
            payload = self.security_manager.verify_token(token)
            if not payload or payload["role"] not in ["admin", "device"]:
                raise HTTPException(status_code=403, detail="Not authorized")
                
            self.device_manager.register_device(name, port, properties)
            return {"success": True}
            
        @self.app.post("/discovery/connect")
        async def connect_to_device_discovery(
            device_name: str,
            token: str = Depends(self.security_manager.oauth2_scheme)
        ):
            """Connect to a discovered device."""
            payload = self.security_manager.verify_token(token)
            if not payload or payload["role"] not in ["admin", "operator"]:
                raise HTTPException(status_code=403, detail="Not authorized")
                
            devices = self.device_manager.get_discovered_devices()
            if device_name not in devices:
                return {"error": "Device not found"}
                
            device_info = devices[device_name]
            try:
                # Create connection to remote device
                config = SDRConfig(
                    center_freq=float(device_info['properties'].get('center_freq', 100e6)),
                    sample_rate=float(device_info['properties'].get('sample_rate', 2.4e6)),
                    gain=float(device_info['properties'].get('gain', 40)),
                    bandwidth=float(device_info['properties'].get('bandwidth', 2.4e6)),
                    device_args={
                        'address': device_info['address'],
                        'port': device_info['port']
                    }
                )
                
                success = self.classifier.add_sdr_device(
                    device_name,
                    device_info['properties'].get('type', 'rtlsdr'),
                    config
                )
                
                return {"success": success}
            except Exception as e:
                return {"error": str(e)}
                
        @self.app.websocket("/ws/{device_id}")
        async def websocket_endpoint(websocket: WebSocket, device_id: str, token: str):
            """WebSocket endpoint for real-time streaming."""
            payload = self.security_manager.verify_token(token)
            if not payload:
                await websocket.close(code=4001)
                return
                
            await self.ws_manager.connect(websocket)
            try:
                self.ws_manager.start_streaming(device_id)
                while True:
                    data = await websocket.receive_bytes()
                    # Decrypt received data
                    decrypted_data = self.security_manager.decrypt_data(data, device_id)
                    # Process decrypted data
                    await websocket.send_bytes(decrypted_data)
            except WebSocketDisconnect:
                self.ws_manager.disconnect(websocket)
                self.ws_manager.stop_streaming(device_id)
                
        @self.app.post("/classify")
        async def classify_signal(device_id: str, num_samples: int = 1024):
            """Classify signal from a device."""
            samples = self.classifier.read_sdr_samples(device_id, num_samples)
            if samples is not None:
                result = self.classifier.classify_signal(samples)
                return result
            return {"error": "Failed to read samples"}
            
        @self.app.post("/train")
        async def train_model(data: Dict[str, Any]):
            """Train the model with provided data."""
            try:
                X = np.array(data['features'])
                y = np.array(data['labels'])
                metrics = self.classifier.train(X, y)
                return {"success": True, "metrics": metrics}
            except Exception as e:
                return {"error": str(e)}
                
        @self.app.post("/save")
        async def save_model(path: str):
            """Save the current model."""
            try:
                self.classifier.save_model(path)
                return {"success": True}
            except Exception as e:
                return {"error": str(e)}
                
        @self.app.post("/load")
        async def load_model(path: str):
            """Load a model from file."""
            try:
                self.classifier.load_model(path)
                return {"success": True}
            except Exception as e:
                return {"error": str(e)}
                
        @self.app.get("/features")
        async def get_feature_importance():
            """Get feature importance scores."""
            try:
                importance = self.classifier.get_feature_importance()
                return {"success": True, "importance": importance}
            except Exception as e:
                return {"error": str(e)}
                
        @self.app.post("/discovery/register")
        async def register_device_discovery(
            name: str,
            port: int,
            properties: Dict[str, str],
            token: str = Depends(self.security_manager.oauth2_scheme)
        ):
            """Register this device for discovery."""
            payload = self.security_manager.verify_token(token)
            if not payload or payload["role"] not in ["admin", "device"]:
                raise HTTPException(status_code=403, detail="Not authorized")
                
            self.device_manager.register_device(name, port, properties)
            return {"success": True}
            
        @self.app.post("/discovery/connect")
        async def connect_to_device_discovery(
            device_name: str,
            token: str = Depends(self.security_manager.oauth2_scheme)
        ):
            """Connect to a discovered device."""
            payload = self.security_manager.verify_token(token)
            if not payload or payload["role"] not in ["admin", "operator"]:
                raise HTTPException(status_code=403, detail="Not authorized")
                
            devices = self.device_manager.get_discovered_devices()
            if device_name not in devices:
                return {"error": "Device not found"}
                
            device_info = devices[device_name]
            try:
                # Create connection to remote device
                config = SDRConfig(
                    center_freq=float(device_info['properties'].get('center_freq', 100e6)),
                    sample_rate=float(device_info['properties'].get('sample_rate', 2.4e6)),
                    gain=float(device_info['properties'].get('gain', 40)),
                    bandwidth=float(device_info['properties'].get('bandwidth', 2.4e6)),
                    device_args={
                        'address': device_info['address'],
                        'port': device_info['port']
                    }
                )
                
                success = self.classifier.add_sdr_device(
                    device_name,
                    device_info['properties'].get('type', 'rtlsdr'),
                    config
                )
                
                return {"success": success}
            except Exception as e:
                return {"error": str(e)}
            
    def start(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the API server."""
        uvicorn.run(self.app, host=host, port=port)

class DeviceDiscoveryListener(ServiceListener):
    """Listener for device discovery events."""
    
    def __init__(self):
        self.devices: Dict[str, Dict[str, Any]] = {}
        
    def add_service(self, zeroconf: zeroconf.Zeroconf, type: str, name: str) -> None:
        """Handle new device discovery."""
        info = zeroconf.get_service_info(type, name)
        if info:
            device_info = {
                'name': name,
                'type': type,
                'address': socket.inet_ntoa(info.addresses[0]),
                'port': info.port,
                'properties': info.properties
            }
            self.devices[name] = device_info
            logging.info(f"Discovered device: {name} at {device_info['address']}:{device_info['port']}")
            
    def remove_service(self, zeroconf: zeroconf.Zeroconf, type: str, name: str) -> None:
        """Handle device removal."""
        if name in self.devices:
            del self.devices[name]
            logging.info(f"Device removed: {name}")
            
    def update_service(self, zeroconf: zeroconf.Zeroconf, type: str, name: str) -> None:
        """Handle device update."""
        self.add_service(zeroconf, type, name)

class NetworkDeviceManager:
    """Manages network device discovery and connection."""
    
    def __init__(self):
        self.zeroconf = zeroconf.Zeroconf()
        self.listener = DeviceDiscoveryListener()
        self.browser = None
        self.discovery_running = False
        self.discovery_thread = None
        self.service_type = "_sdr._tcp.local."
        
    def start_discovery(self) -> None:
        """Start device discovery."""
        if self.discovery_running:
            return
            
        self.discovery_running = True
        self.browser = ServiceBrowser(self.zeroconf, self.service_type, self.listener)
        logging.info("Started device discovery")
        
    def stop_discovery(self) -> None:
        """Stop device discovery."""
        if not self.discovery_running:
            return
            
        self.discovery_running = False
        if self.browser:
            self.browser.cancel()
        self.zeroconf.close()
        logging.info("Stopped device discovery")
        
    def get_discovered_devices(self) -> Dict[str, Dict[str, Any]]:
        """Get list of discovered devices."""
        return self.listener.devices
        
    def register_device(self, name: str, port: int, properties: Dict[str, str]) -> None:
        """Register this device for discovery."""
        info = ServiceInfo(
            self.service_type,
            f"{name}.{self.service_type}",
            addresses=[socket.inet_aton(socket.gethostbyname(socket.gethostname()))],
            port=port,
            properties=properties
        )
        self.zeroconf.register_service(info)
        logging.info(f"Registered device: {name}")
