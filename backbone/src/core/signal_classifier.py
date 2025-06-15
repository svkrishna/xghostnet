"""
Machine learning-based signal classification for GhostNet.
Implements various ML models for signal classification and modulation recognition.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, cast, TypedDict, TypeVar, cast, overload, Set, Mapping
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
from scipy.fft import fft, fftfreq, rfft, rfftfreq
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
from .signal_geolocator import SignalGeolocator
from .signal_types import ModulationType, SignalFeatures, ClassificationResult, DetectionResult, SignalMetrics

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

class SignalDetector:
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        """Initialize signal detector with real-time processing capabilities.
        
        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Size of processing buffer
        """
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

    def process_samples(self, samples: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
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
        snr = metrics['snr']
        signal_detected = snr > self.snr_threshold
        
        return signal_detected, snr, metrics
        
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
        
    def get_signal_quality(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate signal quality metrics."""
        # Calculate signal power
        signal_power = np.mean(samples**2)
        signal_db = 10 * np.log10(signal_power)
        
        # Calculate SNR
        snr = signal_db - self.noise_floor
        
        # Calculate EVM (Error Vector Magnitude)
        evm = np.sqrt(np.mean((samples - np.mean(samples))**2)) / np.sqrt(signal_power)
        
        # Calculate crest factor
        crest_factor = np.max(np.abs(samples)) / np.sqrt(signal_power)
        
        # Calculate RMS value
        rms = np.sqrt(signal_power)
        
        return {
            'snr': float(snr),
            'signal_strength': float(signal_db),
            'evm': float(evm),
            'crest_factor': float(crest_factor),
            'rms': float(rms),
            'noise_floor': float(self.noise_floor)
        }
        
    def detect_modulation(self, samples: np.ndarray) -> str:
        """Detect modulation type of the signal."""
        # Calculate signal statistics
        magnitude = np.abs(samples)
        phase = np.angle(samples)
        
        # Calculate features
        magnitude_std = np.std(magnitude)
        phase_std = np.std(phase)
        magnitude_skew = scipy.stats.skew(magnitude)
        phase_skew = scipy.stats.skew(phase)
        
        # Detect modulation type based on features
        if magnitude_std < 0.1 and phase_std > 0.5:
            return 'PSK'
        elif magnitude_std > 0.5 and phase_std < 0.1:
            return 'ASK'
        elif magnitude_std < 0.1 and phase_std < 0.1:
            return 'FSK'
        else:
            return 'Unknown'

class SignalProcessor:
    """Advanced signal processing with rich feature extraction."""
    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.window = np.hanning(fft_size)
        
    def extract_spectral_features(self, samples: np.ndarray) -> Dict[str, float]:
        """Extract spectral features from signal."""
        # Apply window and compute FFT
        windowed = samples[:self.fft_size] * self.window
        spectrum = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(self.fft_size, 1/self.sample_rate)
        
        # Calculate spectral features
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Spectral centroid
        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum((freqs - centroid)**2 * magnitude) / (np.sum(magnitude) + 1e-10))
        
        # Spectral flatness
        flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)
        
        # Spectral rolloff
        rolloff_threshold = 0.85
        rolloff = freqs[np.where(np.cumsum(magnitude) >= rolloff_threshold * np.sum(magnitude))[0][0]]
        
        # Spectral kurtosis
        kurtosis = np.sum((magnitude - np.mean(magnitude))**4) / (np.sum(magnitude)**2 + 1e-10)
        
        return {
            'spectral_centroid': float(centroid),
            'spectral_bandwidth': float(bandwidth),
            'spectral_flatness': float(flatness),
            'spectral_rolloff': float(rolloff),
            'spectral_kurtosis': float(kurtosis)
        }
        
    def extract_temporal_features(self, samples: np.ndarray) -> Dict[str, float]:
        """Extract temporal features from signal."""
        # Zero crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.signbit(samples)))))
        
        # Root mean square
        rms = float(np.sqrt(np.mean(np.square(samples))))
        
        # Crest factor
        crest = float(np.max(np.abs(samples)) / (rms + 1e-10))
        
        # Energy
        energy = float(np.sum(np.square(samples)))
        
        # Entropy
        hist, _ = np.histogram(samples, bins=50, density=True)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        return {
            'zero_crossing_rate': zcr,
            'rms': rms,
            'crest_factor': crest,
            'energy': energy,
            'entropy': entropy
        }
        
    def extract_modulation_features(self, samples: np.ndarray) -> Dict[str, float]:
        """Extract modulation-specific features."""
        # Amplitude features
        amplitude_mean = float(np.mean(np.abs(samples)))
        amplitude_std = float(np.std(np.abs(samples)))
        amplitude_skew = float(scipy.stats.skew(np.abs(samples)))
        
        # Phase features
        phase = np.unwrap(np.angle(scipy.signal.hilbert(samples)))
        phase_diff = np.diff(phase)
        phase_std = float(np.std(phase_diff))
        phase_skew = float(scipy.stats.skew(phase_diff))
        
        # Frequency features
        inst_freq = np.diff(phase) / (2 * np.pi) * self.sample_rate
        freq_mean = float(np.mean(inst_freq))
        freq_std = float(np.std(inst_freq))
        
        return {
            'amplitude_mean': amplitude_mean,
            'amplitude_std': amplitude_std,
            'amplitude_skew': amplitude_skew,
            'phase_std': phase_std,
            'phase_skew': phase_skew,
            'freq_mean': freq_mean,
            'freq_std': freq_std
        }
        
    def extract_statistical_features(self, samples: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from signal."""
        # Basic statistics
        mean = float(np.mean(samples))
        std = float(np.std(samples))
        skew = float(scipy.stats.skew(samples))
        kurtosis = float(scipy.stats.kurtosis(samples))
        
        # Percentiles
        percentiles = np.percentile(samples, [25, 50, 75])
        
        # Range
        signal_range = float(np.max(samples) - np.min(samples))
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skew,
            'kurtosis': kurtosis,
            'q1': float(percentiles[0]),
            'median': float(percentiles[1]),
            'q3': float(percentiles[2]),
            'range': signal_range
        }
        
    def extract_all_features(self, samples: np.ndarray) -> Dict[str, float]:
        """Extract all available features from signal."""
        features = {}
        
        # Extract features from each category
        features.update(self.extract_spectral_features(samples))
        features.update(self.extract_temporal_features(samples))
        features.update(self.extract_modulation_features(samples))
        features.update(self.extract_statistical_features(samples))
        
        return features
        
    def detect_signal_characteristics(self, samples: np.ndarray) -> Dict[str, Any]:
        """Detect signal characteristics and modulation type."""
        features = self.extract_all_features(samples)
        
        # Detect modulation type
        if features['spectral_flatness'] > 0.8:
            modulation = 'FM'
        elif features['spectral_bandwidth'] < 1000:
            modulation = 'AM'
        elif features['amplitude_std'] / features['amplitude_mean'] > 0.5:
            modulation = 'ASK'
        else:
            modulation = 'PSK'
            
        # Detect signal quality
        snr = 10 * np.log10(features['energy'] / (features['std']**2 + 1e-10))
        quality = 'good' if snr > 20 else 'fair' if snr > 10 else 'poor'
        
        return {
            'modulation_type': modulation,
            'signal_quality': quality,
            'snr': float(snr),
            'features': features
        }
