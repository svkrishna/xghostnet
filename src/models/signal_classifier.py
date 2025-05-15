import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class SignalClassifier(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_classes: int = 3):
        """
        Initialize the signal classifier neural network.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_classes: Number of output classes (drone, vehicle, human)
        """
        super(SignalClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.class_names = ['drone', 'vehicle', 'human']
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Make a prediction from extracted features.
        
        Args:
            features: Dictionary of signal features
            
        Returns:
            Tuple of (predicted class, confidence)
        """
        # Convert features to tensor
        feature_vector = torch.tensor([
            features['spectral_entropy'],
            features['bandwidth'],
            features['burst_duration_mean'],
            features['burst_interval_mean'],
            features['peak_power'],
            features['rms_power']
        ], dtype=torch.float32)
        
        # Add batch dimension
        feature_vector = feature_vector.unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            logits = self(feature_vector)
            probabilities = F.softmax(logits, dim=1)
            
        # Get predicted class and confidence
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        return self.class_names[predicted_idx.item()], confidence.item()
    
    def train_step(self, features: List[Dict[str, float]], labels: List[int]) -> float:
        """
        Perform a single training step.
        
        Args:
            features: List of feature dictionaries
            labels: List of class labels
            
        Returns:
            Training loss
        """
        self.train()
        
        # Convert features to tensor
        feature_tensor = torch.tensor([
            [f['spectral_entropy'], f['bandwidth'], f['burst_duration_mean'],
             f['burst_interval_mean'], f['peak_power'], f['rms_power']]
            for f in features
        ], dtype=torch.float32)
        
        # Convert labels to tensor
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Forward pass
        logits = self(feature_tensor)
        loss = F.cross_entropy(logits, label_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'class_names': self.class_names
        }, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'SignalClassifier':
        """Load a saved model."""
        checkpoint = torch.load(path)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_names = checkpoint['class_names']
        return model 