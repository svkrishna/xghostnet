"""
Triangulation module for GhostNet.
Uses data from multiple receivers to locate targets in 3D space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.spatial import distance
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TargetLocation:
    """Represents a target location estimate."""
    position: Tuple[float, float, float]  # (x, y, z) in meters
    confidence: float
    timestamp: float
    classification: str
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) in m/s

class Triangulator:
    """Performs triangulation using data from multiple receivers."""
    
    def __init__(self, min_receivers: int = 3, max_distance: float = 1000.0):
        """
        Initialize the triangulator.
        
        Args:
            min_receivers: Minimum number of receivers needed for triangulation
            max_distance: Maximum valid distance between receivers in meters
        """
        self.min_receivers = min_receivers
        self.max_distance = max_distance
        self.location_history: List[TargetLocation] = []
        self.max_history = 100  # Maximum number of historical locations to keep
    
    def estimate_location(self, detections: List[Dict]) -> Optional[TargetLocation]:
        """
        Estimate target location from multiple receiver detections.
        
        Args:
            detections: List of detection dictionaries, each containing:
                - position: (x, y, z) of receiver
                - signal_features: Dictionary of signal features
                - timestamp: Detection timestamp
                - classification: Target classification
                
        Returns:
            TargetLocation object if successful, None otherwise
        """
        if len(detections) < self.min_receivers:
            logger.warning(f"Insufficient receivers for triangulation: {len(detections)} < {self.min_receivers}")
            return None
        
        # Extract receiver positions and signal features
        positions = np.array([d['position'] for d in detections])
        features = [d['signal_features'] for d in detections]
        timestamps = [d['timestamp'] for d in detections]
        classifications = [d['classification'] for d in detections]
        
        # Check if receivers are too far apart
        if not self._validate_receiver_distances(positions):
            logger.warning("Receivers are too far apart for reliable triangulation")
            return None
        
        # Estimate initial position using centroid
        initial_position = self._estimate_initial_position(positions, features)
        
        # Refine position using least squares optimization
        try:
            result = least_squares(
                self._residuals,
                initial_position,
                args=(positions, features),
                method='trf',
                loss='soft_l1'
            )
            
            if not result.success:
                logger.warning("Position optimization failed to converge")
                return None
            
            position = tuple(result.x)
            
            # Calculate confidence based on residuals
            confidence = self._calculate_confidence(result.fun)
            
            # Estimate velocity if we have historical data
            velocity = self._estimate_velocity(position, timestamps[0])
            
            # Create location estimate
            location = TargetLocation(
                position=position,
                confidence=confidence,
                timestamp=timestamps[0],
                classification=self._consensus_classification(classifications),
                velocity=velocity
            )
            
            # Update history
            self._update_history(location)
            
            return location
            
        except Exception as e:
            logger.error(f"Error during triangulation: {str(e)}")
            return None
    
    def _validate_receiver_distances(self, positions: np.ndarray) -> bool:
        """Check if receivers are within valid distance range."""
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = distance.euclidean(positions[i], positions[j])
                if dist > self.max_distance:
                    return False
        return True
    
    def _estimate_initial_position(self, positions: np.ndarray,
                                 features: List[Dict]) -> np.ndarray:
        """
        Estimate initial position using weighted centroid.
        Weights are based on signal strength and quality.
        """
        weights = np.array([
            features[i].get('rms_power', 1.0) * features[i].get('confidence', 1.0)
            for i in range(len(features))
        ])
        weights = weights / np.sum(weights)
        
        return np.average(positions, weights=weights, axis=0)
    
    def _residuals(self, position: np.ndarray, receiver_positions: np.ndarray,
                  features: List[Dict]) -> np.ndarray:
        """
        Calculate residuals for least squares optimization.
        Residuals are based on:
        1. Distance between estimated and actual signal strength
        2. Time difference of arrival (if available)
        3. Doppler shift (if available)
        """
        residuals = []
        
        # Calculate expected signal strength based on distance
        for i, receiver_pos in enumerate(receiver_positions):
            # Distance-based path loss
            distance = np.linalg.norm(position - receiver_pos)
            expected_strength = -20 * np.log10(distance)  # Simple path loss model
            
            # Compare with actual signal strength
            actual_strength = features[i].get('rms_power', 0)
            residuals.append(expected_strength - actual_strength)
            
            # Add TDOA residual if available
            if 'timestamp' in features[i]:
                # TODO: Implement TDOA calculation
                pass
            
            # Add Doppler residual if available
            if 'doppler_shift' in features[i]:
                # TODO: Implement Doppler-based residual
                pass
        
        return np.array(residuals)
    
    def _calculate_confidence(self, residuals: np.ndarray) -> float:
        """Calculate confidence score based on residuals."""
        # Normalize residuals
        normalized_residuals = residuals / np.max(np.abs(residuals))
        
        # Calculate confidence as inverse of mean squared error
        mse = np.mean(normalized_residuals ** 2)
        confidence = 1.0 / (1.0 + mse)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _estimate_velocity(self, position: np.ndarray,
                         timestamp: float) -> Optional[Tuple[float, float, float]]:
        """Estimate velocity using historical location data."""
        if not self.location_history:
            return None
        
        # Find most recent location
        recent_locations = [loc for loc in self.location_history
                          if timestamp - loc.timestamp < 5.0]  # Last 5 seconds
        
        if len(recent_locations) < 2:
            return None
        
        # Calculate velocity using least squares
        times = np.array([loc.timestamp for loc in recent_locations])
        positions = np.array([loc.position for loc in recent_locations])
        
        # Fit linear model to each coordinate
        velocities = []
        for i in range(3):  # x, y, z
            coeffs = np.polyfit(times, positions[:, i], 1)
            velocities.append(coeffs[0])  # Slope is velocity
        
        return tuple(velocities)
    
    def _consensus_classification(self, classifications: List[str]) -> str:
        """Determine consensus classification from multiple receivers."""
        # Count occurrences of each classification
        counts = {}
        for cls in classifications:
            counts[cls] = counts.get(cls, 0) + 1
        
        # Return most common classification
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _update_history(self, location: TargetLocation):
        """Update location history."""
        self.location_history.append(location)
        
        # Remove old locations
        if len(self.location_history) > self.max_history:
            self.location_history = self.location_history[-self.max_history:]
    
    def get_location_history(self) -> List[TargetLocation]:
        """Get historical location data."""
        return self.location_history
    
    def clear_history(self):
        """Clear location history."""
        self.location_history.clear() 