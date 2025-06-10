import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import logging
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

@dataclass
class DeviceLocation:
    """Represents a device's known location and signal strength."""
    device_id: str
    latitude: float
    longitude: float
    signal_strength: float  # in dBm
    timestamp: float

@dataclass
class SignalFingerprint:
    """Represents a signal fingerprint at a location."""
    location_id: str
    latitude: float
    longitude: float
    signal_strengths: Dict[str, float]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class SignalGeolocator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.known_devices: Dict[str, DeviceLocation] = {}
        self.fingerprints: List[SignalFingerprint] = []
        self.scaler = StandardScaler()
        self.calibration_mode = False
        self.calibration_data: List[SignalFingerprint] = []
        self.fingerprinting_model = None
        self.kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0], (1e-2, 1e2))
        
    def add_device_location(self, device_id: str, latitude: float, longitude: float, 
                          signal_strength: float, timestamp: float) -> None:
        """Add or update a device's known location."""
        self.known_devices[device_id] = DeviceLocation(
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            signal_strength=signal_strength,
            timestamp=timestamp
        )
        
    def remove_device_location(self, device_id: str) -> None:
        """Remove a device's known location."""
        self.known_devices.pop(device_id, None)
        
    def add_fingerprint(self, fingerprint: SignalFingerprint) -> None:
        """Add a new signal fingerprint."""
        if self.calibration_mode:
            self.calibration_data.append(fingerprint)
            self.logger.info(f"Added calibration fingerprint at {fingerprint.latitude}, {fingerprint.longitude}")
        else:
            self.fingerprints.append(fingerprint)
            self.logger.info(f"Added fingerprint at {fingerprint.latitude}, {fingerprint.longitude}")
        
    def remove_fingerprint(self, location_id: str) -> None:
        """Remove a signal fingerprint."""
        self.fingerprints = [f for f in self.fingerprints if f.location_id != location_id]
        
    def _calculate_distance(self, signal_strength: float, reference_strength: float) -> float:
        """Calculate approximate distance based on signal strength difference.
        
        Uses the inverse square law for signal propagation:
        distance = reference_distance * 10^((reference_strength - signal_strength)/(20))
        """
        # Assuming reference_distance of 1 meter and reference_strength at that distance
        return 10 ** ((reference_strength - signal_strength) / 20)
    
    def _triangulate_position(self, device_locations: List[DeviceLocation]) -> Tuple[float, float]:
        """Triangulate position using multiple known device locations."""
        if len(device_locations) < 3:
            raise ValueError("At least 3 device locations are required for triangulation")
            
        def objective_function(point):
            """Objective function to minimize: sum of squared differences between
            calculated and measured distances."""
            lat, lon = point
            total_error = 0
            
            for device in device_locations:
                # Calculate distance based on signal strength
                measured_distance = self._calculate_distance(
                    device.signal_strength,
                    -30  # Reference signal strength at 1m (typical for WiFi)
                )
                
                # Calculate actual distance using Haversine formula
                actual_distance = self._haversine_distance(
                    lat, lon,
                    device.latitude, device.longitude
                )
                
                total_error += (measured_distance - actual_distance) ** 2
                
            return total_error
            
        # Initial guess: average of known device locations
        initial_lat = np.mean([d.latitude for d in device_locations])
        initial_lon = np.mean([d.longitude for d in device_locations])
        
        # Optimize position
        result = minimize(
            objective_function,
            [initial_lat, initial_lon],
            method='Nelder-Mead'
        )
        
        return result.x[0], result.x[1]
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on the earth."""
        R = 6371000  # Earth's radius in meters
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi/2)**2 + \
            np.cos(phi1) * np.cos(phi2) * \
            np.sin(delta_lambda/2)**2
            
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _fingerprint_distance(self, fp1: Dict[str, float], fp2: Dict[str, float]) -> float:
        """Calculate distance between two signal fingerprints."""
        # Get common devices
        common_devices = set(fp1.keys()) & set(fp2.keys())
        if not common_devices:
            return float('inf')
            
        # Calculate mean squared difference of signal strengths
        differences = [
            (fp1[device] - fp2[device]) ** 2
            for device in common_devices
        ]
        return float(np.mean(differences))
        
    def estimate_position_fingerprinting(self, signal_strengths: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
        """Estimate position using advanced fingerprinting algorithms."""
        if not self.fingerprinting_model or not self.fingerprints:
            return None

        # Prepare input data
        X = np.array([list(signal_strengths.values())])
        X_scaled = self.scaler.transform(X)
        
        # Get prediction
        position = self.fingerprinting_model.predict(X_scaled)[0]
        
        # Calculate uncertainty
        if isinstance(self.fingerprinting_model, GaussianProcessRegressor):
            result = self.fingerprinting_model.predict(X_scaled, return_std=True)
            position = result[0]
            std = result[1]
            uncertainty = float(std[0])
        else:
            # For other models, use distance to nearest neighbors
            fingerprint_values = np.array([list(fp.signal_strengths.values()) for fp in self.fingerprints])
            transformed_values = self.scaler.transform(fingerprint_values)
            distances = cdist(np.asarray(X_scaled), np.asarray(transformed_values))
            uncertainty = float(np.min(distances))

        return float(position[0]), float(position[1]), uncertainty

    def cluster_fingerprints(self, eps: float = 0.5, min_samples: int = 5) -> List[List[SignalFingerprint]]:
        """Cluster similar fingerprints using DBSCAN."""
        if not self.fingerprints:
            return []

        # Extract features
        X = np.array([list(fp.signal_strengths.values()) for fp in self.fingerprints])
        X_scaled = self.scaler.fit_transform(X)

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = clustering.labels_

        # Group fingerprints by cluster
        clusters = []
        for label in set(labels):
            if label != -1:  # Skip noise points
                cluster = [fp for i, fp in enumerate(self.fingerprints) if labels[i] == label]
                clusters.append(cluster)

        return clusters

    def get_fingerprints(self) -> List[SignalFingerprint]:
        """Get all signal fingerprints."""
        return self.fingerprints
        
    def estimate_position(self, signal_strengths: Dict[str, float],
                         timestamp: float,
                         use_fingerprinting: bool = True) -> Optional[Tuple[float, float, float]]:
        """Estimate position using either triangulation or fingerprinting."""
        if use_fingerprinting and self.fingerprints:
            return self.estimate_position_fingerprinting(signal_strengths)
            
        # Fall back to triangulation
        lat, lon = self._triangulate_position([self.known_devices[device_id] for device_id in signal_strengths])
        return lat, lon, 0.0  # No uncertainty for triangulation
            
    def get_known_devices(self) -> List[DeviceLocation]:
        """Get list of all known device locations."""
        return list(self.known_devices.values())

    def start_calibration(self) -> None:
        """Start calibration mode for collecting fingerprints."""
        self.calibration_mode = True
        self.calibration_data = []
        self.logger.info("Calibration mode started")

    def stop_calibration(self) -> None:
        """Stop calibration mode and process collected data."""
        self.calibration_mode = False
        if self.calibration_data:
            self._process_calibration_data()
        self.logger.info("Calibration mode stopped")

    def _process_calibration_data(self) -> None:
        """Process calibration data to create a robust fingerprinting model."""
        # Extract features and targets
        X = np.array([list(fp.signal_strengths.values()) for fp in self.calibration_data])
        y = np.array([[fp.latitude, fp.longitude] for fp in self.calibration_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train multiple models
        models = {
            'knn': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gp': GaussianProcessRegressor(kernel=self.kernel, random_state=42)
        }
        
        # Train and evaluate models
        best_model = None
        best_score = float('inf')
        
        for name, model in models.items():
            model.fit(X_scaled, y)
            score = np.mean(np.abs(model.predict(X_scaled) - y))
            if score < best_score:
                best_score = score
                best_model = model
                self.logger.info(f"Best model: {name} with score: {score}")
        
        self.fingerprinting_model = best_model
        self.fingerprints.extend(self.calibration_data)
        self.calibration_data = []

    def get_fingerprint_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fingerprint database."""
        if not self.fingerprints:
            return {"count": 0}

        # Calculate coverage area
        lats = [fp.latitude for fp in self.fingerprints]
        lons = [fp.longitude for fp in self.fingerprints]
        
        stats = {
            "count": len(self.fingerprints),
            "coverage": {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_lon": min(lons),
                "max_lon": max(lons)
            },
            "devices": len(set(
                device_id 
                for fp in self.fingerprints 
                for device_id in fp.signal_strengths.keys()
            )),
            "clusters": len(self.cluster_fingerprints())
        }
        
        return stats 