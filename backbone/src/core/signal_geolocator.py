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
        
        # Propagation model parameters
        self.propagation_models = {
            'free_space': self._free_space_path_loss,
            'two_ray': self._two_ray_path_loss,
            'log_distance': self._log_distance_path_loss,
            'okumura_hata': self._okumura_hata_path_loss
        }
        
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
        
    def _calculate_distance(self, signal_strength: float, reference_strength: float, 
                          path_loss_exponent: float = 2.0) -> float:
        """Calculate distance using log-distance path loss model.
        
        Args:
            signal_strength: Received signal strength in dBm
            reference_strength: Reference signal strength at 1m in dBm
            path_loss_exponent: Path loss exponent (n)
            
        Returns:
            Estimated distance in meters
        """
        # Guard against extremely weak signals
        if signal_strength < -100:  # dBm threshold
            return float('inf')
            
        # Log-distance path loss model
        distance = 10 ** ((reference_strength - signal_strength) / (10 * path_loss_exponent))
        return max(distance, 0.1)  # Minimum distance of 0.1m
    
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
        
    def _free_space_path_loss(self, distance: float, frequency: float = 2.4e9) -> float:
        """Free space path loss model.
        
        Args:
            distance: Distance in meters
            frequency: Frequency in Hz (default: 2.4 GHz)
            
        Returns:
            Path loss in dB
        """
        wavelength = 3e8 / frequency
        return 20 * np.log10(4 * np.pi * distance / wavelength)
        
    def _two_ray_path_loss(self, distance: float, tx_height: float = 1.5, 
                          rx_height: float = 1.5, frequency: float = 2.4e9) -> float:
        """Two-ray ground reflection model.
        
        Args:
            distance: Distance in meters
            tx_height: Transmitter height in meters
            rx_height: Receiver height in meters
            frequency: Frequency in Hz
            
        Returns:
            Path loss in dB
        """
        wavelength = 3e8 / frequency
        d_cross = 4 * np.pi * tx_height * rx_height / wavelength
        
        if distance < d_cross:
            return self._free_space_path_loss(distance, frequency)
        else:
            return 40 * np.log10(distance) - 20 * np.log10(tx_height) - 20 * np.log10(rx_height)
            
    def _log_distance_path_loss(self, distance: float, reference_distance: float = 1.0,
                              path_loss_exponent: float = 2.0, frequency: float = 2.4e9) -> float:
        """Log-distance path loss model.
        
        Args:
            distance: Distance in meters
            reference_distance: Reference distance in meters
            path_loss_exponent: Path loss exponent
            frequency: Frequency in Hz
            
        Returns:
            Path loss in dB
        """
        pl0 = self._free_space_path_loss(reference_distance, frequency)
        return pl0 + 10 * path_loss_exponent * np.log10(distance / reference_distance)
        
    def _okumura_hata_path_loss(self, distance: float, frequency: float = 2.4e9,
                               tx_height: float = 30, rx_height: float = 1.5,
                               environment: str = 'urban') -> float:
        """Okumura-Hata path loss model.
        
        Args:
            distance: Distance in kilometers
            frequency: Frequency in MHz
            tx_height: Transmitter height in meters
            rx_height: Receiver height in meters
            environment: Environment type ('urban', 'suburban', 'rural')
            
        Returns:
            Path loss in dB
        """
        # Convert frequency to MHz
        freq_mhz = frequency / 1e6
        
        # Calculate base path loss
        a_hr = (1.1 * np.log10(freq_mhz) - 0.7) * rx_height - (1.56 * np.log10(freq_mhz) - 0.8)
        pl = 69.55 + 26.16 * np.log10(freq_mhz) - 13.82 * np.log10(tx_height) - a_hr + \
             (44.9 - 6.55 * np.log10(tx_height)) * np.log10(distance)
             
        # Apply environment correction
        if environment == 'suburban':
            pl -= 2 * (np.log10(freq_mhz / 28))**2 - 5.4
        elif environment == 'rural':
            pl -= 4.78 * (np.log10(freq_mhz))**2 - 18.33 * np.log10(freq_mhz) - 40.98
            
        return pl
        
    def _trilaterate(self, distances: Dict[str, float], 
                    device_locations: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float]:
        """Trilateration using multiple known points and distances.
        
        Args:
            distances: Dictionary of device_id to distance
            device_locations: Dictionary of device_id to (latitude, longitude)
            
        Returns:
            Tuple of (latitude, longitude, uncertainty)
        """
        if len(distances) < 3:
            raise ValueError("At least 3 points required for trilateration")
            
        # Convert to numpy arrays
        points = np.array([device_locations[dev_id] for dev_id in distances.keys()])
        dists = np.array([distances[dev_id] for dev_id in distances.keys()])
        
        # Initial guess (centroid of known points)
        x0 = np.mean(points, axis=0)
        
        def objective(x):
            """Objective function: sum of squared differences between
            calculated and measured distances."""
            return np.sum((np.sqrt(np.sum((points - x)**2, axis=1)) - dists)**2)
            
        # Optimize position
        result = minimize(objective, x0, method='Nelder-Mead')
        
        # Calculate uncertainty based on residuals
        residuals = np.sqrt(np.sum((points - result.x)**2, axis=1)) - dists
        uncertainty = np.sqrt(np.mean(residuals**2))
        
        return result.x[0], result.x[1], uncertainty
        
    def _validate_inputs(self, signal_strengths: Dict[str, float]) -> bool:
        """Validate input signal strengths.
        
        Args:
            signal_strengths: Dictionary of device_id to signal strength
            
        Returns:
            True if inputs are valid, False otherwise
        """
        if not signal_strengths:
            self.logger.warning("No signal strengths provided")
            return False
            
        # Check if all devices exist
        unknown_devices = set(signal_strengths.keys()) - set(self.known_devices.keys())
        if unknown_devices:
            self.logger.warning(f"Unknown devices: {unknown_devices}")
            return False
            
        # Check for valid signal strengths
        for device_id, strength in signal_strengths.items():
            if not isinstance(strength, (int, float)):
                self.logger.warning(f"Invalid signal strength for device {device_id}")
                return False
            if strength > 0:  # Signal strength should be negative in dBm
                self.logger.warning(f"Unrealistic signal strength for device {device_id}")
                return False
                
        return True
        
    def estimate_position(self, signal_strengths: Dict[str, float],
                         timestamp: float,
                         use_fingerprinting: bool = True,
                         propagation_model: str = 'log_distance',
                         path_loss_exponent: float = 2.0) -> Optional[Tuple[float, float, float]]:
        """Estimate position using either fingerprinting or trilateration with propagation models.
        
        Args:
            signal_strengths: Dictionary of device_id to signal strength in dBm
            timestamp: Current timestamp
            use_fingerprinting: Whether to use fingerprinting or trilateration
            propagation_model: Propagation model to use
            path_loss_exponent: Path loss exponent for log-distance model
            
        Returns:
            Tuple of (latitude, longitude, uncertainty) or None if estimation fails
        """
        # Validate inputs
        if not self._validate_inputs(signal_strengths):
            return None
            
        if use_fingerprinting and self.fingerprints:
            return self.estimate_position_fingerprinting(signal_strengths)
            
        # Convert signal strengths to distances using propagation model
        distances = {}
        for device_id, strength in signal_strengths.items():
            if device_id in self.known_devices:
                device = self.known_devices[device_id]
                # Convert signal strength to distance using propagation model
                path_loss = self.propagation_models[propagation_model](
                    distance=1.0,  # Reference distance
                    frequency=2.4e9,  # Default frequency
                    tx_height=1.5,  # Default height
                    rx_height=1.5
                )
                distance = self._calculate_distance(
                    signal_strength=strength,
                    reference_strength=path_loss,
                    path_loss_exponent=path_loss_exponent
                )
                distances[device_id] = distance
                
        if len(distances) >= 3:
            device_locations = {
                dev_id: (self.known_devices[dev_id].latitude, 
                        self.known_devices[dev_id].longitude)
                for dev_id in distances.keys()
            }
            return self._trilaterate(distances, device_locations)
            
        return None

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