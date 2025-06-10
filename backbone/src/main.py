"""
Main application module for GhostNet.
Integrates all components of the passive RF detection system.
"""

import os
import sys
import logging
import time
import json
import argparse
from typing import Dict, List, Optional
import threading
from queue import Queue
import signal

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.signal_processor import SignalProcessor
from src.models.signal_classifier import SignalClassifier
from src.hardware.sdr_interface import SDRManager, SDRConfig
from src.mesh.receiver_mesh import MeshNetwork, DetectionEvent
from src.mesh.triangulation import Triangulator, TargetLocation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ghostnet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GhostNet:
    """Main GhostNet application class."""
    
    def __init__(self, config_path: str):
        """
        Initialize GhostNet system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.running = False
        
        # Initialize components
        self.signal_processor = SignalProcessor(
            sample_rate=self.config['signal_processor']['sample_rate'],
            fft_size=self.config['signal_processor']['fft_size']
        )
        
        self.signal_classifier = SignalClassifier.load_model(
            self.config['model']['path']
        )
        
        self.sdr_manager = SDRManager()
        self._initialize_sdr_devices()
        
        self.mesh_network = MeshNetwork(
            local_node_id=self.config['mesh']['node_id'],
            local_position=tuple(self.config['mesh']['position']),
            local_port=self.config['mesh']['port']
        )
        
        self.triangulator = Triangulator(
            min_receivers=self.config['triangulation']['min_receivers'],
            max_distance=self.config['triangulation']['max_distance']
        )
        
        # Queues for inter-thread communication
        self.detection_queue = Queue()
        self.location_queue = Queue()
        
        # Threads
        self._processing_thread = None
        self._triangulation_thread = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            sys.exit(1)
    
    def _initialize_sdr_devices(self):
        """Initialize SDR devices from configuration."""
        for device_id, device_config in self.config['sdr_devices'].items():
            config = SDRConfig(
                center_freq=device_config['center_freq'],
                sample_rate=device_config['sample_rate'],
                gain=device_config['gain'],
                bandwidth=device_config['bandwidth'],
                device_args=device_config.get('device_args', {})
            )
            
            if not self.sdr_manager.add_device(
                device_id,
                device_config['type'],
                config
            ):
                logger.error(f"Failed to initialize SDR device: {device_id}")
    
    def start(self):
        """Start the GhostNet system."""
        if self.running:
            return
        
        self.running = True
        
        # Start mesh network
        if not self.mesh_network.start():
            logger.error("Failed to start mesh network")
            return
        
        # Start SDR devices
        if not self.sdr_manager.start_all_devices():
            logger.error("Failed to start SDR devices")
            return
        
        # Start processing thread
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        # Start triangulation thread
        self._triangulation_thread = threading.Thread(target=self._triangulation_loop)
        self._triangulation_thread.daemon = True
        self._triangulation_thread.start()
        
        logger.info("GhostNet system started")
    
    def stop(self):
        """Stop the GhostNet system."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop SDR devices
        self.sdr_manager.stop_all_devices()
        
        # Stop mesh network
        self.mesh_network.stop()
        
        # Wait for threads to finish
        if self._processing_thread:
            self._processing_thread.join()
        if self._triangulation_thread:
            self._triangulation_thread.join()
        
        logger.info("GhostNet system stopped")
    
    def _processing_loop(self):
        """Main processing loop for signal analysis."""
        while self.running:
            try:
                # Process samples from each SDR device
                for device_id, device in self.sdr_manager.devices.items():
                    samples = device.read_samples(1024)
                    if samples is not None:
                        # Extract features
                        features = self.signal_processor.extract_features(samples)
                        
                        # Classify signal
                        classification, confidence = self.signal_classifier.predict(features)
                        
                        # Create detection event
                        detection = DetectionEvent(
                            timestamp=time.time(),
                            node_id=self.config['mesh']['node_id'],
                            signal_features=features,
                            confidence=confidence,
                            classification=classification
                        )
                        
                        # Add to detection queue
                        self.detection_queue.put(detection)
                        
                        # Broadcast detection to mesh network
                        self.mesh_network.broadcast_detection(detection)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)
    
    def _triangulation_loop(self):
        """Main triangulation loop for target location."""
        while self.running:
            try:
                # Get detection from queue
                detection = self.detection_queue.get(timeout=1.0)
                
                # Get detections from other nodes
                other_detections = self._get_recent_detections()
                
                # Combine with local detection
                all_detections = [{
                    'position': self.config['mesh']['position'],
                    'signal_features': detection.signal_features,
                    'timestamp': detection.timestamp,
                    'classification': detection.classification
                }]
                
                all_detections.extend(other_detections)
                
                # Estimate target location
                location = self.triangulator.estimate_location(all_detections)
                
                if location:
                    # Add to location queue for visualization/storage
                    self.location_queue.put(location)
                    
                    # Log location
                    logger.info(
                        f"Target located at {location.position} "
                        f"(confidence: {location.confidence:.2f}, "
                        f"classification: {location.classification})"
                    )
                
            except:
                pass
    
    def _get_recent_detections(self) -> List[Dict]:
        """Get recent detections from other nodes."""
        # TODO: Implement detection collection from mesh network
        return []
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        return {
            'running': self.running,
            'sdr_devices': self.sdr_manager.get_device_info(),
            'mesh_nodes': self.mesh_network.get_node_info(),
            'active_targets': len(self.triangulator.get_location_history())
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GhostNet - Passive RF Detection System')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Create GhostNet instance
    ghostnet = GhostNet(args.config)
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("Shutdown signal received")
        ghostnet.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the system
    ghostnet.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ghostnet.stop()

if __name__ == '__main__':
    main() 