"""
Mesh networking module for GhostNet.
Coordinates multiple SDR receivers for triangulation and distributed processing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import threading
from dataclasses import dataclass
from queue import Queue
import json
import socket
import struct
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReceiverNode:
    """Represents a receiver node in the mesh network."""
    node_id: str
    position: Tuple[float, float, float]  # (x, y, z) in meters
    ip_address: str
    port: int
    last_seen: float
    is_active: bool

@dataclass
class DetectionEvent:
    """Represents a detection event from a receiver."""
    timestamp: float
    node_id: str
    signal_features: Dict[str, float]
    confidence: float
    classification: str
    position: Optional[Tuple[float, float, float]] = None

class MeshNetwork:
    """Manages the mesh network of SDR receivers."""
    
    def __init__(self, local_node_id: str, local_position: Tuple[float, float, float],
                 local_port: int = 5000):
        """
        Initialize the mesh network.
        
        Args:
            local_node_id: Unique identifier for this node
            local_position: (x, y, z) position in meters
            local_port: Port for mesh communication
        """
        self.local_node_id = local_node_id
        self.local_position = local_position
        self.local_port = local_port
        
        self.nodes: Dict[str, ReceiverNode] = {}
        self.detection_queue = Queue()
        self.is_running = False
        
        # Network communication
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', local_port))
        
        # Threads
        self._network_thread = None
        self._processing_thread = None
    
    def start(self) -> bool:
        """Start the mesh network."""
        if self.is_running:
            return False
        
        self.is_running = True
        
        # Start network thread
        self._network_thread = threading.Thread(target=self._network_loop)
        self._network_thread.daemon = True
        self._network_thread.start()
        
        # Start processing thread
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        logger.info(f"Started mesh network on port {self.local_port}")
        return True
    
    def stop(self) -> bool:
        """Stop the mesh network."""
        self.is_running = False
        if self._network_thread:
            self._network_thread.join()
        if self._processing_thread:
            self._processing_thread.join()
        self.socket.close()
        return True
    
    def add_node(self, node_id: str, position: Tuple[float, float, float],
                ip_address: str, port: int) -> bool:
        """Add a new node to the mesh network."""
        if node_id in self.nodes:
            return False
        
        self.nodes[node_id] = ReceiverNode(
            node_id=node_id,
            position=position,
            ip_address=ip_address,
            port=port,
            last_seen=time.time(),
            is_active=True
        )
        
        # Send hello message
        self._send_hello(node_id, ip_address, port)
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the mesh network."""
        if node_id not in self.nodes:
            return False
        
        del self.nodes[node_id]
        return True
    
    def broadcast_detection(self, detection: DetectionEvent) -> bool:
        """
        Broadcast a detection event to all nodes.
        
        Args:
            detection: Detection event to broadcast
            
        Returns:
            True if broadcast was successful
        """
        try:
            # Add local position to detection
            detection.position = self.local_position
            
            # Convert detection to JSON
            message = {
                'type': 'detection',
                'data': {
                    'timestamp': detection.timestamp,
                    'node_id': detection.node_id,
                    'signal_features': detection.signal_features,
                    'confidence': detection.confidence,
                    'classification': detection.classification,
                    'position': detection.position
                }
            }
            
            # Broadcast to all nodes
            for node in self.nodes.values():
                if node.is_active:
                    self._send_message(message, node.ip_address, node.port)
            
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast detection: {str(e)}")
            return False
    
    def _network_loop(self):
        """Main network loop for receiving messages."""
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = json.loads(data.decode())
                
                if message['type'] == 'hello':
                    self._handle_hello(message['data'], addr[0], addr[1])
                elif message['type'] == 'detection':
                    self._handle_detection(message['data'])
                elif message['type'] == 'heartbeat':
                    self._handle_heartbeat(message['data'])
                
            except Exception as e:
                logger.error(f"Error in network loop: {str(e)}")
                time.sleep(0.1)
    
    def _processing_loop(self):
        """Main processing loop for handling detections."""
        while self.is_running:
            try:
                # Process detections from queue
                detection = self.detection_queue.get(timeout=1.0)
                self._process_detection(detection)
            except:
                pass
    
    def _handle_hello(self, data: Dict, ip_address: str, port: int):
        """Handle hello message from a new node."""
        node_id = data['node_id']
        position = tuple(data['position'])
        
        if node_id not in self.nodes:
            self.add_node(node_id, position, ip_address, port)
        else:
            # Update existing node
            self.nodes[node_id].last_seen = time.time()
            self.nodes[node_id].is_active = True
    
    def _handle_detection(self, data: Dict):
        """Handle detection message from another node."""
        detection = DetectionEvent(
            timestamp=data['timestamp'],
            node_id=data['node_id'],
            signal_features=data['signal_features'],
            confidence=data['confidence'],
            classification=data['classification'],
            position=tuple(data['position'])
        )
        self.detection_queue.put(detection)
    
    def _handle_heartbeat(self, data: Dict):
        """Handle heartbeat message from a node."""
        node_id = data['node_id']
        if node_id in self.nodes:
            self.nodes[node_id].last_seen = time.time()
            self.nodes[node_id].is_active = True
    
    def _send_hello(self, node_id: str, ip_address: str, port: int):
        """Send hello message to a node."""
        message = {
            'type': 'hello',
            'data': {
                'node_id': self.local_node_id,
                'position': self.local_position
            }
        }
        self._send_message(message, ip_address, port)
    
    def _send_message(self, message: Dict, ip_address: str, port: int):
        """Send a message to a specific node."""
        try:
            data = json.dumps(message).encode()
            self.socket.sendto(data, (ip_address, port))
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
    
    def _process_detection(self, detection: DetectionEvent):
        """
        Process a detection event.
        This method should be overridden by subclasses to implement
        specific detection processing logic.
        """
        # Basic implementation: just log the detection
        logger.info(f"Received detection from {detection.node_id}: "
                   f"{detection.classification} "
                   f"(confidence: {detection.confidence:.2f})")
    
    def get_active_nodes(self) -> List[ReceiverNode]:
        """Get list of active nodes."""
        return [node for node in self.nodes.values() if node.is_active]
    
    def get_node_info(self) -> Dict[str, Dict]:
        """Get information about all nodes."""
        return {
            node_id: {
                'position': node.position,
                'ip_address': node.ip_address,
                'port': node.port,
                'last_seen': node.last_seen,
                'is_active': node.is_active
            }
            for node_id, node in self.nodes.items()
        } 