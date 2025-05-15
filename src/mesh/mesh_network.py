"""
Mesh networking module for GhostNet.
Handles coordination between multiple distributed receivers.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ReceiverNode:
    """Represents a receiver node in the mesh network."""
    id: str
    position: Tuple[float, float]  # (latitude, longitude)
    last_seen: float
    status: str
    capabilities: Dict[str, bool]

class MeshNetwork:
    """Manages the mesh network of distributed receivers."""
    
    def __init__(self):
        self.nodes: Dict[str, ReceiverNode] = {}
        self.last_sync_time = 0
        self.sync_interval = 1.0  # seconds
    
    def add_node(self, node_id: str, position: Tuple[float, float], 
                capabilities: Dict[str, bool]) -> bool:
        """
        Add a new receiver node to the mesh.
        
        Args:
            node_id: Unique identifier for the node
            position: (latitude, longitude) tuple
            capabilities: Dictionary of node capabilities
            
        Returns:
            bool: True if node was added successfully
        """
        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already exists")
            return False
        
        self.nodes[node_id] = ReceiverNode(
            id=node_id,
            position=position,
            last_seen=time.time(),
            status="active",
            capabilities=capabilities
        )
        logger.info(f"Added node {node_id} to mesh network")
        return True
    
    def update_node_status(self, node_id: str, status: str) -> bool:
        """Update the status of a node."""
        if node_id not in self.nodes:
            return False
        
        self.nodes[node_id].status = status
        self.nodes[node_id].last_seen = time.time()
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the mesh."""
        if node_id not in self.nodes:
            return False
        
        del self.nodes[node_id]
        logger.info(f"Removed node {node_id} from mesh network")
        return True
    
    def get_active_nodes(self) -> List[ReceiverNode]:
        """Get list of currently active nodes."""
        current_time = time.time()
        return [
            node for node in self.nodes.values()
            if node.status == "active" and 
            current_time - node.last_seen < self.sync_interval * 2
        ]
    
    def synchronize_nodes(self) -> bool:
        """
        Synchronize all nodes in the mesh.
        This would typically involve time synchronization and data exchange.
        """
        current_time = time.time()
        if current_time - self.last_sync_time < self.sync_interval:
            return False
        
        active_nodes = self.get_active_nodes()
        if not active_nodes:
            logger.warning("No active nodes to synchronize")
            return False
        
        # Update last sync time
        self.last_sync_time = current_time
        
        # In a real implementation, this would:
        # 1. Exchange timing information
        # 2. Share detected signals
        # 3. Coordinate frequency scanning
        # 4. Update node statuses
        
        logger.info(f"Synchronized {len(active_nodes)} nodes")
        return True
    
    def triangulate_position(self, signal_strengths: Dict[str, float]) -> Optional[Tuple[float, float]]:
        """
        Triangulate the position of a signal source using multiple receivers.
        
        Args:
            signal_strengths: Dictionary mapping node IDs to signal strength
            
        Returns:
            Optional[Tuple[float, float]]: Estimated (latitude, longitude) or None
        """
        if len(signal_strengths) < 3:
            logger.warning("Need at least 3 receivers for triangulation")
            return None
        
        # Get positions of nodes that detected the signal
        positions = []
        strengths = []
        for node_id, strength in signal_strengths.items():
            if node_id in self.nodes:
                positions.append(self.nodes[node_id].position)
                strengths.append(strength)
        
        if len(positions) < 3:
            return None
        
        # Simple weighted average for demonstration
        # In a real implementation, this would use proper triangulation algorithms
        weights = np.array(strengths) / sum(strengths)
        lat = sum(p[0] * w for p, w in zip(positions, weights))
        lon = sum(p[1] * w for p, w in zip(positions, weights))
        
        return (lat, lon)
    
    def get_network_status(self) -> Dict:
        """Get the current status of the mesh network."""
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(self.get_active_nodes()),
            "last_sync": self.last_sync_time,
            "nodes": {
                node_id: {
                    "status": node.status,
                    "last_seen": node.last_seen,
                    "capabilities": node.capabilities
                }
                for node_id, node in self.nodes.items()
            }
        } 