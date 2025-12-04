from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
import time

from ...core.signal_geolocator import SignalGeolocator, SignalFingerprint
from ..core.security import SecurityManager

router = APIRouter(prefix="/geolocation", tags=["geolocation"])
security_manager = SecurityManager("your-secret-key-here")
geolocator = SignalGeolocator()

@router.post("/register")
async def register_device_location(
    device_id: str,
    latitude: float,
    longitude: float,
    signal_strength: float,
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Register a device's known location."""
    try:
        geolocator.add_device_location(
            device_id=device_id,
            latitude=latitude,
            longitude=longitude,
            signal_strength=signal_strength,
            timestamp=time.time()
        )
        return {"status": "success", "message": "Device location registered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/estimate")
async def estimate_position(
    signal_strengths: Dict[str, float],
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Estimate position based on signal strengths."""
    try:
        position = geolocator.estimate_position_fingerprinting(signal_strengths)
        if position:
            lat, lon, uncertainty = position
            return {
                "status": "success",
                "position": {
                    "latitude": lat,
                    "longitude": lon,
                    "uncertainty": uncertainty
                }
            }
        return {
            "status": "error",
            "message": "Insufficient data for position estimation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices")
async def get_known_devices(
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Get list of all known device locations."""
    try:
        devices = geolocator.get_known_devices()
        return {
            "status": "success",
            "devices": [
                {
                    "device_id": d.device_id,
                    "latitude": d.latitude,
                    "longitude": d.longitude,
                    "signal_strength": d.signal_strength,
                    "timestamp": d.timestamp
                }
                for d in devices
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fingerprints")
async def list_fingerprints(
    token: str = Depends(security_manager.oauth2_scheme)
):
    """List all stored fingerprints."""
    try:
        fps = geolocator.get_fingerprints()
        return {
            "status": "success",
            "fingerprints": [
                {
                    "location_id": fp.location_id,
                    "latitude": fp.latitude,
                    "longitude": fp.longitude,
                    "signal_strengths": fp.signal_strengths,
                    "timestamp": fp.timestamp,
                    "metadata": fp.metadata,
                }
                for fp in fps
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/devices/{device_id}")
async def remove_device_location(
    device_id: str,
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Remove a device's known location."""
    try:
        geolocator.remove_device_location(device_id)
        return {"status": "success", "message": "Device location removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calibration/start")
async def start_calibration(
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Start calibration mode for collecting fingerprints."""
    try:
        geolocator.start_calibration()
        return {"status": "success", "message": "Calibration mode started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/calibration/stop")
async def stop_calibration(
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Stop calibration mode and process collected data."""
    try:
        geolocator.stop_calibration()
        return {"status": "success", "message": "Calibration mode stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fingerprints")
async def add_fingerprint(
    location_id: str,
    latitude: float,
    longitude: float,
    signal_strengths: Dict[str, float],
    metadata: Optional[Dict] = None,
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Add a new signal fingerprint."""
    try:
        fingerprint = SignalFingerprint(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            signal_strengths=signal_strengths,
            timestamp=time.time(),
            metadata=metadata
        )
        geolocator.add_fingerprint(fingerprint)
        return {"status": "success", "message": "Fingerprint added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fingerprints/statistics")
async def get_fingerprint_statistics(
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Get statistics about the fingerprint database."""
    try:
        stats = geolocator.get_fingerprint_statistics()
        return {"status": "success", "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fingerprints/clusters")
async def get_fingerprint_clusters(
    eps: float = 0.5,
    min_samples: int = 5,
    token: str = Depends(security_manager.oauth2_scheme)
):
    """Get clusters of similar fingerprints."""
    try:
        clusters = geolocator.cluster_fingerprints(eps=eps, min_samples=min_samples)
        return {
            "status": "success",
            "clusters": [
                [
                    {
                        "location_id": fp.location_id,
                        "latitude": fp.latitude,
                        "longitude": fp.longitude,
                        "signal_strengths": fp.signal_strengths,
                        "timestamp": fp.timestamp,
                        "metadata": fp.metadata
                    }
                    for fp in cluster
                ]
                for cluster in clusters
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 