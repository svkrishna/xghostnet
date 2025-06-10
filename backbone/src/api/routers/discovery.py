from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from ..core.security import verify_token

router = APIRouter(prefix="/discovery", tags=["discovery"])

@router.post("/register")
async def register_device_discovery(
    name: str,
    port: int,
    properties: Dict[str, str],
    token: str = Depends(verify_token)
):
    """Register a device for discovery."""
    try:
        from ..core.signal_classifier import SignalClassifier
        classifier = SignalClassifier()
        classifier.network_manager.register_device(name, port, properties)
        return {"status": "success", "message": "Device registered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect")
async def connect_to_device_discovery(
    device_name: str,
    token: str = Depends(verify_token)
):
    """Connect to a discovered device."""
    try:
        from ..core.signal_classifier import SignalClassifier
        classifier = SignalClassifier()
        device = classifier.network_manager.get_discovered_devices().get(device_name)
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")
        return {"status": "success", "device": device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 