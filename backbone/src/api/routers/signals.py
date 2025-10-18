from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
import numpy as np
from ..core.security import verify_token
from ..core.signal_processor import SignalProcessor
from ...hardware.sdr_interface import SDRManager, SDRConfig
import json
import os
from ..core.state import state
from pydantic import BaseModel
from typing import List, Optional


router = APIRouter(prefix="/signals", tags=["signals"])
processor = SignalProcessor(sample_rate=2_000_000, fft_size=1024)
_sdr_manager: SDRManager | None = None


def _load_config() -> dict:
    # Default config path inside container compose mount
    cfg_path = os.environ.get("GHOSTNET_CONFIG", "/app/config/ghostnet_config.json")
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _ensure_device(device_id: str | None) -> tuple[SDRManager, str]:
    global _sdr_manager
    if _sdr_manager is None:
        _sdr_manager = SDRManager()
        cfg = _load_config()
        devices = (cfg.get("sdr_devices") or {})
        # Default to primary if not provided
        chosen = device_id or "primary"
        dev_cfg = devices.get(chosen) or {}
        dev_type = (dev_cfg.get("type") or "rtlsdr")
        s = SDRConfig(
            center_freq=float(dev_cfg.get("center_freq", 2.4e9)),
            sample_rate=float(dev_cfg.get("sample_rate", 2e6)),
            gain=float(dev_cfg.get("gain", 20.0)),
            bandwidth=float(dev_cfg.get("bandwidth", 2e6)),
            device_args=dev_cfg.get("device_args", {}),
        )
        # Add device; fallback to synthetic if add fails
        added = _sdr_manager.add_device(chosen, dev_type, s)
        if added:
            _sdr_manager.start_all_devices()
        else:
            # Leave manager empty; callers will synthesize
            pass
        return _sdr_manager, chosen
    # If manager exists but device_id differs and not known, try add it
    chosen = device_id or "primary"
    if chosen not in _sdr_manager.devices:
        cfg = _load_config()
        devices = (cfg.get("sdr_devices") or {})
        dev_cfg = devices.get(chosen) or {}
        dev_type = (dev_cfg.get("type") or "rtlsdr")
        s = SDRConfig(
            center_freq=float(dev_cfg.get("center_freq", 2.4e9)),
            sample_rate=float(dev_cfg.get("sample_rate", 2e6)),
            gain=float(dev_cfg.get("gain", 20.0)),
            bandwidth=float(dev_cfg.get("bandwidth", 2e6)),
            device_args=dev_cfg.get("device_args", {}),
        )
        if _sdr_manager.add_device(chosen, dev_type, s):
            _sdr_manager.start_all_devices()
    return _sdr_manager, chosen


@router.get("/devices")
async def list_devices(token: str = Depends(verify_token)):
    """List known SDR devices from config and which are active in manager."""
    cfg = _load_config()
    devices_cfg = cfg.get("sdr_devices", {})
    manager = _sdr_manager
    devices = []
    for dev_id, dev_cfg in devices_cfg.items():
        is_active = bool(manager and manager.devices.get(dev_id))
        vx, vy, vz = state.get_device_velocity(dev_id)
        lat, lon, alt_m = state.get_device_position(dev_id)
        devices.append({
            "device_id": dev_id,
            "type": dev_cfg.get("type"),
            "center_freq": dev_cfg.get("center_freq"),
            "sample_rate": dev_cfg.get("sample_rate"),
            "gain": dev_cfg.get("gain"),
            "bandwidth": dev_cfg.get("bandwidth"),
            "active": is_active,
            "receiver_velocity": {"vx": vx, "vy": vy, "vz": vz},
            "receiver_position": {"lat": lat, "lon": lon, "alt_m": alt_m},
        })
    return {"status": "success", "devices": devices}


@router.post("/devices/{device_id}/velocity")
async def set_device_velocity(
    device_id: str,
    vx: float,
    vy: float,
    vz: float,
    token: str = Depends(verify_token),
):
    state.set_device_velocity(device_id, vx, vy, vz)
    return {"status": "success", "device_id": device_id, "receiver_velocity": {"vx": vx, "vy": vy, "vz": vz}}


@router.post("/devices/{device_id}/position")
async def set_device_position(
    device_id: str,
    lat: float,
    lon: float,
    alt_m: float | None = None,
    alt: float | None = None,
    alt_units: str = "m",
    token: str = Depends(verify_token),
):
    # Validate lat/lon
    if lat < -90 or lat > 90:
        raise HTTPException(status_code=400, detail="lat must be within [-90, 90]")
    if lon < -180 or lon > 180:
        raise HTTPException(status_code=400, detail="lon must be within [-180, 180]")
    # Altitude conversion
    meters: float
    if alt_m is not None:
        meters = float(alt_m)
    elif alt is not None:
        if alt_units.lower() == "m":
            meters = float(alt)
        elif alt_units.lower() in ("ft", "feet"):
            meters = float(alt) * 0.3048
        else:
            raise HTTPException(status_code=400, detail="alt_units must be 'm' or 'ft'")
    else:
        meters = 0.0
    # Reasonable altitude range check (-500m to 60000m)
    if meters < -500 or meters > 60000:
        raise HTTPException(status_code=400, detail="altitude out of range [-500, 60000] meters")
    state.set_device_position(device_id, lat, lon, meters)
    return {"status": "success", "device_id": device_id, "receiver_position": {"lat": lat, "lon": lon, "alt_m": meters}}


class DeviceUpdate(BaseModel):
    device_id: str
    receiver_velocity: Optional[dict] = None  # {vx, vy, vz}
    receiver_position: Optional[dict] = None  # {lat, lon, alt_m} or {lat, lon, alt, alt_units}


@router.post("/devices/bulk")
async def bulk_update_devices(
    updates: List[DeviceUpdate],
    token: str = Depends(verify_token),
):
    for upd in updates:
        if upd.receiver_velocity is not None:
            vx = float(upd.receiver_velocity.get('vx', 0))
            vy = float(upd.receiver_velocity.get('vy', 0))
            vz = float(upd.receiver_velocity.get('vz', 0))
            state.set_device_velocity(upd.device_id, vx, vy, vz)
        if upd.receiver_position is not None:
            lat = upd.receiver_position.get('lat', None)
            lon = upd.receiver_position.get('lon', None)
            if lat is None or lon is None:
                raise HTTPException(status_code=400, detail=f"receiver_position requires lat and lon for {upd.device_id}")
            lat = float(lat); lon = float(lon)
            if lat < -90 or lat > 90:
                raise HTTPException(status_code=400, detail=f"lat must be within [-90, 90] for {upd.device_id}")
            if lon < -180 or lon > 180:
                raise HTTPException(status_code=400, detail=f"lon must be within [-180, 180] for {upd.device_id}")
            # Altitude
            if 'alt' in upd.receiver_position:
                alt = float(upd.receiver_position.get('alt', 0))
                units = str(upd.receiver_position.get('alt_units', 'm')).lower()
                if units == 'm':
                    meters = alt
                elif units in ('ft', 'feet'):
                    meters = alt * 0.3048
                else:
                    raise HTTPException(status_code=400, detail=f"alt_units must be 'm' or 'ft' for {upd.device_id}")
            elif 'alt_m' in upd.receiver_position:
                meters = float(upd.receiver_position.get('alt_m', 0))
            else:
                meters = 0.0
            if meters < -500 or meters > 60000:
                raise HTTPException(status_code=400, detail=f"altitude out of range [-500, 60000] meters for {upd.device_id}")
            state.set_device_position(upd.device_id, lat, lon, meters)
    return {"status": "success", "updated": len(updates)}


@router.post("/devices/bulk/validate")
async def bulk_validate_devices(
    updates: List[DeviceUpdate],
    token: str = Depends(verify_token),
):
    report: List[dict] = []
    for upd in updates:
        entry = {"device_id": upd.device_id, "ok": True, "errors": []}
        try:
            if upd.receiver_velocity is not None:
                for k in ('vx','vy','vz'):
                    _ = float(upd.receiver_velocity.get(k, 0))
            if upd.receiver_position is not None:
                lat = upd.receiver_position.get('lat', None)
                lon = upd.receiver_position.get('lon', None)
                if lat is None or lon is None:
                    entry['errors'].append('receiver_position requires lat and lon')
                else:
                    lat = float(lat); lon = float(lon)
                    if lat < -90 or lat > 90:
                        entry['errors'].append('lat out of range [-90, 90]')
                    if lon < -180 or lon > 180:
                        entry['errors'].append('lon out of range [-180, 180]')
                if 'alt' in upd.receiver_position:
                    alt = float(upd.receiver_position.get('alt', 0))
                    units = str(upd.receiver_position.get('alt_units', 'm')).lower()
                    if units not in ('m','ft','feet'):
                        entry['errors'].append("alt_units must be 'm' or 'ft'")
                    meters = alt if units=='m' else alt * 0.3048
                elif 'alt_m' in upd.receiver_position:
                    meters = float(upd.receiver_position.get('alt_m', 0))
                else:
                    meters = 0.0
                if meters < -500 or meters > 60000:
                    entry['errors'].append('altitude out of range [-500, 60000] meters')
        except Exception as e:
            entry['errors'].append(str(e))
        entry['ok'] = len(entry['errors']) == 0
        report.append(entry)
    summary = {
        "total": len(report),
        "ok": sum(1 for r in report if r['ok']),
        "errors": sum(1 for r in report if not r['ok']),
    }
    return {"status": "success", "summary": summary, "report": report}


@router.get("/devices/state")
async def get_devices_state(token: str = Depends(verify_token)):
    """Return only persisted positions and velocities for backups."""
    positions = state.get_all_positions()
    velocities = state.get_all_velocities()
    # Convert tuples to lists for JSON friendliness
    return {
        "status": "success",
        "positions": {k: list(v) for k, v in positions.items()},
        "velocities": {k: list(v) for k, v in velocities.items()},
    }


@router.get("/spectrum")
async def get_spectrum(
    device_id: str | None = None,
    center_freq: float = 2.4e9,
    span_hz: float = 2e6,
    fft_size: int = 1024,
    use_gpu: bool = False,
    token: str = Depends(verify_token),
):
    """Return spectrum from real SDR if available; otherwise synthetic."""
    try:
        manager, choice = _ensure_device(device_id)
        device = manager.devices.get(choice) if manager else None
        if device:
            # Reconfigure center frequency and bandwidth on demand when changed
            try:
                cfg = manager.configs.get(choice)
                if cfg and (abs(cfg.center_freq - center_freq) > 1 or abs(cfg.bandwidth - span_hz) > 1):
                    new_cfg = SDRConfig(
                        center_freq=float(center_freq),
                        sample_rate=cfg.sample_rate,
                        gain=cfg.gain,
                        bandwidth=float(span_hz),
                        device_args=cfg.device_args,
                    )
                    device.stop_streaming()
                    if device.configure(new_cfg):
                        manager.configs[choice] = new_cfg
                        device.start_streaming()
            except Exception:
                # Ignore reconfig failures; proceed with current config
                pass
            # Read samples and compute spectrum
            num_samples = max(fft_size * 8, 16384)
            samples = device.read_samples(num_samples)  # complex IQ
            if samples is None or len(samples) == 0:
                raise RuntimeError("No samples read from SDR")
            # If center_freq/span differs from current, ignore for now; config sets it
            # Optionally enable GPU path
            if use_gpu:
                processor.use_gpu = True
            freqs, _, Sxx = processor.compute_spectrogram(np.asarray(samples))
            psd = 10 * np.log10(np.mean(Sxx, axis=1) + 1e-12)
            return {
                "status": "success",
                "frequencies": freqs.tolist(),
                "magnitudes": psd.tolist(),
                "timestamp": float(len(samples) / processor.sample_rate),
                "device_id": choice,
                "source": "sdr",
                "gpu": bool(processor.use_gpu),
            }
    except Exception:
        pass
    # Fallback synthetic
    num_samples = max(fft_size * 4, 4096)
    t = np.arange(num_samples) / processor.sample_rate
    sig = (
        0.7 * np.sin(2 * np.pi * 10_000 * t)
        + 0.5 * np.sin(2 * np.pi * 50_000 * t)
        + 0.1 * np.random.randn(num_samples)
    )
    if use_gpu:
        processor.use_gpu = True
    freqs, _, Sxx = processor.compute_spectrogram(sig)
    psd = 10 * np.log10(np.mean(Sxx, axis=1) + 1e-12)
    return {
        "status": "success",
        "frequencies": freqs.tolist(),
        "magnitudes": psd.tolist(),
        "timestamp": float(np.max(t)),
        "device_id": device_id or "synthetic",
        "source": "synthetic",
        "gpu": bool(processor.use_gpu),
    }


