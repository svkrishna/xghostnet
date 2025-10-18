from typing import Dict, Tuple
from threading import RLock
import json
import os


class InMemoryState:
    """Simple process-local state for runtime configuration (e.g., velocities)."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._device_velocity_ms: Dict[str, Tuple[float, float, float]] = {}
        self._device_position: Dict[str, Tuple[float, float, float]] = {}  # lat, lon, alt(m)
        self._path = os.environ.get('GHOSTNET_STATE_PATH', '/app/config/receivers_state.json')
        self._load()

    def set_device_velocity(self, device_id: str, vx: float, vy: float, vz: float) -> None:
        with self._lock:
            self._device_velocity_ms[device_id] = (float(vx), float(vy), float(vz))
            self._save()

    def get_device_velocity(self, device_id: str) -> Tuple[float, float, float]:
        with self._lock:
            return self._device_velocity_ms.get(device_id, (0.0, 0.0, 0.0))

    def get_all_velocities(self) -> Dict[str, Tuple[float, float, float]]:
        with self._lock:
            return dict(self._device_velocity_ms)

    def set_device_position(self, device_id: str, lat: float, lon: float, alt_m: float) -> None:
        with self._lock:
            self._device_position[device_id] = (float(lat), float(lon), float(alt_m))
            self._save()

    def get_device_position(self, device_id: str) -> Tuple[float, float, float]:
        with self._lock:
            return self._device_position.get(device_id, (0.0, 0.0, 0.0))

    def get_all_positions(self) -> Dict[str, Tuple[float, float, float]]:
        with self._lock:
            return dict(self._device_position)

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            data = {
                'velocities': self._device_velocity_ms,
                'positions': self._device_position,
            }
            # Convert tuple values to lists for JSON
            serial = {
                'velocities': {k: list(v) for k, v in data['velocities'].items()},
                'positions': {k: list(v) for k, v in data['positions'].items()},
            }
            with open(self._path, 'w') as f:
                json.dump(serial, f)
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if not os.path.exists(self._path):
                return
            with open(self._path, 'r') as f:
                data = json.load(f)
            vels = data.get('velocities', {})
            poss = data.get('positions', {})
            # Back to tuples
            self._device_velocity_ms = {k: tuple(map(float, v)) for k, v in vels.items()}
            self._device_position = {k: tuple(map(float, v)) for k, v in poss.items()}
        except Exception:
            pass


# Global singleton
state = InMemoryState()


