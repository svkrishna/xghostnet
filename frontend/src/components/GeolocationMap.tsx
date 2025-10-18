import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import { Icon } from 'leaflet';
import { colorForDeviceId } from '../api/colors';
import { apiClient } from '../api/client';
import { Box, Typography, Button, CircularProgress, Alert, Paper, Switch, FormControlLabel } from '@mui/material';
import { PlayArrow, Stop, AddLocation, Timeline } from '@mui/icons-material';
import 'leaflet/dist/leaflet.css';

interface Device {
  device_id: string;
  latitude: number;
  longitude: number;
  signal_strength: number;
  timestamp: number;
}

interface Fingerprint {
  location_id: string;
  latitude: number;
  longitude: number;
  signal_strengths: { [key: string]: number };
  timestamp: number;
  metadata?: any;
}

interface Cluster {
  fingerprints: Fingerprint[];
}

const GeolocationMap: React.FC = () => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [fingerprints, setFingerprints] = useState<Fingerprint[]>([]);
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState<string | null>(null);
  const [showClusters, setShowClusters] = useState(false);
  const [stats, setStats] = useState<any | null>(null);
  const [receivers, setReceivers] = useState<any[]>([]);

  const fetchDevices = useCallback(async () => {
    try {
      const response = await apiClient.get<{ devices: Device[] }>('/geolocation/devices');
      setDevices((response.data as any).devices || []);
    } catch (err) {
      setError('Failed to fetch devices');
    }
  }, []);

  const fetchFingerprints = useCallback(async () => {
    try {
      const response = await apiClient.get('/geolocation/fingerprints');
      const clustersRes = await apiClient.get('/geolocation/fingerprints/clusters');
      const statsRes = await apiClient.get('/geolocation/fingerprints/statistics');
      setFingerprints((response.data as any).fingerprints || []);
      setClusters((clustersRes.data as any).clusters || []);
      setStats((statsRes.data as any).statistics || null);
    } catch (err) {
      setError('Failed to fetch fingerprints');
    }
  }, []);

  const fetchReceivers = useCallback(async () => {
    try {
      const res = await apiClient.get('/signals/devices');
      setReceivers(res.data?.devices || []);
    } catch (err) {
      setReceivers([]);
    }
  }, []);

  const fetchClusters = useCallback(async () => {
    try {
      const response = await axios.get<{ clusters: Cluster[] }>('/api/geolocation/fingerprints/clusters');
      setClusters(response.data.clusters);
    } catch (err) {
      setError('Failed to fetch clusters');
    }
  }, []);

  useEffect(() => {
    fetchDevices();
    fetchFingerprints();
    fetchReceivers();
    const interval = setInterval(fetchDevices, 5000);
    return () => clearInterval(interval);
  }, [fetchDevices, fetchFingerprints, fetchReceivers]);

  const handleCalibrationToggle = async () => {
    try {
      setLoading(true);
      if (isCalibrating) {
        await apiClient.post('/geolocation/calibration/stop');
        await fetchFingerprints();
      } else {
        await apiClient.post('/geolocation/calibration/start');
      }
      setIsCalibrating(!isCalibrating);
    } catch (err) {
      setError('Failed to toggle calibration mode');
    } finally {
      setLoading(false);
    }
  };

  const handleAddFingerprint = async (lat: number, lng: number) => {
    if (!isCalibrating) return;

    try {
      const signalStrengths = devices.reduce((acc, device) => ({
        ...acc,
        [device.device_id]: device.signal_strength
      }), {});

      await apiClient.post('/geolocation/fingerprints', {
        location_id: `fp_${Date.now()}`,
        latitude: lat,
        longitude: lng,
        signal_strengths: signalStrengths,
        metadata: {
          timestamp: Date.now(),
          device_count: devices.length
        }
      });

      await fetchFingerprints();
    } catch (err) {
      setError('Failed to add fingerprint');
    }
  };

  const MapEvents: React.FC = () => {
    const map = useMap();

    useEffect(() => {
      if (isCalibrating) {
        map.on('click', (e) => {
          handleAddFingerprint(e.latlng.lat, e.latlng.lng);
        });
      }
      return () => {
        map.off('click');
      };
    }, [isCalibrating, map]);

    return null;
  };

  const getDeviceIcon = (signalStrength: number) => {
    const color = signalStrength > -50 ? 'green' : signalStrength > -70 ? 'yellow' : 'red';
    return new Icon({
      iconUrl: `https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-${color}.png`,
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowSize: [41, 41]
    });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            color={isCalibrating ? 'error' : 'primary'}
            onClick={handleCalibrationToggle}
            disabled={loading}
            startIcon={isCalibrating ? <Stop /> : <PlayArrow />}
          >
            {isCalibrating ? 'Stop Calibration' : 'Start Calibration'}
          </Button>
          <Button
            variant="outlined"
            onClick={() => setShowClusters(!showClusters)}
            startIcon={<Timeline />}
          >
            {showClusters ? 'Hide Clusters' : 'Show Clusters'}
          </Button>
          <Button
            variant="outlined"
            onClick={() => {
              const pts = receivers
                .filter(r => r.receiver_position?.lat || r.receiver_position?.lon)
                .map(r => [r.receiver_position.lat || 0, r.receiver_position.lon || 0]);
              if (pts.length === 0) return;
              // Fit bounds using leaflet instance
              const lats = pts.map(p => p[0]);
              const lons = pts.map(p => p[1]);
              const minLat = Math.min(...lats); const maxLat = Math.max(...lats);
              const minLon = Math.min(...lons); const maxLon = Math.max(...lons);
              const map = (document.querySelector('.leaflet-container') as any)?._leaflet_map;
              if (map && map.fitBounds) {
                map.fitBounds([[minLat, minLon], [maxLat, maxLon]]);
              }
            }}
          >
            Center Receivers
          </Button>
          <FormControlLabel
            control={<Switch checked={showClusters} onChange={() => setShowClusters(!showClusters)} />}
            label="Clusters"
          />
          {stats && (
            <Paper sx={{ p: 1, bgcolor: 'background.default' }}>
              <Typography variant="caption" color="text.secondary">
                {`Fingerprints: ${stats.count}  Devices: ${stats.devices}`}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                {`Lat[${stats.coverage.min_lat.toFixed(2)}, ${stats.coverage.max_lat.toFixed(2)}] Lon[${stats.coverage.min_lon.toFixed(2)}, ${stats.coverage.max_lon.toFixed(2)}]`}
              </Typography>
            </Paper>
          )}
          {showClusters && (
            <Paper sx={{ p: 1, bgcolor: 'background.default' }}>
              <Typography variant="caption" color="text.secondary">Cluster Colors:</Typography>
              {clusters.map((_, idx) => (
                <span key={idx} style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: `hsl(${idx * 30}, 70%, 50%)`, marginLeft: 8, borderRadius: 2 }} />
              ))}
            </Paper>
          )}
          {isCalibrating && (
            <Typography variant="body2" color="text.secondary">
              Click on the map to add fingerprints
            </Typography>
          )}
        </Box>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Box sx={{ flex: 1, position: 'relative' }}>
        <MapContainer
          center={[0, 0]}
          zoom={2}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          <MapEvents />

          {devices.map((device) => (
            <Marker
              key={device.device_id}
              position={[device.latitude, device.longitude]}
              icon={getDeviceIcon(device.signal_strength)}
              eventHandlers={{
                click: () => setSelectedDevice(device.device_id)
              }}
            >
              <Popup>
                <Typography variant="subtitle2">Device: {device.device_id}</Typography>
                <Typography variant="body2">
                  Signal Strength: {device.signal_strength} dBm
                </Typography>
                <Typography variant="body2">
                  Last Update: {new Date(device.timestamp * 1000).toLocaleString()}
                </Typography>
              </Popup>
            </Marker>
          ))}

          {receivers.map((r) => (
            (r.receiver_position?.lat || r.receiver_position?.lon) ? (
              <Marker key={`rx_${r.device_id}`}
                position={[r.receiver_position?.lat || 0, r.receiver_position?.lon || 0]}
              >
                <Popup>
                  <Typography variant="subtitle2">
                    <span style={{ width: 12, height: 12, display: 'inline-block', backgroundColor: colorForDeviceId(r.device_id), borderRadius: 2, marginRight: 6 }} />
                    Receiver: {r.device_id}
                  </Typography>
                  <Typography variant="body2">Active: {String(r.active)}</Typography>
                  <Typography variant="body2">Lat: {r.receiver_position?.lat ?? 0}, Lon: {r.receiver_position?.lon ?? 0}</Typography>
                  <Typography variant="body2">Alt: {r.receiver_position?.alt_m ?? 0} m</Typography>
                </Popup>
              </Marker>
            ) : null
          ))}

          {fingerprints.map((fp) => (
            <Circle
              key={fp.location_id}
              center={[fp.latitude, fp.longitude]}
              radius={10}
              pathOptions={{
                color: 'blue',
                fillColor: 'blue',
                fillOpacity: 0.3
              }}
            >
              <Popup>
                <Typography variant="subtitle2">Fingerprint: {fp.location_id}</Typography>
                <Typography variant="body2">
                  Signal Strengths:
                  {Object.entries(fp.signal_strengths).map(([device, strength]) => (
                    <div key={device}>
                      {device}: {strength} dBm
                    </div>
                  ))}
                </Typography>
              </Popup>
            </Circle>
          ))}

          {showClusters && clusters.map((cluster, clusterIndex) => (
            <Circle
              key={`cluster_${clusterIndex}`}
              center={[
                cluster.fingerprints.reduce((sum, fp) => sum + fp.latitude, 0) / cluster.fingerprints.length,
                cluster.fingerprints.reduce((sum, fp) => sum + fp.longitude, 0) / cluster.fingerprints.length
              ]}
              radius={50}
              pathOptions={{
                color: `hsl(${clusterIndex * 30}, 70%, 50%)`,
                fillColor: `hsl(${clusterIndex * 30}, 70%, 50%)`,
                fillOpacity: 0.2
              }}
            >
              <Popup>
                <Typography variant="subtitle2">Cluster {clusterIndex + 1}</Typography>
                <Typography variant="body2">
                  Fingerprints: {cluster.fingerprints.length}
                </Typography>
              </Popup>
            </Circle>
          ))}
        </MapContainer>
        {receivers.length > 0 && (
          <Paper sx={{ position: 'absolute', bottom: 12, left: 12, p: 1.5, bgcolor: 'background.paper', opacity: 0.9 }}>
            <Typography variant="caption" color="text.secondary">Receivers</Typography>
            <Box sx={{ mt: 1, display: 'flex', gap: 1, flexWrap: 'wrap', maxWidth: 360 }}>
              {receivers.map((r) => (
                <Box key={`legend_${r.device_id}`} sx={{ display: 'flex', alignItems: 'center', mr: 1 }}>
                  <span style={{ width: 10, height: 10, display: 'inline-block', backgroundColor: colorForDeviceId(r.device_id), borderRadius: 2, marginRight: 6 }} />
                  <Typography variant="caption">{r.device_id}</Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        )}
      </Box>
    </Box>
  );
};

export default GeolocationMap; 