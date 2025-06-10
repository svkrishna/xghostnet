import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import { Icon } from 'leaflet';
import axios from 'axios';
import { Box, Typography, Button, CircularProgress, Alert, Paper } from '@mui/material';
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

  const fetchDevices = useCallback(async () => {
    try {
      const response = await axios.get<{ devices: Device[] }>('/api/geolocation/devices');
      setDevices(response.data.devices);
    } catch (err) {
      setError('Failed to fetch devices');
    }
  }, []);

  const fetchFingerprints = useCallback(async () => {
    try {
      const response = await axios.get<{ fingerprints: Fingerprint[] }>('/api/geolocation/fingerprints');
      setFingerprints(response.data.fingerprints);
    } catch (err) {
      setError('Failed to fetch fingerprints');
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
    const interval = setInterval(fetchDevices, 5000);
    return () => clearInterval(interval);
  }, [fetchDevices, fetchFingerprints]);

  const handleCalibrationToggle = async () => {
    try {
      setLoading(true);
      if (isCalibrating) {
        await axios.post('/api/geolocation/calibration/stop');
        await fetchFingerprints();
      } else {
        await axios.post('/api/geolocation/calibration/start');
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

      await axios.post('/api/geolocation/fingerprints', {
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
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
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
      </Box>
    </Box>
  );
};

export default GeolocationMap; 