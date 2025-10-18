import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, IconButton, Chip } from '@mui/material';
import { Refresh, Settings } from '@mui/icons-material';
import { fetchKnownDevices, KnownDevice } from '../api/geolocation';

const DeviceMonitor: React.FC = () => {
  const [devices, setDevices] = useState<KnownDevice[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchDevices = async () => {
    setLoading(true);
    try {
      const list = await fetchKnownDevices();
      setDevices(list);
    } catch (error) {
      console.error('Error fetching devices:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDevices();
    const interval = setInterval(fetchDevices, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Device Status</Typography>
        <IconButton onClick={fetchDevices} disabled={loading}>
          <Refresh />
        </IconButton>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Device ID</TableCell>
              <TableCell>Latitude</TableCell>
              <TableCell>Longitude</TableCell>
              <TableCell>Signal Strength (dBm)</TableCell>
              <TableCell>Last Update</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {devices.map((device) => (
              <TableRow key={device.device_id}>
                <TableCell>{device.device_id}</TableCell>
                <TableCell>{device.latitude.toFixed(6)}</TableCell>
                <TableCell>{device.longitude.toFixed(6)}</TableCell>
                <TableCell>
                  <Chip 
                    label={`${device.signal_strength} dBm`} 
                    color={device.signal_strength > -60 ? 'success' : device.signal_strength > -80 ? 'warning' : 'error' as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>{new Date(device.timestamp * 1000).toLocaleString()}</TableCell>
                <TableCell>
                  <IconButton size="small">
                    <Settings />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default DeviceMonitor; 