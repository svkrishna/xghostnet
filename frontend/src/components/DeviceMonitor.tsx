import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, IconButton, Chip } from '@mui/material';
import { Refresh, Settings } from '@mui/icons-material';

interface Device {
  id: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  sample_rate: number;
  center_freq: number;
  gain: number;
  last_update: string;
}

const DeviceMonitor: React.FC = () => {
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchDevices = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/devices');
      const data = await response.json();
      setDevices(data.devices);
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
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Sample Rate</TableCell>
              <TableCell>Center Frequency</TableCell>
              <TableCell>Gain</TableCell>
              <TableCell>Last Update</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {devices.map((device) => (
              <TableRow key={device.id}>
                <TableCell>{device.id}</TableCell>
                <TableCell>{device.type}</TableCell>
                <TableCell>
                  <Chip 
                    label={device.status} 
                    color={getStatusColor(device.status) as any}
                    size="small"
                  />
                </TableCell>
                <TableCell>{device.sample_rate.toLocaleString()} Hz</TableCell>
                <TableCell>{device.center_freq.toLocaleString()} Hz</TableCell>
                <TableCell>{device.gain} dB</TableCell>
                <TableCell>{new Date(device.last_update).toLocaleString()}</TableCell>
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