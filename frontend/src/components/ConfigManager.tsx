import React, { useState, useEffect } from 'react';
import { Box, Paper, Typography, TextField, Button, Grid, FormControl, InputLabel, Select, MenuItem, Alert, Snackbar } from '@mui/material';
import { Save } from '@mui/icons-material';

interface DeviceConfig {
  device_id: string;
  sample_rate: number;
  center_freq: number;
  gain: number;
  bandwidth: number;
  device_type: string;
}

const ConfigManager: React.FC = () => {
  const [configs, setConfigs] = useState<DeviceConfig[]>([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [currentConfig, setCurrentConfig] = useState<DeviceConfig>({
    device_id: '',
    sample_rate: 0,
    center_freq: 0,
    gain: 0,
    bandwidth: 0,
    device_type: '',
  });
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' as 'success' | 'error' });

  useEffect(() => {
    fetchConfigs();
  }, []);

  const fetchConfigs = async () => {
    try {
      const response = await fetch('http://localhost:8000/devices');
      const data = await response.json();
      setConfigs(data.devices);
    } catch (error) {
      console.error('Error fetching configurations:', error);
      showNotification('Error fetching configurations', 'error');
    }
  };

  const handleDeviceSelect = (deviceId: string) => {
    setSelectedDevice(deviceId);
    const deviceConfig = configs.find(config => config.device_id === deviceId);
    if (deviceConfig) {
      setCurrentConfig(deviceConfig);
    }
  };

  const handleConfigChange = (field: keyof DeviceConfig, value: number | string) => {
    setCurrentConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const saveConfig = async () => {
    try {
      const response = await fetch(`http://localhost:8000/devices/${selectedDevice}/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentConfig),
      });

      if (response.ok) {
        showNotification('Configuration saved successfully', 'success');
        fetchConfigs();
      } else {
        throw new Error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      showNotification('Error saving configuration', 'error');
    }
  };

  const showNotification = (message: string, severity: 'success' | 'error') => {
    setNotification({ open: true, message, severity });
  };

  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Device Configuration
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Select Device</InputLabel>
            <Select
              value={selectedDevice}
              onChange={(e) => handleDeviceSelect(e.target.value)}
              label="Select Device"
            >
              {configs.map((config) => (
                <MenuItem key={config.device_id} value={config.device_id}>
                  {config.device_id} ({config.device_type})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {selectedDevice && (
          <>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Sample Rate (Hz)"
                type="number"
                value={currentConfig.sample_rate}
                onChange={(e) => handleConfigChange('sample_rate', Number(e.target.value))}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Center Frequency (Hz)"
                type="number"
                value={currentConfig.center_freq}
                onChange={(e) => handleConfigChange('center_freq', Number(e.target.value))}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Gain (dB)"
                type="number"
                value={currentConfig.gain}
                onChange={(e) => handleConfigChange('gain', Number(e.target.value))}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Bandwidth (Hz)"
                type="number"
                value={currentConfig.bandwidth}
                onChange={(e) => handleConfigChange('bandwidth', Number(e.target.value))}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                variant="contained"
                color="primary"
                startIcon={<Save />}
                onClick={saveConfig}
                fullWidth
              >
                Save Configuration
              </Button>
            </Grid>
          </>
        )}
      </Grid>

      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
      >
        <Alert
          onClose={handleCloseNotification}
          severity={notification.severity}
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ConfigManager; 