import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, FormControl, InputLabel, Select, MenuItem, Slider, Grid } from '@mui/material';
import Plot from 'react-plotly.js';
import io from 'socket.io-client';

interface SpectrumData {
  frequencies: number[];
  magnitudes: number[];
  timestamp: number;
}

const SpectrumVisualizer: React.FC = () => {
  const [selectedDevice, setSelectedDevice] = useState('');
  const [devices, setDevices] = useState<string[]>([]);
  const [spectrumData, setSpectrumData] = useState<SpectrumData>({
    frequencies: [],
    magnitudes: [],
    timestamp: 0,
  });
  const [fftSize, setFftSize] = useState(1024);
  const [updateRate, setUpdateRate] = useState(100);
  const socketRef = useRef<any>(null);

  useEffect(() => {
    // Fetch available devices
    fetchDevices();

    // Connect to WebSocket
    socketRef.current = io('http://localhost:8000');

    socketRef.current.on('spectrum_data', (data: SpectrumData) => {
      setSpectrumData(data);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    if (selectedDevice) {
      socketRef.current.emit('subscribe_spectrum', {
        device_id: selectedDevice,
        fft_size: fftSize,
        update_rate: updateRate,
      });
    }
  }, [selectedDevice, fftSize, updateRate]);

  const fetchDevices = async () => {
    try {
      const response = await fetch('http://localhost:8000/devices');
      const data = await response.json();
      setDevices(data.devices.map((d: any) => d.id));
    } catch (error) {
      console.error('Error fetching devices:', error);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Real-Time Spectrum
      </Typography>

      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <FormControl fullWidth>
            <InputLabel>Select Device</InputLabel>
            <Select
              value={selectedDevice}
              onChange={(e) => setSelectedDevice(e.target.value)}
              label="Select Device"
            >
              {devices.map((device) => (
                <MenuItem key={device} value={device}>
                  {device}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={4}>
          <Typography gutterBottom>FFT Size</Typography>
          <Slider
            value={fftSize}
            onChange={(_, value) => setFftSize(value as number)}
            min={256}
            max={8192}
            step={256}
            marks={[
              { value: 256, label: '256' },
              { value: 1024, label: '1K' },
              { value: 4096, label: '4K' },
              { value: 8192, label: '8K' },
            ]}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <Typography gutterBottom>Update Rate (ms)</Typography>
          <Slider
            value={updateRate}
            onChange={(_, value) => setUpdateRate(value as number)}
            min={50}
            max={1000}
            step={50}
            marks={[
              { value: 50, label: '50ms' },
              { value: 100, label: '100ms' },
              { value: 500, label: '500ms' },
              { value: 1000, label: '1s' },
            ]}
          />
        </Grid>
      </Grid>

      <Paper sx={{ mt: 2, p: 2 }}>
        <Plot
          data={[
            {
              x: spectrumData.frequencies,
              y: spectrumData.magnitudes,
              type: 'scatter',
              mode: 'lines',
              name: 'Spectrum',
              line: { color: '#2196f3' },
            },
          ]}
          layout={{
            title: { text: 'Real-Time Spectrum' },
            xaxis: { title: { text: 'Frequency (Hz)' } },
            yaxis: { title: { text: 'Magnitude (dB)' } },
            height: 500,
            margin: { l: 50, r: 50, t: 50, b: 50 },
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </Paper>
    </Box>
  );
};

export default SpectrumVisualizer; 