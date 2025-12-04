import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, FormControl, InputLabel, Select, MenuItem, Slider, Grid, TextField, InputAdornment, FormControlLabel, Switch, Chip } from '@mui/material';
import { colorForDeviceId } from '../api/colors';
import Plot from 'react-plotly.js';
import { apiClient } from '../api/client';

interface SpectrumData {
  frequencies: number[];
  magnitudes: number[];
  timestamp: number;
}

const SpectrumVisualizer: React.FC = () => {
  const [selectedDevice, setSelectedDevice] = useState('');
  const [devices, setDevices] = useState<any[]>([]);
  const [spectrumData, setSpectrumData] = useState<SpectrumData>({
    frequencies: [],
    magnitudes: [],
    timestamp: 0,
  });
  const [fftSize, setFftSize] = useState(1024);
  const [updateRate, setUpdateRate] = useState(500);
  const timerRef = useRef<any>(null);
  const [centerMhz, setCenterMhz] = useState<number>(2400);
  const [spanMhz, setSpanMhz] = useState<number>(2);
  const [useGpu, setUseGpu] = useState<boolean>(false);

  useEffect(() => {
    fetchDevices();
  }, []);

  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    const poll = async () => {
      try {
        const res = await apiClient.get('/signals/spectrum', {
          params: {
            device_id: selectedDevice || undefined,
            fft_size: fftSize,
            center_freq: centerMhz * 1e6,
            span_hz: spanMhz * 1e6,
            use_gpu: useGpu,
          },
        });
        const data = res.data as { frequencies?: number[]; magnitudes?: number[]; timestamp?: number };
        setSpectrumData({
          frequencies: data.frequencies || [],
          magnitudes: data.magnitudes || [],
          timestamp: data.timestamp || 0,
        });
      } catch (err) {
        // ignore transient errors
      }
    };
    poll();
    timerRef.current = setInterval(poll, updateRate);
    return () => clearInterval(timerRef.current);
  }, [selectedDevice, fftSize, updateRate]);

  const fetchDevices = async () => {
    try {
      const res = await apiClient.get('/signals/devices');
      const data = res.data as { devices?: any[] };
      const list = data.devices || [];
      setDevices(list);
      if (!selectedDevice && list.length > 0) {
        setSelectedDevice(list[0].device_id);
      }
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
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Select Device</InputLabel>
            <Select
              value={selectedDevice}
              onChange={(e) => setSelectedDevice(e.target.value)}
              label="Select Device"
            >
              {devices.map((d) => (
                <MenuItem key={d.device_id} value={d.device_id} title={`Type: ${d.type}\nSR: ${d.sample_rate} Hz\nBW: ${d.bandwidth} Hz`}>
                  <span style={{ color: d.active ? '#4caf50' : '#f44336', marginRight: 6 }}>●</span>
                  <span style={{ marginRight: 8 }}>{d.device_id}</span>
                  <span style={{ width: 12, height: 12, display: 'inline-block', backgroundColor: colorForDeviceId(d.device_id), borderRadius: 2 }} />
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} md={3}>
          <TextField
            fullWidth
            type="number"
            label="Center (MHz)"
            value={centerMhz}
            onChange={(e) => setCenterMhz(Number(e.target.value))}
            InputProps={{ endAdornment: <InputAdornment position="end">MHz</InputAdornment> }}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>
            Receiver Velocity (m/s)
            {selectedDevice && (
              <Chip size="small" sx={{ ml: 1, bgcolor: colorForDeviceId(selectedDevice), color: '#000' }} label={selectedDevice} />
            )}
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={4}>
              <TextField
                fullWidth
                type="number"
                label="vx"
                value={(() => {
                  const d = devices.find(x => x.device_id === selectedDevice);
                  return d?.receiver_velocity?.vx ?? 0;
                })()}
                onChange={async (e) => {
                  const vx = Number(e.target.value);
                  const d = devices.find(x => x.device_id === selectedDevice);
                  if (!d) return;
                  await apiClient.post(`/signals/devices/${selectedDevice}/velocity`, null, { params: { vx, vy: d.receiver_velocity?.vy ?? 0, vz: d.receiver_velocity?.vz ?? 0 } });
                  fetchDevices();
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                type="number"
                label="vy"
                value={(() => {
                  const d = devices.find(x => x.device_id === selectedDevice);
                  return d?.receiver_velocity?.vy ?? 0;
                })()}
                onChange={async (e) => {
                  const vy = Number(e.target.value);
                  const d = devices.find(x => x.device_id === selectedDevice);
                  if (!d) return;
                  await apiClient.post(`/signals/devices/${selectedDevice}/velocity`, null, { params: { vx: d.receiver_velocity?.vx ?? 0, vy, vz: d.receiver_velocity?.vz ?? 0 } });
                  fetchDevices();
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                type="number"
                label="vz"
                value={(() => {
                  const d = devices.find(x => x.device_id === selectedDevice);
                  return d?.receiver_velocity?.vz ?? 0;
                })()}
                onChange={async (e) => {
                  const vz = Number(e.target.value);
                  const d = devices.find(x => x.device_id === selectedDevice);
                  if (!d) return;
                  await apiClient.post(`/signals/devices/${selectedDevice}/velocity`, null, { params: { vx: d.receiver_velocity?.vx ?? 0, vy: d.receiver_velocity?.vy ?? 0, vz } });
                  fetchDevices();
                }}
              />
            </Grid>
          </Grid>
        </Grid>

        <Grid item xs={12} md={6}>
          <Typography variant="subtitle2" gutterBottom>Receiver Position</Typography>
          <Grid container spacing={1}>
            <Grid item xs={4}>
              <TextField
                fullWidth
                type="number"
                label="lat"
                value={(() => { const d = devices.find(x => x.device_id === selectedDevice); return d?.receiver_position?.lat ?? 0; })()}
                onChange={async (e) => {
                  const lat = Number(e.target.value);
                  const d = devices.find(x => x.device_id === selectedDevice);
                  if (!d) return;
                  await apiClient.post(`/signals/devices/${selectedDevice}/position`, null, { params: { lat, lon: d.receiver_position?.lon ?? 0, alt_m: d.receiver_position?.alt_m ?? 0 } });
                  fetchDevices();
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                type="number"
                label="lon"
                value={(() => { const d = devices.find(x => x.device_id === selectedDevice); return d?.receiver_position?.lon ?? 0; })()}
                onChange={async (e) => {
                  const lon = Number(e.target.value);
                  const d = devices.find(x => x.device_id === selectedDevice);
                  if (!d) return;
                  await apiClient.post(`/signals/devices/${selectedDevice}/position`, null, { params: { lat: d.receiver_position?.lat ?? 0, lon, alt_m: d.receiver_position?.alt_m ?? 0 } });
                  fetchDevices();
                }}
              />
            </Grid>
            <Grid item xs={4}>
              <TextField
                fullWidth
                type="number"
                label="alt_m"
                value={(() => { const d = devices.find(x => x.device_id === selectedDevice); return d?.receiver_position?.alt_m ?? 0; })()}
                onChange={async (e) => {
                  const alt_m = Number(e.target.value);
                  const d = devices.find(x => x.device_id === selectedDevice);
                  if (!d) return;
                  await apiClient.post(`/signals/devices/${selectedDevice}/position`, null, { params: { lat: d.receiver_position?.lat ?? 0, lon: d.receiver_position?.lon ?? 0, alt_m } });
                  fetchDevices();
                }}
              />
            </Grid>
          </Grid>
        </Grid>

        <Grid item xs={12} md={3}>
          <TextField
            fullWidth
            type="number"
            label="Span (MHz)"
            value={spanMhz}
            onChange={(e) => setSpanMhz(Number(e.target.value))}
            InputProps={{ endAdornment: <InputAdornment position="end">MHz</InputAdornment> }}
          />
        </Grid>

        <Grid item xs={12} md={3}>
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

        <Grid item xs={12} md={3}>
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

        <Grid item xs={12} md={3}>
          <FormControlLabel
            control={<Switch checked={useGpu} onChange={(_, v) => setUseGpu(v)} />}
            label="Use GPU"
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