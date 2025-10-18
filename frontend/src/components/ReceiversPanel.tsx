import React, { useEffect, useState } from 'react';
import { Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Button, Chip, Snackbar, Alert, Dialog, DialogTitle, DialogContent, DialogActions, List, ListItem, ListItemText } from '@mui/material';
import { colorForDeviceId } from '../api/colors';
import { apiClient } from '../api/client';
import { saveAs } from 'file-saver';

interface DeviceRow {
  device_id: string;
  type?: string;
  active?: boolean;
  sample_rate?: number;
  bandwidth?: number;
  receiver_velocity?: { vx?: number; vy?: number; vz?: number };
  receiver_position?: { lat?: number; lon?: number; alt_m?: number };
}

const ReceiversPanel: React.FC = () => {
  const [rows, setRows] = useState<DeviceRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{open: boolean, message: string, severity: 'success'|'error'}>({open: false, message: '', severity: 'success'});
  const [validateOpen, setValidateOpen] = useState(false);
  const [validateReport, setValidateReport] = useState<any>({ summary: null, report: [] });

  const load = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get('/signals/devices');
      setRows(res.data?.devices || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const updateCell = (idx: number, path: string[], value: number) => {
    setRows(prev => {
      const next = [...prev];
      const row = { ...next[idx] } as any;
      let cursor = row;
      for (let i = 0; i < path.length - 1; i++) {
        const k = path[i];
        cursor[k] = cursor[k] ? { ...cursor[k] } : {};
        cursor = cursor[k];
      }
      cursor[path[path.length - 1]] = value;
      next[idx] = row;
      return next;
    });
  };

  const bulkSave = async () => {
    setSaving(true);
    try {
      const payload = rows.map(r => ({
        device_id: r.device_id,
        receiver_velocity: r.receiver_velocity,
        receiver_position: r.receiver_position,
      }));
      await apiClient.post('/signals/devices/bulk', payload);
      await load();
      setToast({ open: true, message: `Saved ${payload.length} receivers`, severity: 'success' });
    } finally {
      setSaving(false);
    }
  };

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json;charset=utf-8' });
    saveAs(blob, 'receivers_layout.json');
  };

  const importJson = async (file: File) => {
    const text = await file.text();
    const data = JSON.parse(text);
    if (!Array.isArray(data)) return;
    const payload = data.map((r: any) => ({
      device_id: r.device_id,
      receiver_velocity: r.receiver_velocity,
      receiver_position: r.receiver_position,
    }));
    const validate = await apiClient.post('/signals/devices/bulk/validate', payload);
    const rep = validate.data;
    setValidateReport(rep);
    if (rep?.summary?.errors > 0) {
      setValidateOpen(true);
      return;
    }
    await apiClient.post('/signals/devices/bulk', payload);
    await load();
    setToast({ open: true, message: `Imported ${payload.length} receivers`, severity: 'success' });
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6">Receivers</Typography>
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button variant="contained" onClick={bulkSave} disabled={saving || loading}>Save All</Button>
          <Button size="small" variant="outlined" onClick={() => {
            setRows(prev => prev.map(r => ({
              ...r,
              receiver_position: {
                ...(r.receiver_position || {}),
                alt_m: (r.receiver_position?.alt_m ?? 0) * 0.3048,
                alt_units: 'm'
              }
            }))
            );
          }}>Set all alt to meters</Button>
          <Button size="small" variant="outlined" onClick={() => {
            setRows(prev => prev.map(r => ({
              ...r,
              receiver_position: {
                ...(r.receiver_position || {}),
                alt_m: Math.round(((r.receiver_position?.alt_m ?? 0) / 0.3048) * 100) / 100,
                alt_units: 'ft'
              }
            }))
            );
          }}>Set all alt to feet</Button>
          <Button variant="outlined" onClick={exportJson}>Export JSON</Button>
          <Button variant="outlined" component="label">
            Import JSON
            <input hidden type="file" accept="application/json" onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) importJson(f);
            }} />
          </Button>
        </Box>
      </Box>
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Status</TableCell>
              <TableCell>Device ID</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>SR (Hz)</TableCell>
              <TableCell>BW (Hz)</TableCell>
              <TableCell align="center" colSpan={3}>Velocity (m/s)</TableCell>
              <TableCell align="center" colSpan={3}>Position (lat, lon, alt m)</TableCell>
            </TableRow>
              <TableRow>
              <TableCell></TableCell>
              <TableCell></TableCell>
              <TableCell></TableCell>
              <TableCell></TableCell>
              <TableCell></TableCell>
              <TableCell>vx</TableCell>
              <TableCell>vy</TableCell>
              <TableCell>vz</TableCell>
              <TableCell>lat</TableCell>
              <TableCell>lon</TableCell>
                <TableCell>alt (m/ft)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((r, idx) => (
              <TableRow key={r.device_id}>
                <TableCell>
                  <Chip size="small" label={r.active ? 'Active' : 'Inactive'} color={r.active ? 'success' as any : 'default' as any} />
                </TableCell>
                <TableCell>
                  <span style={{ width: 12, height: 12, display: 'inline-block', backgroundColor: colorForDeviceId(r.device_id), borderRadius: 2, marginRight: 6 }} />
                  {r.device_id}
                </TableCell>
                <TableCell>{r.type}</TableCell>
                <TableCell>{r.sample_rate}</TableCell>
                <TableCell>{r.bandwidth}</TableCell>
                <TableCell>
                  <TextField size="small" type="number" value={r.receiver_velocity?.vx ?? 0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') { bulkSave(); }
                      if (e.key === 'ArrowDown' && idx < rows.length - 1) {
                        const next = (e.target as HTMLInputElement).closest('tr')?.nextElementSibling?.querySelector('input') as HTMLInputElement;
                        next?.focus();
                      }
                      if (e.key === 'ArrowUp' && idx > 0) {
                        const prev = (e.target as HTMLInputElement).closest('tr')?.previousElementSibling?.querySelector('input') as HTMLInputElement;
                        prev?.focus();
                      }
                    }}
                    onChange={(e) => updateCell(idx, ['receiver_velocity','vx'], Number(e.target.value))} />
                </TableCell>
                <TableCell>
                  <TextField size="small" type="number" value={r.receiver_velocity?.vy ?? 0}
                    onChange={(e) => updateCell(idx, ['receiver_velocity','vy'], Number(e.target.value))} />
                </TableCell>
                <TableCell>
                  <TextField size="small" type="number" value={r.receiver_velocity?.vz ?? 0}
                    onChange={(e) => updateCell(idx, ['receiver_velocity','vz'], Number(e.target.value))} />
                </TableCell>
                <TableCell>
                  <TextField size="small" type="number" value={r.receiver_position?.lat ?? 0}
                    error={typeof r.receiver_position?.lat === 'number' && (r.receiver_position!.lat! < -90 || r.receiver_position!.lat! > 90)}
                    helperText={typeof r.receiver_position?.lat === 'number' && (r.receiver_position!.lat! < -90 || r.receiver_position!.lat! > 90) ? 'lat [-90,90]' : ''}
                    onChange={(e) => updateCell(idx, ['receiver_position','lat'], Number(e.target.value))} />
                </TableCell>
                <TableCell>
                  <TextField size="small" type="number" value={r.receiver_position?.lon ?? 0}
                    error={typeof r.receiver_position?.lon === 'number' && (r.receiver_position!.lon! < -180 || r.receiver_position!.lon! > 180)}
                    helperText={typeof r.receiver_position?.lon === 'number' && (r.receiver_position!.lon! < -180 || r.receiver_position!.lon! > 180) ? 'lon [-180,180]' : ''}
                    onChange={(e) => updateCell(idx, ['receiver_position','lon'], Number(e.target.value))} />
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField size="small" type="number" value={r.receiver_position?.alt_m ?? 0}
                      onChange={(e) => updateCell(idx, ['receiver_position','alt_m'], Number(e.target.value))} />
                    <Button size="small" variant="outlined" onClick={() => {
                      const m = r.receiver_position?.alt_m ?? 0;
                      const ft = m / 0.3048;
                      updateCell(idx, ['receiver_position','alt_m'], Math.round(ft * 100) / 100);
                      updateCell(idx, ['receiver_position','alt_units'], 'ft');
                    }}>to ft</Button>
                    <Button size="small" variant="outlined" onClick={() => {
                      const ft = r.receiver_position?.alt_m ?? 0;
                      const m = ft * 0.3048;
                      updateCell(idx, ['receiver_position','alt_m'], Math.round(m * 100) / 100);
                      updateCell(idx, ['receiver_position','alt_units'], 'm');
                    }}>to m</Button>
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
    <Snackbar open={toast.open} autoHideDuration={3000} onClose={() => setToast({...toast, open: false})}>
      <Alert onClose={() => setToast({...toast, open: false})} severity={toast.severity} sx={{ width: '100%' }}>
        {toast.message}
      </Alert>
    </Snackbar>
    <Dialog open={validateOpen} onClose={() => setValidateOpen(false)} maxWidth="md" fullWidth>
      <DialogTitle>Import Validation</DialogTitle>
      <DialogContent dividers>
        <Typography variant="body2" gutterBottom>
          {validateReport?.summary ? `OK: ${validateReport.summary.ok}  Errors: ${validateReport.summary.errors}  Total: ${validateReport.summary.total}` : ''}
        </Typography>
        <List dense>
          {validateReport?.report?.map((r: any, idx: number) => (
            <ListItem key={idx} secondaryAction={
              <Button size="small" onClick={() => {
                // Focus row in table for quick fix if it exists
                const i = rows.findIndex(x => x.device_id === r.device_id);
                if (i >= 0) {
                  const el = document.querySelectorAll('table tbody tr')[i] as HTMLElement;
                  el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
              }}>Focus</Button>
            }>
              <ListItemText
                primary={`${r.device_id} — ${r.ok ? 'OK' : 'Errors'}`}
                secondary={(r.errors || []).join('; ')}
              />
            </ListItem>
          ))}
        </List>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setValidateOpen(false)}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default ReceiversPanel;
