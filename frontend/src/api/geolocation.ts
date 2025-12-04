import { apiClient } from './client';

export interface KnownDevice {
  device_id: string;
  latitude: number;
  longitude: number;
  signal_strength: number;
  timestamp: number;
}

export async function fetchKnownDevices(): Promise<KnownDevice[]> {
  const res = await apiClient.get('/geolocation/devices');
  const data = res.data as { status?: string; devices?: KnownDevice[] };
  if (data && data.status === 'success' && Array.isArray(data.devices)) {
    return data.devices as KnownDevice[];
  }
  return [];
}


