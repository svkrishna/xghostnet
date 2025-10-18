import axios from 'axios';

const baseURL = process.env.REACT_APP_API_BASE || '/api';
export const apiClient = axios.create({
  baseURL,
  timeout: 15000,
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('ghostnet_token') || process.env.REACT_APP_API_TOKEN || 'dev-token';
  if (token) {
    config.headers = config.headers || {};
    (config.headers as any).Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      localStorage.removeItem('ghostnet_token');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);


