import React, { useState } from 'react';
import { Box, Drawer, AppBar, Toolbar, Typography, List, ListItem, ListItemIcon, ListItemText, IconButton, CssBaseline, Chip, Button } from '@mui/material';
import { Menu as MenuIcon, SignalCellularAlt, Settings, Storage, Devices, LocationOn, Login as LoginIcon } from '@mui/icons-material';
import Login from './Login';
import { apiClient } from '../api/client';
import SpectrumVisualizer from './SpectrumVisualizer';
import SignalRecorder from './SignalRecorder';
import DeviceMonitor from './DeviceMonitor';
import ConfigManager from './ConfigManager';
import GeolocationMap from './GeolocationMap';
import ReceiversPanel from './ReceiversPanel';

const drawerWidth = 240;

const Dashboard: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [currentView, setCurrentView] = useState('spectrum');

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Spectrum View', icon: <SignalCellularAlt />, view: 'spectrum' },
    { text: 'Signal Recorder', icon: <Storage />, view: 'recorder' },
    { text: 'Device Monitor', icon: <Devices />, view: 'devices' },
    { text: 'Geolocation', icon: <LocationOn />, view: 'geolocation' },
    { text: 'Receivers', icon: <Devices />, view: 'receivers' },
    { text: 'Configuration', icon: <Settings />, view: 'config' },
    { text: 'Login', icon: <LoginIcon />, view: 'login' },
  ];

  const drawer = (
    <div>
      <Toolbar />
      <List>
        {menuItems.map((item) => (
          <ListItem 
            button 
            key={item.text} 
            onClick={() => setCurrentView(item.view)}
            selected={currentView === item.view}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </div>
  );

  const renderContent = () => {
    switch (currentView) {
      case 'spectrum':
        return <SpectrumVisualizer />;
      case 'recorder':
        return <SignalRecorder />;
      case 'devices':
        return <DeviceMonitor />;
      case 'geolocation':
        return <GeolocationMap />;
      case 'config':
        return <ConfigManager />;
      case 'receivers':
        return <ReceiversPanel />;
      case 'login':
        return <Login />;
      default:
        return <SpectrumVisualizer />;
    }
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            SDR Dashboard
          </Typography>
          <Chip size="small" color="success" label="Authenticated" sx={{ mr: 2 }} />
          <Button color="inherit" onClick={() => { localStorage.removeItem('ghostnet_token'); window.location.reload(); }}>
            Logout
          </Button>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: '64px',
        }}
      >
        {renderContent()}
      </Box>
    </Box>
  );
};

export default Dashboard; 