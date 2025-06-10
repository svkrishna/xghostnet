import React, { useState, useEffect } from 'react';
import { Box, Button, List, ListItem, ListItemText, Typography, IconButton } from '@mui/material';
import { PlayArrow, Stop, Delete, Download } from '@mui/icons-material';

interface Recording {
  id: string;
  filename: string;
  timestamp: string;
  duration: number;
}

const SignalRecorder: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selectedDevice, setSelectedDevice] = useState('');

  useEffect(() => {
    // Fetch existing recordings
    fetchRecordings();
  }, []);

  const fetchRecordings = async () => {
    try {
      const response = await fetch('http://localhost:8000/recordings');
      const data = await response.json();
      setRecordings(data.recordings);
    } catch (error) {
      console.error('Error fetching recordings:', error);
    }
  };

  const startRecording = async () => {
    try {
      const response = await fetch('http://localhost:8000/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_id: selectedDevice }),
      });
      if (response.ok) {
        setIsRecording(true);
      }
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = async () => {
    try {
      const response = await fetch('http://localhost:8000/stop_recording', {
        method: 'POST',
      });
      if (response.ok) {
        setIsRecording(false);
        fetchRecordings();
      }
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  };

  const playRecording = async (recordingId: string) => {
    try {
      await fetch(`http://localhost:8000/play/${recordingId}`, {
        method: 'POST',
      });
    } catch (error) {
      console.error('Error playing recording:', error);
    }
  };

  const deleteRecording = async (recordingId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/recordings/${recordingId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        fetchRecordings();
      }
    } catch (error) {
      console.error('Error deleting recording:', error);
    }
  };

  const downloadRecording = async (recordingId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/recordings/${recordingId}/download`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `recording_${recordingId}.wav`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading recording:', error);
    }
  };

  return (
    <Box>
      <Box sx={{ mb: 2 }}>
        <Button
          variant="contained"
          color={isRecording ? "error" : "primary"}
          onClick={isRecording ? stopRecording : startRecording}
          startIcon={isRecording ? <Stop /> : <PlayArrow />}
        >
          {isRecording ? "Stop Recording" : "Start Recording"}
        </Button>
      </Box>

      <Typography variant="h6" gutterBottom>
        Recordings
      </Typography>

      <List>
        {recordings.map((recording) => (
          <ListItem
            key={recording.id}
            secondaryAction={
              <Box>
                <IconButton onClick={() => playRecording(recording.id)}>
                  <PlayArrow />
                </IconButton>
                <IconButton onClick={() => downloadRecording(recording.id)}>
                  <Download />
                </IconButton>
                <IconButton onClick={() => deleteRecording(recording.id)}>
                  <Delete />
                </IconButton>
              </Box>
            }
          >
            <ListItemText
              primary={recording.filename}
              secondary={`Duration: ${recording.duration}s | Recorded: ${recording.timestamp}`}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
};

export default SignalRecorder; 