#!/usr/bin/env python3
import sys
import librosa
import librosa.core
import matplotlib.pyplot as plt
import librosa.display

class Audio():
    """Audio file and its data.
    PATH contains string of path of this audil file.
    WAVEFORM contains array of waveform.
    SAMPLE_RATE contains information about audio's sample rate.
    """
    def __init__(self, audio_path, sr=None):
        self.PATH = audio_path
        self.WAVEFORM, self.SAMPLE_RATE = librosa.load(audio_path, sr=sr)
        
        