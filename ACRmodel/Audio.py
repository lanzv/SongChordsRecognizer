#!/usr/bin/env python3
import sys
import librosa
import librosa.core
import matplotlib.pyplot as plt
import librosa.display
import csv
import numpy as np
class Audio():
    """Audio file and its data.
    PATH contains string of path of this audil file.
    WAVEFORM contains array of waveform.
    SAMPLE_RATE contains information about audio's sample rate.
    """
    def __init__(self, audio_path, sr=None):
        self.PATH = audio_path
        self.WAVEFORM, self.SAMPLE_RATE = librosa.load(audio_path, sr=sr)


class BillboardFeatures():
    """
    """
    def __init__(self, features_path):
        tuning_path = features_path + "tuning.csv"
        bothchroma_path = features_path + "bothchroma.csv"

        # Get Tuning information
        self.FILE = None
        self.START = None
        self.END = None
        self.TUNING = None

        with open(tuning_path, newline='') as csvfile:
            tuning = csv.reader(csvfile)
            tuning_list = []
            for row in tuning:
                if len(tuning_list) == 0:
                    tuning_list = row
                else:
                    raise NotImplemented("Billboard tuning.csv file has unknown format.")
            self.FILE = tuning_list[0]
            self.START = tuning_list[1]
            self.END = tuning_list[2]
            self.TUNING = tuning_list[3]

        # Get NNLS Chroma
        self.TIME_BINS = None
        self.CHROMA = None

        with open(bothchroma_path, newline='') as csvfile:
            bothchroma = csv.reader(csvfile)
            bothchrom_list = []
            for row in bothchroma:
                bothchrom_list.append(row)
            bothchrom_list = np.array(bothchrom_list)

            self.TIME_BINS = bothchrom_list[:, 1].astype(np.float)
            self.CHROMA = bothchrom_list[:, 2:].astype(np.float)
        