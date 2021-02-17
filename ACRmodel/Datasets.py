#!/usr/bin/env python3
from ChordsData import ChordsData
from Audio import Audio
import os
from glob import glob
import pickle
import lzma
import librosa
import numpy as np
import re
import sys

class IsophonicsDataset():
    """Isophonics Dataset.
    The train set contains 225 Bealtes, Queen, Carole King or Zweieck songs. 
    DATA contains audio waveform features.
    LABELS contains chord annotation for all song duration.
    """
    SAMPLE_RATE = 44100
    NFFT = 2**14

    
    def __init__(self, audio_directory, annotations_directory):
        
        self.DATA = []
        self.LABELS = []

        audio_paths = sorted(glob(os.path.join(audio_directory, '*.wav')))
        annotations_paths = sorted(glob(os.path.join(annotations_directory, '*.lab')))

        if not len(audio_paths) == len(annotations_paths):
            raise Exception("The number of WAV files doesn't equal the number of LAB files.")

        for audio_path, lab_path in zip(audio_paths, annotations_paths):
            self.DATA.append(Audio(audio_path, self.SAMPLE_RATE))
            self.LABELS.append(ChordsData(lab_path))

        print("[INFO] The Dataset was successfully initialized.")


       
    def get_preprocessed_dataset(self, window_size=5, flattened_window=True, ms_intervals=100, to_skip=5):
        """
        Preprocess IsophonicsDataset dataset.
        Create features from self.DATA and its corresponding targets from self.LABELS.
        
        Parameters
        ----------
        window : int
            how many spectrograms on left and on right we should take 
        flattened_window : bool
            True if we want to flatten spectrograms to one array, otherwise False
        ms_intervals : int
            miliseconds between generated spectrogram
        to_skip : int
            how many spectrogram we want to skip when creating new feature set
        Returns
        -------
        prep_data : np array
            flattened window of logarithmized mel spectrograms arround specific time point

        prep_targets : np array
            integers of chord labels for specific time point
        """
        prep_data = []
        prep_targets = []
        hop_length = int(self.SAMPLE_RATE/(1000/ms_intervals))
        k = 0
        # Iterate over all audio files
        for audio, label in zip(self.DATA, self.LABELS):
            print(k)
            k = k+1
            # Get log mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(audio.WAVEFORM, audio.SAMPLE_RATE, n_fft=self.NFFT, hop_length=hop_length)
            log_spectrogram = librosa.amplitude_to_db(mel_spectrogram)
            mel_length, num_samples = log_spectrogram.shape

            # Collect data for each spectrogram sample
            j = 0 # labels index
            for i in [index for index in range(num_samples) if index%to_skip==0]:
                # Get data window with zero margin
                if flattened_window:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((abs(min(0, i-window_size)), mel_length)),
                            np.array(log_spectrogram[:, max(0, i-window_size):min(i+window_size+1, num_samples)]).swapaxes(0,1),
                            np.zeros((abs(min(0, (num_samples)-(i+window_size+1))), mel_length))
                        ), axis = 0).flatten()
                    )
                else:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((abs(min(0, i-window_size)), mel_length)),
                            np.array(log_spectrogram[:, max(0, i-window_size):min(i+window_size+1, num_samples)]).swapaxes(0,1),
                            np.zeros((abs(min(0, (num_samples)-(i+window_size+1))), mel_length))
                        ), axis = 0)
                    )


                # Get label
                second = float(i)/(float(self.SAMPLE_RATE) / float(hop_length))
                while j < len(label.START) and second > label.START[j] :
                    j = j + 1
                if j == len(label.START):
                    prep_targets.append(IsophonicsDataset.get_integered_chord("N"))
                else:
                    prep_targets.append(IsophonicsDataset.get_integered_chord(label.CHORD[j]))

        print("[INFO] The Dataset was successfully preprocessed.")
        return np.array(prep_data), np.array(prep_targets)




    @staticmethod
    def get_integered_chord(chord):
        """
        Map chord label in string with its index.
        
        Parameters
        ----------
        chord : string
            labled chord, for instance N for none chord, or G#:min7
        Returns
        -------
        chord_index : int 
            index of chord passed on input, N has 0, other chord are integer in range of 1 and 24
        """
        chord = re.sub('6|7|9|11|13|maj|\/[0-9]|\/\#[0-9]|\/b[0-9]|\(.*\)', '', chord)
        chord = re.sub(':$', '', chord)
        chords = {
            "N" : 0, 
            "C" : 1,
            "C:min" : 2, 
            "C#" : 3,       "Db" : 3, 
            "C#:min" : 4,   "Db:min" : 4, 
            "D" : 5,        
            "D:min": 6,
            "D#" : 7,       "Eb" : 7,
            "D#:min": 8,    "Eb:min" : 8, 
            "E" : 9,
            "E:min": 10, 
            "F" : 11, 
            "F:min": 12, 
            "F#": 13,       "Gb" : 13,
            "F#:min": 14,   "Gb:min" : 14,
            "G" : 15, 
            "G:min" : 16, 
            "G#": 17,       "Ab" : 17,
            "G#:min" : 18,  "Ab:min" : 18,
            "A" : 19,       
            "A:min" : 20, 
            "A#" :21,       "Bb" : 21,
            "A#:min":22,    "Bb:min" : 22,
            "B":23,         "Cb" : 23,
            "B:min":24,     "Cb:min" : 24
        }
        if chord in chords:
            return chords[chord]
        else:
            return 0



    def save_preprocessed_dataset(self, dest = "./Datasets/preprocessed_IsophonicsDataset.ds", window_size=5, flattened_window=True, ms_intervals=100, to_skip=5):
        """
        Save preprocessed data from this dataset to destination path 'dest' by default as a .ds file.
        
        Parameters
        ----------
        dest : str
            path to preprocessed data
        window : int
            how many spectrograms on left and on right we should take 
        flattened_window : bool
            True if we want to flatten spectrograms to one array, otherwise False
        ms_intervals : int
            miliseconds between generated spectrogram
        to_skip : int
            how many spectrogram we want to skip when creating new feature set
        """
        # Serialize the dataset.
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((self.get_preprocessed_dataset(window_size, flattened_window, ms_intervals, to_skip)), dataset_file)

        print("[INFO] The Dataset was saved successfully.")


    @staticmethod
    def load_preprocessed_dataset(dest = "./Datasets/preprocessed_IsophonicsDataset.ds"): 
        """
        Load preprocessed data from this dataset from destination path 'dest'. Targets and Data are stored by default as a .ds file.
        
        Parameters
        ----------
        dest : str
            path to preprocessed data
        Returns
        -------
        prep_data : np array
            flattened window of logarithmized mel spectrograms arround specific time point

        prep_targets : np array
            integers of chord labels for specific time point
        """
        with lzma.open(dest, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        print("[INFO] The Dataset was loaded successfully.")
        return dataset