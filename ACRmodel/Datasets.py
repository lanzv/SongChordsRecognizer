#!/usr/bin/env python3
from Annotations import ChordSequence, KeySequence
from annotation_maps import chords_map, keys_map, key_modes_map
from Audio import Audio
import os
from glob import glob
import pickle
import lzma
import librosa
import numpy as np
import re
from Spectrograms import log_mel_spectrogram

class IsophonicsDataset():
    """Isophonics Dataset.
    The train set contains 225 Bealtes, Queen, Carole King or Zweieck songs. 
    DATA contains audio waveform features.
    LABELS contains chord annotation for all song duration.
    """
    def __init__(self, audio_directory, annotations_directory, sample_rate=44100, nfft=2**14):
        
        self.SAMPLE_RATE = sample_rate
        self.NFFT = nfft

        self.DATA = []
        self.CHORDS = []
        self.KEYS =  []

        audio_paths = sorted(glob(os.path.join(audio_directory, '*.wav')))
        chord_annotations_paths = sorted(glob(os.path.join(annotations_directory+'/CHORDS', '*.lab')))
        key_annotations_paths = sorted(glob(os.path.join(annotations_directory+'/KEYS', '*.lab')))

        if not (len(audio_paths) == len(chord_annotations_paths) and len(audio_paths) == len(key_annotations_paths)):
            raise Exception("The number of WAV files doesn't equal the number of annotation files.")

        for audio_path, chord_lab_path, key_lab_path in zip(audio_paths, chord_annotations_paths, key_annotations_paths):
            self.DATA.append(Audio(audio_path, self.SAMPLE_RATE))
            self.CHORDS.append(ChordSequence(chord_lab_path))
            self.KEYS.append(KeySequence(key_lab_path))

        print("[INFO] The Dataset was successfully initialized.")

    


       
    def get_preprocessed_dataset(self, window_size=5, flattened_window=True, ms_intervals=100, to_skip=5, norm_to_C=False, spectrogram_generator=log_mel_spectrogram):
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
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        Returns
        -------
        prep_data : np array
            flattened window of logarithmized mel spectrograms arround specific time point
        prep_targets : np array
            integers of chord labels for specific time point
        """
        prep_data = []
        prep_targets = []
        hop_length = int(self.SAMPLE_RATE/(ms_intervals/10))
        print(hop_length)
        k = 0
        # Iterate over all audio files
        for audio, chords, keys in zip(self.DATA, self.CHORDS, self.KEYS):
            print(k)
            k = k+1
            # Get log mel spectrogram
            spectrogram = IsophonicsDataset.preprocess_audio(waveform=audio.WAVEFORM, sample_rate=audio.SAMPLE_RATE, spectrogram_generator=spectrogram_generator, nfft=self.NFFT, hop_length=hop_length, norm_to_C=norm_to_C, key=keys.get_first_key())
            spec_length, num_samples = spectrogram.shape

            # Collect data for each spectrogram sample
            j = 0 # labels index
            for i in [index for index in range(num_samples) if index%to_skip==0]:
                # Get data window with zero margin
                if flattened_window:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((abs(min(0, i-window_size)), spec_length)),
                            np.array(spectrogram[:, max(0, i-window_size):min(i+window_size+1, num_samples)]).swapaxes(0,1),
                            np.zeros((abs(min(0, (num_samples)-(i+window_size+1))), spec_length))
                        ), axis = 0).flatten()
                    )
                else:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((abs(min(0, i-window_size)), spec_length)),
                            np.array(spectrogram[:, max(0, i-window_size):min(i+window_size+1, num_samples)]).swapaxes(0,1),
                            np.zeros((abs(min(0, (num_samples)-(i+window_size+1))), spec_length))
                        ), axis = 0)
                    )


                # Get label
                second = float(i)/(float(self.SAMPLE_RATE) / float(hop_length))
                while j < len(chords.START) and second > chords.START[j] :
                    j = j + 1
                if j == len(chords.START):
                    prep_targets.append(IsophonicsDataset.get_integered_chord("N", norm_to_C, keys.get_first_key()))
                else:
                    prep_targets.append(IsophonicsDataset.get_integered_chord(chords.CHORD[j], norm_to_C, keys.get_first_key()))

        print("[INFO] The Dataset was successfully preprocessed.")
        return np.array(prep_data), np.array(prep_targets)


    @staticmethod
    def preprocess_audio(waveform, sample_rate, spectrogram_generator, nfft, hop_length, norm_to_C=False, key='C'):
        """
        Preprocess audio waveform, shift pitches to C major key (and its modes ... dorian, phrygian, aiolian, lydian, ...) and generate mel and log spectrograms.
        
        Parameters
        ----------
        waveform : list of floats
            data of audio waveform
        sample_rate : int
            audio sample rate
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        nfft : int
            length of FFT, power of 2
        hop_length : int
            number of target rate, sample_rate/hop_length = interval between two spectrograms in miliseconds
        norm_to_C : bool
            True, if we want to normalize all songs to C major key (and its modes)
        key : string
            label of audio music key
        Returns
        -------
        log_spectrogram : list of float lists
            list of logarithmized song mel spectrograms
        """
        # Get number of half tones to transpose
        if norm_to_C:
            splited_key = key.split(":")
            if len(splited_key) == 1:
                mode_shift = 0
            elif len(splited_key) == 2:
                mode_shift = key_modes_map[splited_key[1]]
            else:
                raise Exception("Some key mode format is not supported.")
            to_shift = keys_map[splited_key[0]] - mode_shift
            n_steps = -(to_shift%12) if to_shift%12 < 7 else 12-(to_shift%12)
        else:
            n_steps = 0
        # transpose song to C    
        waveform_shifted = librosa.effects.pitch_shift(waveform, sample_rate, n_steps=n_steps)
        # Get spectrogram
        spectrogram = spectrogram_generator(waveform_shifted, sample_rate, nfft, hop_length)

        return spectrogram



    @staticmethod
    def get_integered_chord(chord, norm_to_C=False, key='C'):
        """
        Map chord label in string with its index.
        
        Parameters
        ----------
        chord : string
            labled chord, for instance N for none chord, or G#:min7
        norm_to_C : bool
            True, if we want to normalize chord to C major key (and its modes)
        key : string
            true label of audio music key
        Returns
        -------
        chord_index : int 
            index of chord passed on input (or its normalization alternative), N has 0, other chord are integer in range of 1 and 24
        """
        # Get number of half tones to transpose
        if norm_to_C:
            splited_key = key.split(":")
            if len(splited_key) == 1:
                mode_shift = 0
            elif len(splited_key) == 2:
                mode_shift = key_modes_map[splited_key[1]]
            else:
                raise Exception("Some key mode format is not supported.")
            to_shift = keys_map[splited_key[0]] - mode_shift
            n_steps = -(to_shift%12) if to_shift%12 < 7 else 12-(to_shift%12)
        else:
            n_steps = 0
        # Simplify chord label
        chord = re.sub('6|7|9|11|13|maj|\/[0-9]|\/\#[0-9]|\/b[0-9]|\(.*\)', '', chord)
        chord = re.sub(':$', '', chord)
        # Get chord index
        if chord in chords_map and not chord == "N":
            if chords_map[chord] + 2*n_steps <= 0:
                return chords_map[chord] + 2*n_steps + 24
            elif chords_map[chord] + 2*n_steps > 24:
                return chords_map[chord] + 2*n_steps - 24
            else:
                return chords_map[chord] + 2*n_steps
        else:
            return 0



    def save_preprocessed_dataset(self, dest = "./Datasets/preprocessed_IsophonicsDataset.ds", window_size=5, flattened_window=True, ms_intervals=100, to_skip=5, norm_to_C=False, spectrogram_generator=log_mel_spectrogram):
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
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        """
        # Serialize the dataset.
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((self.get_preprocessed_dataset(window_size, flattened_window, ms_intervals, to_skip, norm_to_C, spectrogram_generator)), dataset_file)

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