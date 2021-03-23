import numpy as np
from ACR_Training.Spectrograms import log_mel_spectrogram
from ACR_Training.Datasets import IsophonicsDataset
from ACR_Training.annotation_maps import keys_map, chords_map

class DataPreprocessor():

    @staticmethod
    def flatten_preprocess(waveform, sample_rate=44100, hop_length=512, nfft=2*14, window_size=5, spectrogram_generator=log_mel_spectrogram, norm_to_C=False, key='C'):
        """
        Preprocess function that prepares data features as a spectrogram array flattened with context arround.

        Parameters
        ----------
        waveform : list of floats
            data of audio waveform
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        nfft : int
            length of FFT, power of 2
        window_size : int
            how many spectrograms on left and on right we should take 
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        Returns
        -------
        prep_data : np array
            flattened window of spectrograms arround specific time point
        """
        prep_data = []
        # Iterate over all audio files
        # Get spectrogram
        spectrogram = IsophonicsDataset.preprocess_audio(waveform=waveform, sample_rate=sample_rate, spectrogram_generator=spectrogram_generator, nfft=nfft, hop_length=hop_length, norm_to_C=norm_to_C, key=key)
        spec_length, num_samples = spectrogram.shape

        # Collect data for each spectrogram sample
        for i in range(num_samples):
            # Get data window with zero margin
            prep_data.append(
                np.concatenate((
                    np.zeros((abs(min(0, i-window_size)), spec_length)),
                    np.array(spectrogram[:, max(0, i-window_size):min(i+window_size+1, num_samples)]).swapaxes(0,1),
                    np.zeros((abs(min(0, (num_samples)-(i+window_size+1))), spec_length))
                ), axis = 0).flatten()
            )

        return np.array(prep_data)

    @staticmethod
    def transpose(chord_sequence, from_key='C', to_key='C'):
        """
        Transpose chord indices.

        Parameters
        ----------
        chord_sequence : int list
            list of chord indices we want to transpose
        from_key : String
            key string that chord sequence should be transposed from, C, C#, D, ..
        to_key : String
            key string that chord sequence should be transposed to, C, C#, D, ..
        Returns
        -------
        transposed_sequence : int list
            list of chord indices transposed from from_key to to_key
        """
        from_ind = keys_map[from_key]
        to_ind = keys_map[to_key]
        diff = (to_ind + len(keys_map) - from_ind) % len(keys_map)

        transposed_sequence = []
        for chord in chord_sequence:
            if (chord + 2*diff < len(chords_map)):
                transposed_sequence.append(chord+2*diff)
            else:
                transposed_sequence.append((chord+2*diff)%len(chords_map) + 1)

        return transposed_sequence

    @staticmethod
    def chord_indices_to_notations(chord_sequence):
        """
        Create a notation chord sequence ('C, D, D, E, D,..') from chord indices.

        Parameters
        ----------
        chord_sequence : int list
            chord indices
        Returns
        -------
        chord_notations : String list
            list of chords mapped to its notations
        """
        chord_notations = []

        not_list = list(chords_map.keys())
        ind_list = list(chords_map.values())
        for chord in chord_sequence:
            position = ind_list.index(chord)
            chord_notations.append(not_list[position])

        return chord_notations