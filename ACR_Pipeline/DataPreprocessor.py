from typing import Sequence
import numpy as np
from ACR_Training.Spectrograms import log_mel_spectrogram
from ACR_Training.Datasets import IsophonicsDataset
from ACR_Training.annotation_maps import keys_map, chords_map, N_CHORDS, N_KEYS

class DataPreprocessor():

    @staticmethod
    def sequence_preprocess(waveform, sample_rate=44100, hop_length=512, nfft=2*14, n_frames=1000, spectrogram_generator=log_mel_spectrogram, norm_to_C=False, key='C'):
        """
        Preprocess function that prepares data features as a spectrogram sequences of n_frames frames.

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
        n_frames : int
             how many frames should be included in a subsequence of a song
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        norm_to_C : bool
            True if we want to transpose all songs to C key
        key : string
            label of audio music key
        Returns
        -------
        prep_data : np array
            sequences of song's chroma vectors separated to n_frames frames
        """
        prep_data = []

        spectrogram = IsophonicsDataset.preprocess_audio(waveform=waveform, sample_rate=sample_rate, spectrogram_generator=spectrogram_generator, nfft=nfft, hop_length=hop_length, norm_to_C=norm_to_C, key=key)

        for i in range((int)(spectrogram.shape[0]/n_frames)):
            # Get chroma
            prep_data.append(spectrogram[i*n_frames:(i+1)*n_frames])

        # Embed zero chromas to fill n_frames frames
        last_ind = (int)(spectrogram.shape[0]/n_frames)
        _, n_features = spectrogram.shape
        prep_data.append(
            np.concatenate((
                np.array(spectrogram[last_ind*n_frames:]),
                np.zeros((n_frames - (len(spectrogram) - last_ind*n_frames), n_features))
            ), axis=0 )
        )

        return np.array(prep_data)


    @staticmethod
    def flatten_preprocess(waveform, sample_rate=44100, hop_length=512, nfft=2*14, window_size=5, spectrogram_generator=log_mel_spectrogram, norm_to_C=False, key='C', skip_coef=1):
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
        norm_to_C : bool
            True if we want to transpose all songs to C key
        key : string
            label of audio music key
        skip_coef : int
            coeficient that multiplies window shifts -> some spectrogram are skipped in the flattened window
        Returns
        -------
        prep_data : np array
            flattened window of spectrograms arround specific time point
        """
        prep_data = []
        # Iterate over all audio files
        # Get spectrogram
        spectrogram = IsophonicsDataset.preprocess_audio(waveform=waveform, sample_rate=sample_rate, spectrogram_generator=spectrogram_generator, nfft=nfft, hop_length=hop_length, norm_to_C=norm_to_C, key=key)
        spectrogram = np.array(spectrogram)
        spec_length, num_samples = spectrogram.shape

        # Collect data for each spectrogram sample
        for i in range(num_samples):
            # Get data window with zero margin
            n_pre_zeros, window_indices, n_post_zeros = DataPreprocessor.__get_flatten_indices(actual_index=i, num_samples=num_samples, skip_coef=skip_coef, window_size=window_size)
            prep_data.append(
                np.concatenate((
                    np.zeros((n_pre_zeros, spec_length)),
                    np.array(spectrogram[:, window_indices]).swapaxes(0,1),
                    np.zeros((n_post_zeros, spec_length))
                ), axis = 0).flatten()
            )

        return np.array(prep_data)



    @staticmethod
    def __get_flatten_indices(actual_index, num_samples, skip_coef=1, window_size=5):
        """
        Find indices of spectrogram included in the window.

        Parameters
        ----------
        actual_index : int
            index of acutal spectrogram that we are creating a window arround
        num_samples : int
            number of spectrogram samples, the maximum index
        skip_coef : int
            coeficient that multiplies window shifts -> some spectrogram are skipped in the flattened window
        window_size : int
            how many spectrograms on left and on right we should take
        Returns
        -------
        n_pre_zeros : int
            number of zeros before spectrogram in the flattened window
        window_indices : int list
            indices of spectrogram included in the flattened window
        n_post_zeros : int
            number of zeros after spectrogram in the flattened window
        """
        n_pre_zeros = 0
        window_indices = []
        n_post_zeros = 0
        for i in range(window_size * 2 + 1):
            if (actual_index - window_size*skip_coef) + i*skip_coef >= 0 and (actual_index - window_size*skip_coef) + i*skip_coef <  num_samples: 
                window_indices.append((actual_index - window_size*skip_coef) + i*skip_coef)
            elif (actual_index - window_size*skip_coef) + i*skip_coef < 0 :
                n_pre_zeros = n_pre_zeros + 1
            elif (actual_index - window_size*skip_coef) + i*skip_coef >= num_samples:
                n_post_zeros = n_post_zeros + 1
            else:
                raise Exception("DataPreprocessor __get_flatten_indices faced to unexptected situation.")

        return n_pre_zeros, window_indices, n_post_zeros




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
        diff = (to_ind + N_KEYS - from_ind) % N_KEYS

        transposed_sequence = []
        for chord in chord_sequence:
            if chord == 0: # chord is unknown -> N
                transposed_sequence.append(chord)
            elif (chord + 2*diff < N_CHORDS):
                transposed_sequence.append(chord+2*diff)
            else:
                transposed_sequence.append((chord+2*diff)%N_CHORDS + 1)

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