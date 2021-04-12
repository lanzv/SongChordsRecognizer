#!/usr/bin/env python3
from numpy.core.records import array
from ACR_Training.Annotations import ChordSequence, KeySequence, SongDescription
from ACR_Training.annotation_maps import chords_map, keys_map, key_modes_map
from ACR_Training.Audio import Audio, BillboardFeatures
import os
from glob import glob
import pickle
import lzma
import librosa
import numpy as np
import re
from ACR_Training.Spectrograms import log_mel_spectrogram
from ACR_Training.SegmentationModels import SegmentationCRNN

class Dataset():
    def __init__(self, audio_directory=None, annotations_directory=None):
        self.DATA = []
        self.CHORDS = []


    @staticmethod
    def get_integered_chord(chord, norm_to_C=False, key='C') -> int:
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


    @staticmethod
    def songs_to_sequences(FEATURESs, CHORDs, TIME_BINSs, KEYs, n_frames=500, norm_to_C=False) -> tuple:
        """
        Preprocess dataset.
        Divide features from FEATURESs to n_frames frames long sequences and do the same with targets from self.CHORDS.

        Parameters
        ----------
        FEATURESs : list of lists of float
            list of song features we want to separate to sequences about n_frames elements
        CHORDs : list of Chord objects
            list of song chords
        TIME_BINSs :
            time points mapped with features
        KEYs : list of strings
            song's key
        n_frames : int
             how many frames should be included in a subsequence of a song
        Returns
        -------
        prep_data : np array
            sequences of song's chroma vectors separated to n_frames frames
        prep_targets : np array
            sequences of integers of chord labels for specific chroma vector sequences
        """
        prep_data = []
        prep_targets = []
        for features, chords, time_bins, key in zip(FEATURESs, CHORDs, TIME_BINSs, KEYs):
            j = 0
            for i in range((int)(features.shape[0]/n_frames)):
                # Get chroma
                prep_data.append(features[i*n_frames:(i+1)*n_frames])
                # Get labels
                prep_targets.append([])
                for chord_ind in range(i*n_frames, (i+1)*n_frames):
                    second = time_bins[chord_ind]
                    while j < len(chords.START) and second > chords.START[j] :
                        j = j + 1

                    if j == len(chords.START):
                        prep_targets[-1].append(Dataset.get_integered_chord("N", norm_to_C, key))
                    else:
                        prep_targets[-1].append(Dataset.get_integered_chord(chords.CHORD[j], norm_to_C, key))

            # Embed zero chromas to fill n_frames frames
            last_ind = (int)(features.shape[0]/n_frames)
            _, n_features = features.shape
            prep_data.append(
                np.concatenate((
                    np.array(features[last_ind*n_frames:]),
                    np.zeros((n_frames - (len(features) - last_ind*n_frames), n_features))
                ), axis=0 )
            )
            # Embed N chords to fill n_frames frames
            prep_targets.append([])
            for chord_ind in range(last_ind*n_frames, (last_ind+1) * n_frames):
                if chord_ind < len(time_bins):
                    second = time_bins[chord_ind]
                    while j < len(chords.START) and second > chords.START[j] :
                        j = j + 1

                    if j == len(chords.START):
                        prep_targets[-1].append(Dataset.get_integered_chord("N", norm_to_C, key))
                    else:
                        prep_targets[-1].append(Dataset.get_integered_chord(chords.CHORD[j], norm_to_C, key))
                else:
                    prep_targets[-1].append(Dataset.get_integered_chord("N", norm_to_C, key))

        print("[INFO] The Dataset was successfully preprocessed.")
        return np.array(prep_data), np.array(prep_targets)





class IsophonicsDataset(Dataset):
    """Isophonics Dataset.
    The train set contains 225 Bealtes, Queen, Carole King or Zweieck songs. 
    DATA contains audio waveform features.
    CHORDS contains chord annotation for all song duration.
    KEYS contains key annotation for all song duration.
    """
    def __init__(self, audio_directory=None, annotations_directory=None, sample_rate=44100, nfft=2**14):
        
        self.SAMPLE_RATE = sample_rate
        self.NFFT = nfft

        self.DATA = []
        self.CHORDS = []
        self.KEYS =  []

        if (not audio_directory == None) and (not annotations_directory == None):

            audio_paths = sorted(glob(os.path.join(audio_directory, '*.wav')))
            chord_annotations_paths = sorted(glob(os.path.join(annotations_directory+'/CHORDS', '*.lab')))
            key_annotations_paths = sorted(glob(os.path.join(annotations_directory+'/KEYS', '*.lab')))

            if not (len(audio_paths) == len(chord_annotations_paths) and len(audio_paths) == len(key_annotations_paths)):
                raise Exception("The number of WAV files doesn't equal the number of annotation files.")

            for audio_path, chord_lab_path, key_lab_path in zip(audio_paths, chord_annotations_paths, key_annotations_paths):
                self.DATA.append(Audio(audio_path, self.SAMPLE_RATE))
                self.CHORDS.append(ChordSequence(chord_lab_path))
                self.KEYS.append(KeySequence(key_lab_path))

            print("[INFO] The Isophonics Dataset was successfully initialized.")
        else:
            print("[INFO] The Isophonics Dataset was successfully initialized without any data or annotations.")
    

    def get_preprocessed_dataset(self, hop_length=512, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, n_frames=500) -> tuple:
        """
        Preprocess Isophonics dataset.
        Divide spectrogram features geenrated from self.DATA audio waveforms to n_frames frames long sequences and do the same with targets from self.CHORDS.

        Parameters
        ----------
        hop_length : int
            number of samples between successive spectrogram columns
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        n_frames : int
             how many frames should be included in a subsequence of a song
        Returns
        -------
        prep_data : np array
            sequences of song's spectrogram vectors separated to n_frames frames
        prep_targets : np array
            sequences of integers of chord labels for specific spectrogram vector sequences
        """
        FEATURESs = []
        CHORDs = self.CHORDS
        TIME_BINSs = []
        KEYs = []
        for audio, keys in zip(self.DATA, self.KEYS):
            FEATURESs.append((IsophonicsDataset.preprocess_audio(waveform=audio.WAVEFORM, sample_rate=audio.SAMPLE_RATE, spectrogram_generator=spectrogram_generator, nfft=self.NFFT, hop_length=hop_length, norm_to_C=norm_to_C, key=keys.get_first_key()).swapaxes(0,1)))
            num_samples, _ = FEATURESs[-1].shape
            TIME_BINSs.append([float(i)/(float(self.SAMPLE_RATE) / float(hop_length)) for i in range(num_samples)])
            KEYs.append(keys.get_first_key())

        return Dataset.songs_to_sequences(FEATURESs=FEATURESs, CHORDs=CHORDs, TIME_BINSs=TIME_BINSs, KEYs=KEYs, n_frames=n_frames, norm_to_C=norm_to_C)


       
    def preprocess_single_chords_list(self, window_size=5, flattened_window=True, hop_length=4410, to_skip=5, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, skip_coef=1) -> tuple:
        """
        Preprocess IsophonicsDataset dataset.
        Create features from self.DATA and its corresponding targets from self.CHORDS.
        
        Parameters
        ----------
        window_size : int
            how many spectrograms on left and on right we should take 
        flattened_window : bool
            True if we want to flatten spectrograms to one array, otherwise False
        hop_length : int
            number of samples between successive spectrogram columns
        to_skip : int
            how many spectrogram we want to skip when creating new feature set
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        skip_coef : int
            coeficient that multiplies window shifts -> some spectrogram are skipped in the flattened window
        Returns
        -------
        prep_data : np array
            window (flettend or not) of logarithmized mel spectrograms arround specific time point
        prep_targets : np array
            integers of chord labels for specific time point
        """
        prep_data = []
        prep_targets = []
        k = 0
        # Iterate over all audio files
        for audio, chords, keys in zip(self.DATA, self.CHORDS, self.KEYS):
            print(k)
            k = k+1
            # Get log mel spectrogram
            spectrogram = IsophonicsDataset.preprocess_audio(waveform=audio.WAVEFORM, sample_rate=audio.SAMPLE_RATE, spectrogram_generator=spectrogram_generator, nfft=self.NFFT, hop_length=hop_length, norm_to_C=norm_to_C, key=keys.get_first_key())
            spectrogram = np.array(spectrogram)
            spec_length, num_samples = spectrogram.shape

            # Collect data for each spectrogram sample
            j = 0 # labels index
            for i in [index for index in range(num_samples) if index%to_skip==0]:
                # Get data window with zero margin
                n_pre_zeros, window_indices, n_post_zeros = IsophonicsDataset.__get_flatten_indices(i, num_samples, skip_coef, window_size)
                if flattened_window:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((n_pre_zeros, spec_length)),
                            np.array(spectrogram[:, window_indices]).swapaxes(0,1),
                            np.zeros((n_post_zeros, spec_length))
                        ), axis = 0).flatten()
                    )
                else:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((n_pre_zeros, spec_length)),
                            np.array(spectrogram[:, window_indices]).swapaxes(0,1),
                            np.zeros((n_post_zeros, spec_length))
                        ), axis = 0)
                    )


                # Get label
                second = float(i)/(float(self.SAMPLE_RATE) / float(hop_length))
                while j < len(chords.START) and second > chords.START[j] :
                    j = j + 1
                if j == len(chords.START):
                    prep_targets.append(Dataset.get_integered_chord("N", norm_to_C, keys.get_first_key()))
                else:
                    prep_targets.append(Dataset.get_integered_chord(chords.CHORD[j], norm_to_C, keys.get_first_key()))

        print("[INFO] The Isophonics Dataset was successfully preprocessed.")
        return np.array(prep_data), np.array(prep_targets)


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
                raise Exception("Isophonics __get_flatten_indices faced to unexptected situation.")

        return n_pre_zeros, window_indices, n_post_zeros



    @staticmethod
    def preprocess_audio(waveform, sample_rate, spectrogram_generator, nfft, hop_length, norm_to_C=False, key='C') -> list:
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
            number of samples between successive spectrogram columns
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



    def save_preprocessed_dataset(self, dest = "./Datasets/preprocessed_IsophonicsDataset.ds", hop_length=512, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, n_frames=500):
        """
        Save preprocessed data from this dataset to destination path 'dest' by default as a .ds file.
        
        Parameters
        ----------
        dest : str
            path to preprocessed data
        hop_length : int
            number of samples between successive spectrogram columns
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        n_frames : int
             how many frames should be included in a subsequence of a song
        """
        # Serialize the dataset.
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((self.get_preprocessed_dataset(hop_length=hop_length, norm_to_C=norm_to_C, spectrogram_generator=spectrogram_generator, n_frames=n_frames)), dataset_file)

        print("[INFO] The Preprocessed Isophonics Dataset was saved successfully.")



    @staticmethod
    def load_preprocessed_dataset(dest = "./Datasets/preprocessed_IsophonicsDataset.ds") -> tuple:
        """
        Load preprocessed data from this dataset from destination path 'dest'. Targets and preprocessed Data are stored by default as a .ds file.
        
        Parameters
        ----------
        dest : str
            path to preprocessed data
        Returns
        -------
        prep_data : np array
            sequences of song's spectrogram vectors separated to n_frames frames
        prep_targets : np array
            sequences of integers of chord labels for specific spectrogram vector sequences
        """
        with lzma.open(dest, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        print("[INFO] The Preprocessed Isophonics Dataset was loaded successfully.")
        return dataset



    def save_dataset(self, dest = "./Datasets/IsophonicsDataset.ds"):
        """
        Save data from this dataset to destination path 'dest' by default as a .ds file.
        
        Parameters
        ----------
        dest : str
            path to dataset
        """
        # Serialize the dataset.
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((self.DATA, self.CHORDS, self.KEYS, self.SAMPLE_RATE, self.NFFT), dataset_file)

        print("[INFO] The Isophonics Dataset was saved successfully.")


           
    @staticmethod
    def load_dataset(dest = "./Datasets/IsophonicsDataset.ds") -> 'IsophonicsDataset':
        """
        Load data from this dataset from destination path 'dest'. Targets and Data are stored by default as a .ds file.
        
        Parameters
        ----------
        dest : str
            path to data
        Returns
        -------
        dataset : IsophonicsDataset object
            dataset containing DATA, CHORDS and KEYS loaded from a file
        """
        with lzma.open(dest, "rb") as dataset_file:
            loaded_dataset = pickle.load(dataset_file)

        data, chords, keys, sample_rate, nfft  = loaded_dataset

        dataset = IsophonicsDataset()

        dataset.DATA = data
        dataset.CHORDS = chords
        dataset.KEYS = keys
        dataset.SAMPLE_RATE = sample_rate
        dataset.NFFT = nfft

        print("[INFO] The Isophonics Dataset was loaded successfully.")
        return dataset



    def save_segmentation_samples(self, dest="./Datasets/IsophonicsSegmentation.seg", song_indices=[0, 10, 20, 30, 40, 50, 60, 70], hop_length=512, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, n_frames=500):
        """
        Save preprocessed data from this dataset with its target and chord changes to destination path 'dest' by default as a .seg file.

        Parameters
        ----------
        dest : str
            path to preprocessed data
        song_indices : list of integers
            list of song indices as representing samples
        hop_length : int
            number of samples between successive spectrogram columns
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        n_frames : int
             how many frames should be included in a subsequence of a song
        """
        data = []
        chords = []
        gold_targets = []
        # Iterate over all song indices on the input
        for song_ind in song_indices:
            # Prprocess audio
            preprocessed_audio = IsophonicsDataset.preprocess_audio(
                waveform=self.DATA[song_ind].WAVEFORM,
                sample_rate=self.DATA[song_ind].SAMPLE_RATE,
                spectrogram_generator=spectrogram_generator,
                nfft=self.NFFT, hop_length=hop_length,
                norm_to_C=norm_to_C, key=self.KEYS[song_ind].get_first_key()
            ).swapaxes(0,1)

            num_samples, _ = preprocessed_audio.shape

            # Convert data and chord targets to sequences
            data_in_seqs, targets_in_seqs = Dataset.songs_to_sequences(
                FEATURESs=[preprocessed_audio],
                CHORDs=[self.CHORDS[song_ind]],
                TIME_BINSs=[[float(i)/(float(self.SAMPLE_RATE) / float(hop_length)) for i in range(num_samples)]],
                KEYs=self.KEYS[song_ind].get_first_key(),
                n_frames=n_frames,
                norm_to_C=norm_to_C
            )

            # Add song's sequences to lists as a new element
            data.append(data_in_seqs)
            chords.append(targets_in_seqs)
            gold_targets.append(SegmentationCRNN.labels2changes(targets = chords[-1]))

        # Save all three np arrays generated in this function .. data, chords, gold_targets aka chord changes
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((data, chords, gold_targets), dataset_file)

        print("[INFO] The Isophonics segmentation samples was saved successfully.")



    @staticmethod
    def load_segmentation_samples(dest = "./Datasets/IsophonicsSegmentation.seg") -> tuple:
        """
        Load preprocessed data and targets with its chord changes points from destination path 'dest'. This kind of data are stored by default as a .seg file.

        Parameters
        ----------
        dest : str
            path to segmentation samples
        Returns
        -------
        prep_data : np array
            sequences of song's spectrogram vectors separated to n_frames frames
        prep_chords : np array
            array of n_frame long sequences of integers of chord labels
        prep_chord_changes : np array
            array of n_frame long sequences of 0s and 1s where 0 means 'not chord change' and 1 means 'chord change'
        """
        with lzma.open(dest, "rb") as segmentation_samles:
            loaded_samples = pickle.load(segmentation_samles)

        print("[INFO] The Isophonics segmentation samples was loaded successfully.")
        return loaded_samples




class BillboardDataset(Dataset):
    """Billboard Dataset.
    The train set contains 890 a representative samples of American popular music from the 1950s through the 1990s. 
    DATA contains audio waveform features.
    CHORDS contains a chord sequence.
    DESC contains a audio description like tonic, metre, ect...
    """
    def __init__(self, audio_directory=None, annotations_directory=None, sample_rate=44100, nfft=2**14, audio=False):

        self.SAMPLE_RATE = sample_rate
        self.NFFT = nfft

        self.DATA = []
        self.CHORDS = []
        self.DESC =  []

        if (not audio_directory == None) and (not annotations_directory == None):
            audio_paths = sorted(glob(os.path.join(audio_directory, 'CHORDINO/*/'))) if not audio else sorted(glob(os.path.join(audio_directory, 'WAV/*/*.wav')))
            chord_annotations_paths = sorted(glob(os.path.join(annotations_directory, 'LABs/*/')))
            desc_annotations_paths = sorted(glob(os.path.join(annotations_directory, 'DESCRIPTIONs/*/')))

            if not (len(audio_paths) == len(chord_annotations_paths) and len(audio_paths) == len(desc_annotations_paths)):
                raise Exception("The number of WAV files doesn't equal the number of annotation files.")

            for audio_path, chord_lab_path, desc_path in zip(audio_paths, chord_annotations_paths, desc_annotations_paths):
                if audio:
                    self.DATA.append(Audio(audio_path, self.SAMPLE_RATE))
                else:
                    self.DATA.append(BillboardFeatures(audio_path))
                self.CHORDS.append(ChordSequence(chord_lab_path+"full.lab"))
                self.DESC.append(SongDescription(desc_path+"salami_chords.txt"))


            self.DATA = np.array(self.DATA)
            self.CHORDS = np.array(self.CHORDS)
            self.DESC = np.array(self.DESC)
            print("[INFO] The Billboard Dataset was successfully initialized.")
        else:
            print("[INFO] The Billboard Dataset was successfulyy initialized without any data or annotations.")




    def get_preprocessed_dataset(self, hop_length=512, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, n_frames=500) -> tuple:
        """
        Preprocess Billboard dataset.
        Divide spectrogram features generated from self.DATA audio waveforms to n_frames frames long sequences and do the same with targets from self.CHORDS.

        Parameters
        ----------
        hop_length : int
            number of samples between successive spectrogram columns
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        n_frames : int
             how many frames should be included in a subsequence of a song
        Returns
        -------
        prep_data : np array
            sequences of song's spectrogram vectors separated to n_frames frames
        prep_targets : np array
            sequences of integers of chord labels for specific spectrogram vector sequences
        """
        FEATURESs = []
        CHORDs = self.CHORDS
        TIME_BINSs = []
        KEYs = []
        norm_to_C = False
        k = 0
        for data, desc in zip(self.DATA, self.DESC):
            print(k)
            if isinstance(data, BillboardFeatures):
                FEATURESs.append(data.CHROMA)
                TIME_BINSs.append(data.TIME_BINS)
            elif isinstance(data, Audio):
                FEATURESs.append((IsophonicsDataset.preprocess_audio(waveform=data.WAVEFORM, sample_rate=data.SAMPLE_RATE, spectrogram_generator=spectrogram_generator, nfft=self.NFFT, hop_length=hop_length, norm_to_C=norm_to_C, key=desc.TONIC).swapaxes(0,1)))
                num_samples, _ = FEATURESs[-1].shape
                TIME_BINSs.append([float(i)/(float(self.SAMPLE_RATE) / float(hop_length)) for i in range(num_samples)])
            k = k + 1
            KEYs.append(desc.TONIC)

        return Dataset.songs_to_sequences(FEATURESs=FEATURESs, CHORDs=CHORDs, TIME_BINSs=TIME_BINSs, KEYs=KEYs, n_frames=n_frames, norm_to_C=norm_to_C)



    def preprocess_single_chords_list(self, window_size=5, flattened_window=True, hop_length=4410, to_skip=5, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, skip_coef=1) -> tuple:
        """
        Preprocess Billboard dataset.
        Create features from self.DATA and its corresponding targets from self.CHORDS.

        Parameters
        ----------
        window_size : int
            how many spectrograms on left and on right we should take
        flattened_window : bool
            True if we want to flatten spectrograms to one array, otherwise False
        hop_length : int
            number of samples between successive spectrogram columns
        to_skip : int
            how many spectrogram we want to skip when creating new feature set
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        skip_coef : int
            coeficient that multiplies window shifts -> some spectrogram are skipped in the flattened window
        Returns
        -------
        prep_data : np array
            window (flettend or not) of logarithmized mel spectrograms arround specific time point
        prep_targets : np array
            integers of chord labels for specific time point
        """
        prep_data = []
        prep_targets = []
        k = 0
        # Iterate over all audio files
        for audio, chords, desc in zip(self.DATA, self.CHORDS, self.DESC):
            print(k)
            k = k+1
            # Get log mel spectrogram
            spectrogram = IsophonicsDataset.preprocess_audio(waveform=audio.WAVEFORM, sample_rate=audio.SAMPLE_RATE, spectrogram_generator=spectrogram_generator, nfft=self.NFFT, hop_length=hop_length, norm_to_C=norm_to_C, key=desc.TONIC)
            spectrogram = np.array(spectrogram)
            spec_length, num_samples = spectrogram.shape

            # Collect data for each spectrogram sample
            j = 0 # labels index
            for i in [index for index in range(num_samples) if index%to_skip==0]:
                # Get data window with zero margin
                n_pre_zeros, window_indices, n_post_zeros = IsophonicsDataset.__get_flatten_indices(i, num_samples, skip_coef, window_size)
                if flattened_window:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((n_pre_zeros, spec_length)),
                            np.array(spectrogram[:, window_indices]).swapaxes(0,1),
                            np.zeros((n_post_zeros, spec_length))
                        ), axis = 0).flatten()
                    )
                else:
                    prep_data.append(
                        np.concatenate((
                            np.zeros((n_pre_zeros, spec_length)),
                            np.array(spectrogram[:, window_indices]).swapaxes(0,1),
                            np.zeros((n_post_zeros, spec_length))
                        ), axis = 0)
                    )


                # Get label
                second = float(i)/(float(self.SAMPLE_RATE) / float(hop_length))
                while j < len(chords.START) and second > chords.START[j] :
                    j = j + 1
                if j == len(chords.START):
                    prep_targets.append(Dataset.get_integered_chord("N", norm_to_C, desc.TONIC))
                else:
                    prep_targets.append(Dataset.get_integered_chord(chords.CHORD[j], norm_to_C, desc.TONIC))

        print("[INFO] The Billboard Dataset was successfully preprocessed.")
        return np.array(prep_data), np.array(prep_targets)



    def save_preprocessed_dataset(self, dest = "./Datasets/preprocessed_BillboardDataset.ds", hop_length=512, norm_to_C=False, spectrogram_generator=log_mel_spectrogram, n_frames=500):
        """
        Save preprocessed data from this dataset to destination path 'dest' by default as a .ds file.

        Parameters
        ----------
        dest : str
            path to preprocessed data
        hop_length : int
            number of samples between successive spectrogram columns
        norm_to_C : bool
            True if we want to transpose all songs to C key
        spectrogram_generator : method from Spectrograms.py
            function that generates spectrogram
        n_frames : int
             how many frames should be included in a subsequence of a song
        """

        # Serialize the dataset.
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((self.get_preprocessed_dataset(hop_length=hop_length, norm_to_C=norm_to_C, spectrogram_generator=spectrogram_generator, n_frames=n_frames)), dataset_file)

        print("[INFO] The Preprocessed Billboard Dataset was saved successfully.")



    @staticmethod
    def load_preprocessed_dataset(dest = "./Datasets/preprocessed_BillboardDataset.ds") -> tuple:
        """
        Load preprocessed data from this dataset from destination path 'dest'. Targets and preprocessed Data are stored by default as a .ds file.

        Parameters
        ----------
        dest : str
            path to preprocessed data
        Returns
        -------
        prep_data : np array
            sequences of song's chroma vectors separated to n_frames frames
        prep_targets : np array
            sequences of integers of chord labels for specific chroma vector sequences
        """
        with lzma.open(dest, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        print("[INFO] The Preprocessed Billboard Dataset was loaded successfully.")
        return dataset




    def save_segmentation_samples(self, dest="./Datasets/BillboardSegmentation.seg", song_indices=[0, 10, 20, 30, 40, 50, 60, 70], n_frames=500):
        """
        Save preprocessed data from this dataset with its target and chord changes to destination path 'dest' by default as a .seg file.

        Parameters
        ----------
        dest : str
            path to preprocessed data
        song_indices : list of integers
            list of song indices as representing samples
        norm_to_C : bool
            True if we want to transpose all songs to C key
        n_frames : int
             how many frames should be included in a subsequence of a song
        """
        data = []
        chords = []
        gold_targets = []
        # Iterate over all song indices on the input
        for song_ind in song_indices:

            # Convert data and chord targets to sequences
            data_in_seqs, targets_in_seqs = Dataset.songs_to_sequences(
                FEATURESs=[self.DATA[song_ind].CHROMA],
                CHORDs=[self.CHORDS[song_ind]],
                TIME_BINSs=[self.DATA[song_ind].TIME_BINS],
                KEYs=self.DESC[song_ind].TONIC,
                n_frames=n_frames,
                norm_to_C=False
            )

            # Add song's sequences to lists as a new element
            data.append(data_in_seqs)
            chords.append(targets_in_seqs)
            gold_targets.append(SegmentationCRNN.labels2changes(targets = chords[-1]))

        # Save all three np arrays generated in this function .. data, chords, gold_targets aka chord changes
        with lzma.open(dest, "wb") as dataset_file:
            pickle.dump((data, chords, gold_targets), dataset_file)

        print("[INFO] The Billboard segmentation samples was saved successfully.")



    @staticmethod
    def load_segmentation_samples(dest = "./Datasets/BillboardSegmentation.seg") -> tuple:
        """
        Load preprocessed data and targets with its chord changes points from destination path 'dest'. This kind of data are stored by default as a .seg file.

        Parameters
        ----------
        dest : str
            path to segmentation samples
        Returns
        -------
        prep_data : np array
            sequences of song's chroma vectors separated to n_frames frames
        prep_chords : np array
            array of n_frame long sequences of integers of chord labels
        prep_chord_changes : np array
            array of n_frame long sequences of 0s and 1s where 0 means 'not chord change' and 1 means 'chord change'
        """
        with lzma.open(dest, "rb") as segmentation_samles:
            loaded_samples = pickle.load(segmentation_samles)

        print("[INFO] The Billboard segmentation samples was loaded successfully.")
        return loaded_samples

    
