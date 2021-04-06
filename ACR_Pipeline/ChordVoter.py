import librosa
import numpy as np

class ChordVoter():

    @staticmethod
    def vote_for_beats(chord_sequence, waveform, sample_rate=44100, hop_length=512):
        """
        The function will find beats of the song and for each one will choose the most frequent chord predicted during two adjacent beats.

        Parameters
        ----------
        chord_sequence : int array
            sequence of predicted chords with the same sample rate and hop length
        waveform : list of floats
            data of audio waveform
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        Returns
        -------
        voted_chords : int array
            sequence of chords corresponding to each beat
        bpm : int
            beats per minute value
        beat_times : float array
            list of time points in seconds of beats
        """
        bpm, beats = librosa.beat.beat_track(y=waveform, sr=sample_rate, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate, hop_length=hop_length)


        voted_chords = []

        for i in range(len(beats)):
            if i+1 < len(beats):
                voted_chords.append(
                    np.bincount(chord_sequence[beats[i]:beats[i+1]]).argmax()
                )
            else:
                voted_chords.append(
                    np.bincount(chord_sequence[beats[i]:-1]).argmax()
                )

        return voted_chords, bpm, beat_times

    @staticmethod
    def _encode_sequence_to_counts(sequence):
        """
        The function will covert the chord sequence [1, 2, 2, 2, 2, 1, 1, 2, 6, 6 ]
        To the sequence of counts [(1, 1), (2, 4), (1, 2), (1, 6)]

        Parameters
        ----------
        sequence : int list
            list of chord indeces
        Returns
        -------
        counts : list of tuples (chord index, count)
            sequence of chords without repeating but with its count
        """
        counts = []

        acutal_element = -1
        counter = 0
        for i in sequence:
            if acutal_element == i:
                counter = counter + 1
            elif counter != 0:
                counts.append((acutal_element, counter))
                counter = 1
                acutal_element = i
            else:
                counter = 1
                acutal_element = i
        if counter != 0:
            counts.append((acutal_element, counter))

        return counts