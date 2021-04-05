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