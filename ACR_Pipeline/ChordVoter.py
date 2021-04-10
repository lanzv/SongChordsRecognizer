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
        #bpm, beats = librosa.beat.beat_track(y=waveform, sr=sample_rate, hop_length=hop_length)
        #voted_chords = ChordVoter._beat_chord_bpm_estimation(beats, chord_sequence)
        bpm, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate, hop_length=hop_length)

        voted_chords, beats = ChordVoter._beat_chord_harmony_estimation(
            ChordVoter._get_n_of_beat_elements(bpm, sample_rate, hop_length),
            ChordVoter._encode_sequence_to_counts(chord_sequence)
        )

        beat_times = librosa.frames_to_time(beats, sr=sample_rate, hop_length=hop_length)

        return voted_chords, bpm, beat_times

    @staticmethod
    def _encode_sequence_to_counts(sequence):
        """
        The function will convert the chord sequence [1, 2, 2, 2, 2, 1, 1, 2, 6, 6 ]
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

    @staticmethod
    def _beat_chord_harmony_estimation(one_beat_elements, count_encoded_sequence):
        """
        The function will take the length of same chord subsequence and will estimate how many beats could be that.
        The number of beats coressponding to the chord duration is the number how many times the chord is added to 
        the final result list of chords .. chord_beats.

        Parameters
        ----------
        one_beat_elements : int
            how many sequence elements coressponds to one beat by the simple BPM estimation
        count_encoded_sequence : tuple list
            list of (chord, count) tuples that encoded the chord and its duration length
        Returns
        -------
        chord_beats : int list
            sequence of chords mapped to each beat
        beats : int list
            beat points
        """
        chord_beats = []
        beats = []
        counts = 0
        for (chord, count) in count_encoded_sequence:
            for i in range(round(count/one_beat_elements)):
                chord_beats.append(chord)
                beats.append(counts + i * one_beat_elements)
            counts = counts + count

        return chord_beats, beats

    @staticmethod
    def _beat_chord_bpm_estimation(beats, chord_sequence):
        """
        The function will consider all chords during two beats and will pick the most frequent one.

        Parameters
        ----------
        beats : int list
            list of beat indices of song
        chord_sequence : int list
            list of chord indeces played in the song
        Returns
        -------
        chord_beats : int list
            sequence of chords mapped to each beat
        """
        chord_beats = []

        for i in range(len(beats)):
            if i+1 < len(beats):
                chord_beats.append(
                    np.bincount(chord_sequence[beats[i]:beats[i+1]]).argmax()
                )
            else:
                chord_beats.append(
                    np.bincount(chord_sequence[beats[i]:-1]).argmax()
                )

        return chord_beats

    @staticmethod
    def _get_n_of_beat_elements(bpm, sample_rate, hop_length):
        """
        The function will compute the number of elements of one beat.

        Parameters
        ----------
        bpm : int
            beats per minute value
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        Returns
        -------
        n_beat_elements : int
            the number of elements of one beat
        """
        # Get sample time ~ nth_element
        nth_sample_element = 3000
        times = librosa.frames_to_time([nth_sample_element], sr=sample_rate, hop_length=hop_length)

        # How many elements are in one minute
        n_minute_elements =  nth_sample_element / (times[0] / 60)

        # How many elements are in one beat
        n_beat_elements = n_minute_elements/bpm

        return n_beat_elements
