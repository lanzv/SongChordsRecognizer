#!/usr/bin/env python3
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Datasets import BillboardDataset, IsophonicsDataset
from ACR_Training.Spectrograms import cqt_spectrogram, log_mel_spectrogram, cqt_chromagram, stft_chromagram

from glob import glob
import mir_eval
import librosa
import numpy as np
from ACR_Pipeline.DataPreprocessor import DataPreprocessor


class Evaluator():
    """
    Evlauator class that provides static method to evaluate chords using mir eval library.
    """

    @staticmethod
    def eval_isophonics_testset(gold, predictions, sample_rate=22050, hop_length=512):
        """
        The function will evaluate weighted accuracy score of lab files and predictions of billboard test dataset, using mir eval library.

        Parameters
        ----------
        gold : str or int list list
            list of songs and their song sequences or path of the annotations directory containing .lab files
        predictions : int list list
            list of songs and their song sequences
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        Returns
        -------
        score : float
            mir_eval score, weighted accuracy
        """
        durations = np.empty(1)
        comparisons = np.empty(1)

        # Get list of lab files if gold is path
        if isinstance(gold, str):
            gold_files = sorted(glob(os.path.join(gold+'/CHORDS', '*.lab')))
        else:
            gold_files = gold

        # Collect durations and comparisons
        for song_gold, chord_sequence in zip(gold_files, predictions):
            n_sequences,n_frames,n_features = chord_sequence.shape
            chord_sequence = np.array(chord_sequence.reshape((-1,n_features))).argmax(axis=1)
            song_gold = np.array(song_gold.reshape(-1))
            if isinstance(gold, str):
                ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(song_gold)
            else:
                ref_intervals, ref_labels = Evaluator._get_label_intervals(chord_sequence=song_gold, sample_rate=sample_rate, hop_length=hop_length)
            est_intervals, est_labels = Evaluator._get_label_intervals(chord_sequence=chord_sequence, sample_rate=sample_rate, hop_length=hop_length)
            est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
            (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
            durations = np.insert(durations, -1, mir_eval.util.intervals_to_durations(intervals))
            comparisons = np.insert(comparisons, -1 , mir_eval.chord.triads(ref_labels, est_labels))

        # Get Score
        durations = durations[:-1]
        comparisons = comparisons[:-1]
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)

        return score



    @staticmethod
    def eval_billboard_testset(annotations_directory, predictions, sample_rate=22050, hop_length=512):
        """
        The function will evaluate weighted accuracy score of lab files and predictions of billboard test dataset, using mir eval library.

        Parameters
        ----------
        annotations_directory : str
            path of the annotations directory with LABs folder containing /*/full.lab files
        predictions : int list list
            list of songs and their song sequences
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        Returns
        -------
        score : float
            mir_eval score, weighted accuracy
        """
        durations = np.empty(1)
        comparisons = np.empty(1)

        # Get list of lab files
        lab_files = sorted(glob(os.path.join(annotations_directory, 'LABs/*/')))

        # Collect durations and comparisons
        for lab, chord_sequence in zip(lab_files, predictions):
            n_sequences,n_frames,n_features = chord_sequence.shape
            chord_sequence = np.array(chord_sequence.reshape((n_sequences*n_frames,n_features))).argmax(axis=1)
            ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(lab+"full.lab")
            est_intervals, est_labels = Evaluator._get_label_intervals(chord_sequence=chord_sequence, sample_rate=sample_rate, hop_length=hop_length)
            est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
            (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
            durations = np.insert(durations, -1, mir_eval.util.intervals_to_durations(intervals))
            comparisons = np.insert(comparisons, -1 , mir_eval.chord.triads(ref_labels, est_labels))

        # Get Score
        durations = durations[:-1]
        comparisons = comparisons[:-1]
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)

        return score

    @staticmethod
    def mir_eval_score_from_labs(gold_lab, predicted_lab):
        """
        The function will evaluate weighted accuracy score of two lab files, the gold one and the predicted one, using mir eval library.

        Parameters
        ----------
        gold_lab : str
            path of the .LAB file that contains correct chords of the song
        predicted_lab : str
            path of the .LAB file that contains predicted chords of the song
        Returns
        -------
        score : float
            mir_eval score, weighted accuracy
        """
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gold_lab)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(predicted_lab)
        return Evaluator.__mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels)

    @staticmethod
    def mir_eval_score_from_sequences(gold_lab, predicted_chords, sample_rate=22050, hop_length=512):
        """
        The function will evaluate weighted accuracy score of the gold lab file and predicted chord sequence, using mir eval library.
        
        Parameters
        ----------
        gold_lab : str
            path of the .LAB file that contains correct chords of the song
        predicted_chords : int list
            list of predicted chord indices
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        Returns
        -------
        score : float
            mir_eval score, weighted accuracy
        """
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(gold_lab)
        est_intervals, est_labels = Evaluator._get_label_intervals(chord_sequence=predicted_chords, sample_rate=sample_rate, hop_length=hop_length)
        return Evaluator.__mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels)

    @staticmethod
    def _get_label_intervals(chord_sequence, sample_rate=22050, hop_length=512):
        """
        The function will parse the chord sequence and create intervals and labels corresponding to the mir_eval format.
        
        Parameters
        ----------
        chord_sequence : int list
            list of chord indices
        sample_rate : int
            audio sample rate
        hop_length : int
            number of samples between successive spectrogram columns
        Returns
        -------
        intervals : np array
            array of time intervals of chords in the song
        labels : str list
            list of chords mapped to those intervals
        """
        times = librosa.frames_to_time([i for i in range(len(chord_sequence))],
            sr=sample_rate, hop_length=hop_length)
        intervals = []
        labels = []
        last_chord = -1
        for time, chord in zip(times, chord_sequence):
            if last_chord != chord and not last_chord == -1:
                intervals.append([left_border,time])
                labels.append(chord)
                last_chord = chord
                left_border = time
            elif last_chord == -1:
                left_border = 0
                last_chord = chord

        return np.array(intervals), DataPreprocessor.chord_indices_to_notations(labels)

    @staticmethod
    def __mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels):
        """
        The function will compute the score for already prepared gold and predicted intervals and labels.

        Parameters
        ----------
        ref_intervals : np array
            array of time intervals of gold in the song
        ref_labels : str list
            list of gold chords mapped to ref intervals
        est_intervals : np array
            array of time intervals of predicted chords in the song
        est_labels : str list
            list of predicted chords mapped to est intervals
        Returns
        -------
        score : float
            mir_eval score, weighted accuracy
        """
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)

        return score
