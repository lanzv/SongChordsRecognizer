#!/usr/bin/env python3
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Datasets import BillboardDataset, IsophonicsDataset
from ACR_Training.Spectrograms import cqt_spectrogram, log_mel_spectrogram, cqt_chromagram, stft_chromagram
import pickle
import lzma
import mir_eval
import librosa
import numpy as np
from ACR_Pipeline.DataPreprocessor import DataPreprocessor


class Evaluator():
    """
    """
    @staticmethod
    def mir_eval_score_from_labs(gold_lab, predicted_lab):
        """
        """
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gold_lab)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(predicted_lab)
        return Evaluator.__mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels)

    @staticmethod
    def mir_eval_score_from_sequences(gold_lab, predicted_chords, sample_rate=22050, hop_length=512):
        """
        """
        ref_intervals, ref_labels = mir_eval.io.load_labeled_intervals(gold_lab)
        est_intervals, est_labels = Evaluator._get_label_intervals(chord_sequence=predicted_chords, sample_rate=sample_rate, hop_length=hop_length)
        return Evaluator.__mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels)

    @staticmethod
    def _get_label_intervals(chord_sequence, sample_rate=22050, hop_length=512):
        """
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
    def __mir_eval_score(ref_intervals, ref_labels,est_intervals, est_labels):
        """
        """
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(), mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels, est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)

        return score


def main():
    a = [0, 0, 0, 1, 5, 5, 5, 5, 5, 5, 5, 7, 1, 2, 2, 2, 2, 2, 4, 4, 4]
    print(Evaluator.mir_eval_score_from_sequences("./a.lab", a))

if __name__ == "__main__":

    main()