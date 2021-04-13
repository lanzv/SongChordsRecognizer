#!/usr/bin/env python3
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Datasets import BillboardDataset, IsophonicsDataset
from ACR_Training.Spectrograms import cqt_spectrogram, log_mel_spectrogram, cqt_chromagram, stft_chromagram
import pickle
import lzma
import mir_eval


class Evaluator():
    """
    """
    @staticmethod
    def mir_eval_score_from_labs(gold_chords, prediction_chords, x, y):
        """
        """
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(x)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(y)
        return Evaluator.__mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels)

    @staticmethod
    def mir_eval_score_from_sequences(gold_chords, predicted_chords):
        """
        """
        ref_intervals, ref_labels = Evaluator. _get_label_intervals(gold_chords)
        est_intervals, est_labels = Evaluator._get_label_intervals(gold_chords)
        return Evaluator.__mir_eval_score(ref_intervals, ref_labels, est_intervals, est_labels)

    @staticmethod
    def _get_label_intervals(chord_sequence):
        """
        """
        intervals = []
        labels = []

        return intervals, labels

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
    print(Evaluator.mir_eval_score(0, 0, "./a.lab", "./a.lab"))

if __name__ == "__main__":

    main()