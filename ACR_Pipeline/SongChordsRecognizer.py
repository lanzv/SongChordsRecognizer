#!/usr/bin/env python3
import argparse
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Models import MLP_scalered
from ACR_Training.Spectrograms import log_mel_spectrogram
from ACR_Pipeline.KeyRecognizer import KeyRecognizer
from ACR_Pipeline.DataPreprocessor import DataPreprocessor
from ACR_Pipeline.ChordVoter import ChordVoter


parser = argparse.ArgumentParser()
# Song Chords Recognizer arguments
parser.add_argument("--waveform", default=[], type=list, help="")
parser.add_argument("--sample_rate", default=44100, type=int, help="")

# Training args
parser.add_argument("--seed", default=42, type=int, help="Random seed.")



def main(args):
    # Arguments
    hop_length = 1024
    window_size = 5
    waveform = args.waveform
    sample_rate = 44100
    spectrogram_type = log_mel_spectrogram
    skip_coef=22


    # Load models
    basic_mlp = MLP_scalered.load('./ACR_Pipeline/models/basic_mlp.model')
    C_transposed_mlp = MLP_scalered.load('./ACR_Pipeline/models/C_transposed_mlp.model')



    # Preprocess Data
    x = DataPreprocessor.flatten_preprocess(
        waveform=waveform,
        sample_rate=sample_rate,
        hop_length=hop_length,
        window_size=window_size,
        spectrogram_generator=spectrogram_type,
        norm_to_C=False,
        skip_coef=skip_coef
    )

    # Get list of played chords
    baisc_chord_prediction = basic_mlp.predict(x)
    chords, counts = np.unique(baisc_chord_prediction, return_counts=True)
    chord_counts = dict(zip(chords, counts))

    # Get song's key (not really tonic, A minor/ailoian is same as a C major or D dorian)
    key = KeyRecognizer.estimate_key(chord_counts)

    # Tranapose Song to a C major
    x_transposed = DataPreprocessor.flatten_preprocess(
        waveform=waveform,
        sample_rate=sample_rate,
        hop_length=hop_length,
        window_size=window_size,
        spectrogram_generator=spectrogram_type,
        norm_to_C=True,
        key=key,
        skip_coef=skip_coef
    )

    # Get chord sequence of a song
    transposed_chord_prediction = C_transposed_mlp.predict(x_transposed)

    # Chord voting for each beat
    chord_sequence = ChordVoter.vote_for_beats(
        chord_sequence=transposed_chord_prediction,
        waveform=waveform, sample_rate=sample_rate,
        hop_length=hop_length
    )

    # Transpose to the original sequence
    original_chord_sequence = DataPreprocessor.transpose(
        chord_sequence=chord_sequence,
        from_key = 'C',
        to_key = key
    )

    return DataPreprocessor.chord_indices_to_notations(original_chord_sequence)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)