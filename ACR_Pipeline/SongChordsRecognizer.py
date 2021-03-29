#!/usr/bin/env python3
import argparse
import numpy as np
import librosa
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Models import MLP_scalered, CRNN
from ACR_Training.Spectrograms import log_mel_spectrogram, cqt_spectrogram
from ACR_Pipeline.KeyRecognizer import KeyRecognizer
from ACR_Pipeline.DataPreprocessor import DataPreprocessor
from ACR_Pipeline.ChordVoter import ChordVoter

# Ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser()
# Song Chords Recognizer arguments
parser.add_argument("--sample_rate_mlp", default=44100, type=int, help="")
parser.add_argument("--hop_length_mlp", default=1024, type=int, help="")
parser.add_argument("--sample_rate_crnn", default=22050, type=int, help="")
parser.add_argument("--hop_length_crnn", default=512, type=int, help="")
parser.add_argument("--window_size", default=5, type=int, help="")
parser.add_argument("--n_frames", default=1000, type=int, help="")
parser.add_argument("--skip_coef", default=22, type=int, help="")


# Training args
parser.add_argument("--seed", default=42, type=int, help="Random seed.")



def main(args, waveform, sample_rate):
    # Resample the waveform
    waveform = librosa.resample(waveform, sample_rate, args.sample_rate_mlp)



    # Load models
    basic_mlp = MLP_scalered.load('C:\\Users\\vojte\\source\\repos\\SongChordsRecognizer\\ACR_Pipeline\\models\\original_mlp.model')
    C_transposed_crnn = CRNN()
    C_transposed_crnn.load('C:\\Users\\vojte\\source\\repos\\SongChordsRecognizer\\ACR_Pipeline\\models\\transposed_crnn.h5')



    # Preprocess Data
    x = DataPreprocessor.flatten_preprocess(
        waveform=waveform,
        sample_rate=args.sample_rate_mlp,
        hop_length=args.hop_length_mlp,
        window_size=args.window_size,
        spectrogram_generator=log_mel_spectrogram,
        norm_to_C=False,
        skip_coef=args.skip_coef
    )

    # Get list of played chords
    baisc_chord_prediction = basic_mlp.predict(x)
    chords, counts = np.unique(baisc_chord_prediction, return_counts=True)
    chord_counts = dict(zip(chords, counts))

    # Get song's key (not really tonic, A minor/ailoian is same as a C major or D dorian)
    key = KeyRecognizer.estimate_key(chord_counts)

    # Tranapose Song to a C major
    resampled_waveform = librosa.resample(waveform, args.sample_rate_mlp, args.sample_rate_crnn)
    x_transposed = DataPreprocessor.sequence_preprocess(
        waveform=resampled_waveform,
        sample_rate=args.sample_rate_crnn,
        hop_length=args.hop_length_crnn,
        n_frames=args.n_frames,
        spectrogram_generator=cqt_spectrogram,
        norm_to_C=True,
        key=key,
    )

    # Get chord sequence of a song
    transposed_chord_prediction = C_transposed_crnn.predict(x_transposed).argmax(axis=2).flatten()


    # Chord voting for each beat
    chord_sequence, bpm = ChordVoter.vote_for_beats(
        chord_sequence=transposed_chord_prediction,
        waveform=resampled_waveform, sample_rate=args.sample_rate_crnn,
        hop_length=args.hop_length_crnn
    )

    # Transpose to the original sequence
    original_chord_sequence = DataPreprocessor.transpose(
        chord_sequence=chord_sequence,
        from_key = 'C',
        to_key = key
    )


    # Print data on the standard output
    print(key)
    print(bpm)
    print(DataPreprocessor.chord_indices_to_notations(original_chord_sequence))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Get input data from standard input
    waveform = np.array(sys.stdin.readline().split(';')).astype(np.float)
    sample_rate = float(sys.stdin.readline())

    main(args, waveform, sample_rate)