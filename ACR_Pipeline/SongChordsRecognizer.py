#!/usr/bin/env python3
import argparse
import numpy as np
import librosa
import sys, os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Models import MLP_scalered, CRNN, CRNN_basic_WithStandardScaler
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
parser.add_argument("--sample_rate_original", default=22050, type=int, help="Sample rate of songs that the original model was trained on.")
parser.add_argument("--hop_length_original", default=512, type=int, help="Hop length of songs that the original model was trained on.")
parser.add_argument("--sample_rate_transposed", default=22050, type=int, help="Sample rate of songs that the transposed model was trained on.")
parser.add_argument("--hop_length_transposed", default=512, type=int, help="Hop length of songs that the original transposed was trained on.")
parser.add_argument("--window_size", default=5, type=int, help="In case that the MLP model is used, the size of the spectrogram window used as a features.")
parser.add_argument("--n_frames", default=1000, type=int, help="In case that the CRNN model is used, the length of the chord sequence that is the model trained on.")
parser.add_argument("--skip_coef", default=22, type=int, help="In case that the MLP model is used, the coeficient that next spectrogram index is multiplied by when creating the window.")
parser.add_argument("--original_model_path", default="./ACR_Pipeline/models/original_crnn.h5", type=str, help="Path to the model predicting original songs.")
parser.add_argument("--transposed_model_path", default="./ACR_Pipeline/models/transposed_crnn.h5", type=str, help="Path to the model predicting songs transposed to C.")
parser.add_argument("--original_preprocessor_path", default="C:/Users/vojte/source/repos/SongChordsRecognizer/ACR_Pipeline/models/original_preprocessor.bin", type=str, help="Path to the model predicting original songs.")
parser.add_argument("--transposed_preprocessor_path", default="C:/Users/vojte/source/repos/SongChordsRecognizer/ACR_Pipeline/models/transposed_preprocessor.bin", type=str, help="Path to the model predicting songs transposed to C.")

# Training args
parser.add_argument("--seed", default=42, type=int, help="Random seed.")



def main(args, waveform, sample_rate):
    # Resample the waveform
    waveform = librosa.resample(waveform, sample_rate, args.sample_rate_original)



    # Load models
    """ MLP Part, not accurate
    basic_mlp = MLP_scalered.load('C:\\Users\\vojte\\source\\repos\\SongChordsRecognizer\\ACR_Pipeline\\models\\original_mlp.model')
    """
    basic_crnn = CRNN_basic_WithStandardScaler()
    basic_crnn.load(args.original_model_path, args.original_preprocessor_path)
    C_transposed_crnn = CRNN_basic_WithStandardScaler()
    C_transposed_crnn.load(args.transposed_model_path, args.transposed_preprocessor_path)



    # Preprocess Data
    """ MLP Part, not that accurate
    x = DataPreprocessor.flatten_preprocess(
        waveform=waveform,
        sample_rate=args.sample_rate_mlp,
        hop_length=args.hop_length_mlp,
        window_size=args.window_size,
        spectrogram_generator=log_mel_spectrogram,
        norm_to_C=False,
        skip_coef=args.skip_coef
    )
    """
    x = DataPreprocessor.sequence_preprocess(
        waveform=waveform,
        sample_rate=args.sample_rate_original,
        hop_length=args.hop_length_original,
        n_frames=args.n_frames,
        spectrogram_generator=cqt_spectrogram,
        norm_to_C=False,
    )

    # Get list of played chords
    """ MLP Part, not that accurate
    baisc_chord_prediction = basic_crnn.predict(x)
    """
    baisc_chord_prediction = basic_crnn.predict(x).argmax(axis=2).flatten()
    chords, counts = np.unique(baisc_chord_prediction, return_counts=True)
    chord_counts = dict(zip(chords, counts))

    # Get song's key (not really tonic, A minor/ailoian is same as a C major or D dorian)
    key = KeyRecognizer.estimate_key(chord_counts)

    # Tranapose Song to a C major
    resampled_waveform = librosa.resample(waveform, args.sample_rate_original, args.sample_rate_transposed) # In case that the transposed model has different sample_rate
    x_transposed = DataPreprocessor.sequence_preprocess(
        waveform=resampled_waveform,
        sample_rate=args.sample_rate_transposed,
        hop_length=args.hop_length_transposed,
        n_frames=args.n_frames,
        spectrogram_generator=cqt_spectrogram,
        norm_to_C=True,
        key=key,
    )

    # Get chord sequence of a song
    transposed_chord_prediction = C_transposed_crnn.predict(x_transposed).argmax(axis=2).flatten()


    # Chord voting for each beat
    chord_sequence, bpm, beat_times, n_quarters_for_bar = ChordVoter.vote_for_beats(
        chord_sequence=transposed_chord_prediction,
        waveform=resampled_waveform, sample_rate=args.sample_rate_transposed,
        hop_length=args.hop_length_transposed
    )

    # Transpose to the original sequence
    original_chord_sequence = DataPreprocessor.transpose(
        chord_sequence=chord_sequence,
        from_key = 'C',
        to_key = key
    )


    # Print data on the standard output in JSON format
    json_out = {}
    json_out["Key"] = key
    json_out["BPM"] = bpm
    json_out["ChordSequence"] = DataPreprocessor.chord_indices_to_notations(original_chord_sequence)
    json_out["BeatTimes"] = beat_times.tolist()
    json_out["BarQuarters"] = n_quarters_for_bar
    print(json.dumps(json_out))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Get input data from standard input
    json_in = json.load(sys.stdin)

    # Parse input data
    waveform = np.array(json_in["Waveform"])
    sample_rate = float(json_in["SampleRate"])

    main(args, waveform, sample_rate)