#!/usr/bin/env python3
import argparse

from Datasets import BillboardDataset, IsophonicsDataset
import sys
from Spectrograms import cqt_spectrogram, log_mel_spectrogram


def save_preprocessed_Isophonics(args):
    # Get spectrogram type
    if args.spectrogram_type == "cqt":
        spectrogram = cqt_spectrogram
    elif args.spectrogram_type == "log_mel":
        spectrogram = log_mel_spectrogram

    # Prepare and Save Isophonics dataset
    #data = IsophonicsDataset(args.isophonics_audio_directory, args.isophonics_annotations_directory, sample_rate=args.sample_rate)
    data = IsophonicsDataset.load_dataset("./SavedDatasets/Isophonics_22050.ds")
    data.save_preprocessed_dataset(dest=args.isophonics_prep_dest, hop_length=args.hop_length, norm_to_C=args.norm_to_C, spectrogram_generator=spectrogram, n_frames=args.n_frames)


def save_preprocessed_Billboard(args):
    # Prepare and Save Billboard dataset
    data = BillboardDataset(audio_directory=args.billboard_audio_directory, annotations_directory=args.billboard_annotations_directory)
    data.save_preprocessed_dataset(dest=args.billboard_prep_dest, n_frames=args.n_frames)


parser = argparse.ArgumentParser()
# Directories, destinations, folders, files
parser.add_argument("--isophonics_audio_directory", default="./Datasets/Isophonics/AUDIO", type=str, help="Path to ISOPHONICS directory with audio files.")
parser.add_argument("--isophonics_annotations_directory", default="./Datasets/Isophonics/ANNOTATIONS", type=str, help="Path to ISOPHONICS directory with chord annotations.")
parser.add_argument("--billboard_audio_directory", default="./Datasets/Billboard/AUDIO", type=str, help="Path to BILLBOARD directory with audio files.")
parser.add_argument("--billboard_annotations_directory", default="./Datasets/Billboard/ANNOTATIONS", type=str, help="Path to BILLBOARD directory with chord annotations.")
parser.add_argument("--isophonics_prep_dest", default="./PreprocessedDatasets/isophonics_new.ds", type=str, help="Preprocessed ISOPHONICS dataset destination.")
parser.add_argument("--billboard_prep_dest", default="./PreprocessedDatasets/billboard_new.ds", type=str, help="Preprocessed BILLBOARD dataset destination.")

# Dataset preprocessing args
parser.add_argument("--dataset", default="isophonics", type=str, help="Dataset we want to preprocess, {isophonics, billboard}")
#           Isophonics
parser.add_argument("--sample_rate", default=44100, type=int, help="Sample rate for each song.")
parser.add_argument("--hop_length", default=512, type=int, help="10*(sample_rate/hop_length) is a number of miliseconds between two frames.")
parser.add_argument("--window_size", default=8, type=int, help="Spectrograms on left, and also spectrogram on right of the time bin -> window_size*2 + 1 spectrograms grouped together.")
parser.add_argument("--flattened_window", default=False, type=bool, help="Whether the spectrogram window should be flatten to one array or it sould be array of spectrograms.")
parser.add_argument("--ms_intervals", default=430.6640625, type=float, help="Miliseconds between generated spectrograms.")
parser.add_argument("--to_skip", default=10, type=int, help="How many spectrogram we want to skip when creating spectrogram window.")
parser.add_argument("--norm_to_C", default=True, type=bool, help="Whether we want to transpose all songs to C key (or D dorian, .. A minor, ...)")
parser.add_argument("--spectrogram_type", default="cqt", type=str, help="Spectrogram types, {cqt,log_mel}")
#           Billboard
parser.add_argument("--n_frames", default=1000, type=int, help="Length of song subsequence we are consinder when predicting chords to keep some context.")

# Training args
parser.add_argument("--test_size", default=0.3, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")



def main(args):
    if args.dataset == "isophonics":
        save_preprocessed_Isophonics(args)
    elif args.dataset == "billboard":
        save_preprocessed_Billboard(args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)