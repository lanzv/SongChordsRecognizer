#!/usr/bin/env python3
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\"))
from ACR_Training.Datasets import BillboardDataset, IsophonicsDataset
from ACR_Training.Spectrograms import cqt_spectrogram, log_mel_spectrogram, cqt_chromagram, stft_chromagram
import pickle
import lzma

def save_preprocessed_Isophonics(args):
    # Get spectrogram type
    if args.feature_type == "cqt_spec":
        spectrogram = cqt_spectrogram
    elif args.feature_type == "log_mel_spec":
        spectrogram = log_mel_spectrogram
    elif args.feature_type == "cqt_chrom":
        spectrogram = cqt_chromagram
    elif args.feature_type == "stft_chrom":
        spectrogram = stft_chromagram

    # Prepare and Save Isophonics dataset
    data = IsophonicsDataset(args.isophonics_audio_directory, args.isophonics_annotations_directory, sample_rate=args.sample_rate)
    #data = IsophonicsDataset.load_dataset("./ACR_Training/SavedDatasets/Isophonics_22050.ds")
    #data.save_segmentation_samples(dest="./ACR_Training/Segmentations/Isophonics50.seg",hop_length=args.hop_length, norm_to_C=args.norm_to_C, spectrogram_generator=spectrogram, n_frames=args.n_frames)
    #data.save_preprocessed_dataset(dest=args.isophonics_prep_dest, hop_length=args.hop_length, norm_to_C=args.norm_to_C, spectrogram_generator=spectrogram, n_frames=args.n_frames)
    with lzma.open(args.isophonics_prep_dest, "wb") as dataset_file:
        pickle.dump((data.preprocess_single_chords_list(args.window_size, args.flattened_window, args.hop_length, args.to_skip, args.norm_to_C, spectrogram, args.skip_coef)), dataset_file)
    print("[INFO] The Dataset was saved successfully.")

def save_preprocessed_Billboard(args):
    # Get spectrogram type
    if args.feature_type == "cqt_spec":
        spectrogram = cqt_spectrogram
    elif args.feature_type == "log_mel_spec":
        spectrogram = log_mel_spectrogram
    elif args.feature_type == "cqt_chrom":
        spectrogram = cqt_chromagram
    elif args.feature_type == "stft_chrom":
        spectrogram = stft_chromagram


    # Prepare and Save Billboard dataset
    data = BillboardDataset(audio_directory=args.billboard_audio_directory, annotations_directory=args.billboard_annotations_directory, audio = args.billboard_audio)
    with lzma.open(args.isophonics_prep_dest, "wb") as dataset_file:
        pickle.dump((data.preprocess_single_chords_list(args.window_size, args.flattened_window, args.hop_length, args.to_skip, args.norm_to_C, spectrogram, args.skip_coef)), dataset_file)
    #data.save_preprocessed_dataset(dest=args.billboard_prep_dest, hop_length=args.hop_length, norm_to_C=args.norm_to_C, spectrogram_generator=spectrogram, n_frames=args.n_frames)
    #data.save_segmentation_samples(dest="./ACR_Training/Segmentations/Billboard1000.seg", n_frames=500)


parser = argparse.ArgumentParser()
# Directories, destinations, folders, files
parser.add_argument("--isophonics_audio_directory", default="./ACR_Training2/Datasets/Isophonics/AUDIO", type=str, help="Path to ISOPHONICS directory with audio files.")
parser.add_argument("--isophonics_annotations_directory", default="./ACR_Training2/Datasets/Isophonics/ANNOTATIONS", type=str, help="Path to ISOPHONICS directory with chord annotations.")
parser.add_argument("--billboard_audio_directory", default="./ACR_Training/Datasets/Billboard_testset/AUDIO", type=str, help="Path to BILLBOARD directory with audio files.")
parser.add_argument("--billboard_annotations_directory", default="./ACR_Training/Datasets/Billboard_testset/ANNOTATIONS", type=str, help="Path to BILLBOARD directory with chord annotations.")
parser.add_argument("--isophonics_prep_dest", default="./ACR_Training/PreprocessedDatasets/isophonics_new.ds", type=str, help="Preprocessed ISOPHONICS dataset destination.")
parser.add_argument("--billboard_prep_dest", default="./ACR_Training/PreprocessedDatasets/billboard_new.ds", type=str, help="Preprocessed BILLBOARD dataset destination.")

# Dataset preprocessing args
parser.add_argument("--dataset", default="billboard", type=str, help="Dataset we want to preprocess, {isophonics, billboard}")
#           Isophonics
parser.add_argument("--sample_rate", default=44100, type=int, help="Sample rate for each song.")
parser.add_argument("--hop_length", default=22050, type=int, help="10*(sample_rate/hop_length) is a number of miliseconds between two frames.")
parser.add_argument("--window_size", default=5, type=int, help="Spectrograms on left, and also spectrogram on right of the time bin -> window_size*2 + 1 spectrograms grouped together.")
parser.add_argument("--flattened_window", default=True, type=bool, help="Whether the spectrogram window should be flatten to one array or it sould be array of spectrograms.")
parser.add_argument("--to_skip", default=1, type=int, help="How many spectrogram we want to skip when creating spectrogram window.")
parser.add_argument("--norm_to_C", default=True, type=bool, help="Whether we want to transpose all songs to C key (or D dorian, .. A minor, ...)")
parser.add_argument("--skip_coef", default=1, type=int, help="Spectrogram shifts in the window are multiplayed by this ceofs -> we have window with indices 0, 5, 10, 15, 20, .. instead of 8,9,10,11,12")
parser.add_argument("--feature_type", default="cqt_spec", type=str, help="Spectrogram types, {cqt_spec,log_mel_spec,cqt_chrom,stft_chrom}")
#           Billboard
parser.add_argument("--n_frames", default=1000, type=int, help="Length of song subsequence we are consinder when predicting chords to keep some context.")
parser.add_argument("--billboard_audio", default=True, type=bool, help="Whether wav audio files are part of the Billboard Dataset")

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