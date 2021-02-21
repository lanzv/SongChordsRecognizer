#!/usr/bin/env python3
import argparse

from numpy.lib.npyio import save
from Datasets import IsophonicsDataset 
from Models import MLP, CNN
import sklearn
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--audio_directory", default="./Datasets/Isophonics/AUDIO", type=str, help="Path to directory with audio files.")
parser.add_argument("--annotations_directory", default="./Datasets/Isophonics/ANNOTATIONS", type=str, help="Path to directory with chord annotations.")
parser.add_argument("--test_size", default=0.3, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--seed", default=42, type=int, help="Random seed")

def main(args):
    # Prepare Data
    data = IsophonicsDataset(args.audio_directory, args.annotations_directory)
    data.save_preprocessed_dataset(dest="./new_dataset.ds",window_size=5, flattened_window=False, ms_intervals=500, to_skip=1)
    sys.exit()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
