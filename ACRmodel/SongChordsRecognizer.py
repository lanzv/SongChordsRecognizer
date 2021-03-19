#!/usr/bin/env python3
import argparse


parser = argparse.ArgumentParser()
# Song Chords Recognizer arguments
parser.add_argument("--waveform", default=[], type=list, help="")

# Training args
parser.add_argument("--seed", default=42, type=int, help="Random seed.")



def main(args):
    # Get list of played chords

    # Get song's key (not really tonic, A moll is same as a C major or D dorian)

    # Tranapose Song to a C major

    # Get chord sequence of a song

    # Print chords in a original key

    print("[]")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)