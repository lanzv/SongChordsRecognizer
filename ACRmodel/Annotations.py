#!/usr/bin/env python3
import sys


class Chords():
    """Chord sequence of specific audio parsed from LAB file.
    START list of starting chord times in miliseconds
    END list of ending chord times in miliseconds
    CHORD sequence of chords (labeled as strings)
    """
    def __init__(self, lab_path):

        self.START, self.END, self.CHORD = parse_lab(lab_path)


class Keys():
    """Key sequence of specific audio parsed from LAB file.
    START list of starting key times in miliseconds
    END list of ending key times in miliseconds
    KEY sequence of keys (labeled as strings)
    """
    def __init__(self, lab_path):

        self.START, self.END, self.KEYS = parse_lab(lab_path)



@staticmethod
def parse_lab(lab_path):
    """
    Load .lab file containing data sequence.

    Parameters
    ----------
    lab_path : str
        path to lab file
    Returns
    -------
    starts : float list
        starting label time in miliseconds in respect to whole song
    ends : float list
        ending label time in miliseconds in respect to whole song
    labels : str list
        sequence of data (labeled as strings)
    """
    starts, ends, labels = [], [], []
    with open(lab_path, 'r') as lab_file:
        for line in lab_file:
            if line:
                splits = line.split()
                if len(splits) == 3:
                    starts.append(float(splits[0]))
                    ends.append(float(splits[1]))
                    labels.append(splits[2])
    return starts, ends, labels