#!/usr/bin/env python3
import sys


class ChordsData():
    """Chord sequence of specific audio parsed from LAB file.
    START list of starting chord time in miliseconds
    END list of ending chord time in miliseconds
    CHORD sequence of chords (labeled as strings)
    """
    def __init__(self, lab_path):

        self.START, self.END, self.CHORD = ChordsData.parse_lab(lab_path)

    @staticmethod
    def parse_lab(lab_path):
        """
        Load .lab file containing sequence of chords.

        Parameters
        ----------
        lab_path : str
            path to lab file
        Returns
        -------
        starts : float list
            starting chord time in miliseconds in respect to whole song
        ends : float list
            ending chord time in miliseconds in respect to whole song
        chords : str list
            sequence of chords (labeled as strings)
        """
        starts, ends, chords = [], [], []
        with open(lab_path, 'r') as lab_file:
            for line in lab_file:
                if line:
                    splits = line.split()
                    if len(splits) == 3:
                        starts.append(float(splits[0]))
                        ends.append(float(splits[1]))
                        chords.append(splits[2])
        return starts, ends, chords