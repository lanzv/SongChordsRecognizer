#!/usr/bin/env python3
import sys


class ChordSequence():
    """Chord sequence of specific audio parsed from LAB file.
    START list of starting chord times in miliseconds
    END list of ending chord times in miliseconds
    CHORD sequence of chords (labeled as strings)
    """
    def __init__(self, lab_path):

        self.START, self.END, self.CHORD = ChordSequence.parse_lab(lab_path)

    @staticmethod
    def parse_lab(lab_path):
        """
        Load .lab file containing chord sequence.

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
            sequence of chords (labeled as strings)
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

class KeySequence():
    """Key sequence of specific audio parsed from LAB file.
    START list of starting key times in miliseconds
    END list of ending key times in miliseconds
    KEY sequence of keys (labeled as strings)
    """
    def __init__(self, lab_path):

        self.START, self.END, self.KEYS = KeySequence.parse_lab(lab_path)
    

    def get_first_key(self):
        """
        Get first audio key that is not Silence or None key. For instance, when we want to ignore modulations.

        Returns
        -------
        key : string
            first audio music key
        """
        for key in self.KEYS:
            if not (key == "N" or key == "S"):
                return key
        return "N"

    @staticmethod
    def parse_lab(lab_path):
        """
        Load .lab file containing key sequence.

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
            sequence of keys (labeled as strings)
        """
        starts, ends, labels = [], [], []
        with open(lab_path, 'r') as lab_file:
            for line in lab_file:
                if line:
                    splits = line.split()
                    if len(splits) == 3:
                        starts.append(float(splits[0]))
                        ends.append(float(splits[1]))
                        # Silence
                        if not splits[2] == "Silence":
                            raise Exception("The key lab file has an unexpected format.")
                        labels.append("S")
                    if len(splits) == 4:
                        starts.append(float(splits[0]))
                        ends.append(float(splits[1]))
                        # Key
                        if not splits[2] == "Key":
                            raise Exception("The key lab file has an unexpected format.")
                        labels.append(splits[3])
        return starts, ends, labels

class SongDescription():
    """
    """
    def __init__(self, desc_path):
        self.TITLE, self.ARTIST, self.METRE, self.TONIC = SongDescription.parse_lab(desc_path)

    @staticmethod
    def parse_lab(desc_path):
        """
        """
        title, artist, metre, tonic = "", "", "", ""
        with open(desc_path, 'r') as decs_file:
            for line in decs_file:
                if line:
                    splits = line.split(": ")
                    if len(splits) == 2:
                        if splits[0] == "# title":
                            title = splits[1].split('\n')[0]
                        elif splits[0] == "# artist":
                            artist = splits[1].split('\n')[0]
                        elif splits[0] == "# metre":
                            metre = splits[1].split('\n')[0]
                        elif splits[0] == "# tonic":
                            tonic = splits[1].split('\n')[0]
                    else:
                        break
        return title, artist, metre, tonic