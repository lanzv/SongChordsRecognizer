


class KeyRecognizer():

    key_chords = {
        # C major, D minor, E minor, F major, G major, A minor, B minor
        'C' : [1, 6, 10, 11, 15, 20, 24],
        # Cis/Des major, Dis/Es minor, F minor, Fis/Ges major, Gis/As major, Ais/Bes minor, C minor
        'C#/Db' : [3, 8, 12, 13, 17, 22, 2],
        # D major, E minor, Fis/Ges minor, G major, A major, B minor, Cis/Des minor
        'D' : [5, 10, 14, 15, 19, 24, 4],
        # Dis/Es major, F minor, G minor, Gis/As major, Ais/Bes major, C minor, D minor
        'D#/Eb' : [7, 12, 16, 17, 21, 2, 6],
        # E major, Fis/Ges minor, Gis/As minor, A major, B major, Cis/Des minor, Dis/Es minor
        'E' : [9, 14, 18, 19, 23, 4, 8],
        # F major, G minor, A minor, Ais/Bes major, C major, D minor, E minor
        'F' : [11, 16, 20, 21, 1, 6, 10],
        # Fis/Ges major, Gis/Des minor, Ais/Bes minor, B major, Cis/Des major, Dis/Es minor, F minor
        'F#/Gb' : [13, 18, 22, 23, 3, 8, 12],
        # G major, A minor, B minor, C major, D major, E minor, Fis/Ges minor
        'G' : [15, 20, 24, 1, 5, 10, 14],
        # Gis/As major, Ais/Bes minor, C minor, Cis/Des major, Dis/Es major, F minor, G minor
        'G#/Ab' : [17, 22, 2, 3, 7, 12, 16],
        # A major, B minor, Cis/Des minor, D major, E major, Fis/Ges minor, Gis/As minor
        'A' : [19, 24, 4, 5, 9, 14, 18],
        # Ais/Bes major, C minor, D minor, Dis/Es major, F major, G minor, A minor
        'A#/Bb' : [21, 2, 6, 7, 11, 16, 20],
        # B major, Cis/Des minor, Dis/Es minor, E major, Fis/Ges major, Gis/As minor, Ais/Bes minor
        'B' : [23, 4, 8, 9, 13, 18, 22],
    }

    @staticmethod
    def estimate_key(chord_counts):
        """
        Function will estimate key based on the chord counts predicted (with not that high accuracy) from the audio.
        All estimated keys are in Ionian modus. The function doesn't estimate the modus (dorian, phrygian, lydian, ...).

        Parameters
        ----------
        chord_counts : dictionary, key: int, value: int
            dictionary of chord counts in the song, a key indicates the chord index, a value indicates its count in the audio
        Returns
        -------
        key : np array
            flattened window of spectrograms arround specific time point
        """
        scores = {}
        # Iterate over all keys
        for key in KeyRecognizer.key_chords:
            score = 0
            # Sum number of chords that fits the key
            for chord_ind in KeyRecognizer.key_chords[key]:
                if chord_ind in chord_counts:
                    score = score + chord_counts[chord_ind]
            scores[key] = score
        
        # Get the most common key with the highest score
        key = max(scores, key=scores.get).split("/")[0]
        return key
        