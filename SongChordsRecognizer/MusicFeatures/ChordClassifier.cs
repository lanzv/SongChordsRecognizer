using SongChordsRecognizer.Graphs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SongChordsRecognizer.MusicFeatures
{
    /// <summary>
    /// Static class that provides functions which will classify chords from some Chromagram.
    /// </summary>
    static class ChordClassifier
    {
        /// <summary>
        /// The function generates chord for each chromagram sample.
        /// </summary>
        /// <param name="chromagram">Chromagram graph.</param>
        /// <returns>The list of chords generated from chomagram.</returns>
        public static List<Chord> GetChords(Chromagram chromagram)
        {
            List<Chord> chords = new List<Chord>();
            double[][] chromagramData = chromagram.GetData();
            for(int i = 0; i < chromagramData.Length; i++)
            {
                chords.Add(ClassifyChord(chromagramData[i]));
            }
            return chords;
        }



        /// <summary>
        /// The private static function that estimate the best match of chord
        /// that corresponds to the passed chromagram sample. 
        /// Function compares triad chords and seventh chords. If probability of 
        /// the most likely seventh chord is higher than 0.6 and the rest of the seventh chord is 
        /// same as the most likely triad, then the seventh chord is used.
        /// Otherwise the most likely triad chord is used.
        /// </summary>
        /// <param name="chromagramSample">One sample of chromagram.</param>
        /// <returns>Chord that corresponds to the chromagram sample.</returns>
        private static Chord ClassifyChord(double[] chromagramSample)
        {
            var best_triads = GetMostLikelyTriad(chromagramSample);
            var best_seventhChord = GetMostLikelySeventhChord(chromagramSample);

            // if chords are same but the seventh has the extra seventh
            // and its probability is more than 0.1
            if (best_triads.Chord.Root.HalfTones == best_seventhChord.Chord.Root.HalfTones && 
                best_triads.Chord.Third.HalfTones == best_seventhChord.Chord.Third.HalfTones &&
                best_triads.Chord.Fifth.HalfTones == best_seventhChord.Chord.Fifth.HalfTones &&
                best_seventhChord.Prob > 0.6
                )
            {
                return best_seventhChord.Chord;
            }
            else return best_triads.Chord;
        }



        /// <summary>
        /// The private static function that estimate the best match of triad chord
        /// that corresponds to the passed chromagram sample. 
        /// Intensities of every chromagam feature are normalized to the [0,1] interval.
        /// The normalized result is used as a probabilities that the specific chroma is used 
        /// in the sample and builds the chord. Probability of chords with perfect fifth
        /// is multiplayed by 2. The most likely triad chord is choosed.
        /// </summary>
        /// <param name="chromagramSample">One sample of chromagram.</param>
        /// <returns>Tuple of most likely Triad Chord and its probability that corresponds to the chromagram sample.</returns>
        private static (Triad Chord, double Prob) GetMostLikelyTriad(double[] chromagramSample)
        {
            List<double> probabilities = new List<double>();
            List<Triad> triads = ChordsGenerator.GetListOfTriads();

            double normalize_coef = chromagramSample.Max();

            foreach (Triad t in triads)
            {
                double prob = 1;
                prob *= chromagramSample[t.Root.HalfTones] / normalize_coef;
                prob *= chromagramSample[t.Third.HalfTones] / normalize_coef;
                prob *= chromagramSample[t.Fifth.HalfTones] / normalize_coef;
                prob *= (t.Fifth.HalfTones + t.Third.HalfTones == 7) ? 2 : 1; // Increase chances for triads with perfect fifth
                probabilities.Add(prob);
            }

            double max_prob = probabilities.Max();
            return (triads[probabilities.IndexOf(max_prob)], max_prob);
        }



        /// <summary>
        /// The private static function that estimate the best match of seventh chord
        /// that corresponds to the passed chromagram sample. 
        /// Intensities of every chromagam feature are normalized to the [0,1] interval.
        /// The normalized result is used as a probabilities that the specific chroma is used 
        /// in the sample and builds the seventh chord. Probability of chords with perfect fifth
        /// is multiplayed by 2. The most likely seventh chord is choosed.
        /// </summary>
        /// <param name="chromagramSample">One sample of chromagram.</param>
        /// <returns>Tuple of most likely Seventh Chord and its probability that corresponds to the chromagram sample.</returns>
        private static (SeventhChord Chord, double Prob) GetMostLikelySeventhChord(double[] chromagramSample)
        {
            List<double> probabilities = new List<double>();
            List<SeventhChord> seventhChords = ChordsGenerator.GetListOfSeventh();

            double normalize_coef = chromagramSample.Max();

            foreach (SeventhChord s in seventhChords)
            {
                double prob = 1;
                prob *= chromagramSample[s.Root.HalfTones] / normalize_coef;
                prob *= chromagramSample[s.Third.HalfTones] / normalize_coef;
                prob *= chromagramSample[s.Fifth.HalfTones] / normalize_coef;
                prob *= (s.Fifth.HalfTones + s.Third.HalfTones == 7) ? 2 : 1; // Increase chances for seventh chords with perfect fifth
                prob *= chromagramSample[s.Seventh.HalfTones] / normalize_coef;
                probabilities.Add(prob);
            }

            double max_prob = probabilities.Max();
            return (seventhChords[probabilities.IndexOf(max_prob)], max_prob);
        }
    }
}