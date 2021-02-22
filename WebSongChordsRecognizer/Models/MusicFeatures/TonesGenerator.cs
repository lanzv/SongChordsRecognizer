using System;
using System.Collections.Generic;

namespace SongChordsRecognizer.MusicFeatures
{
    /// <summary>
    /// Static class that provides important sets connected with Tone or Chroma classes. 
    /// </summary>
    static class TonesGenerator
    {
        /// <summary>
        /// The list of base frequencies that their multiples
        /// by 2^n generate all tones for all 9 octaves. 
        /// </summary>
        public static readonly float[] BaseFrequencies = new float[] {
            16.35160f, // C
            17.32391f, // C#
            18.35405f, // D
            19.44544f, // D#
            20.60172f, // E
            21.82676f, // F
            23.12465f, // F#
            24.49971f, // G
            25.95654f, // G#
            27.50000f, // A
            29.13524f, // A#
            30.86771f  // B
        };

        /// <summary>
        /// The list of chroma, all twelve tones with 
        /// descriptions and half tones, which says how many half
        /// tones are between that chroma and the chroma C. 
        /// </summary>
        public static readonly List<Chroma> Chroma = new List<Chroma>() { 
            new Chroma(){ Description = "C", HalfTones = 0},
            new Chroma(){ Description = "C#", HalfTones = 1},
            new Chroma(){ Description = "D", HalfTones = 2},
            new Chroma(){ Description = "D#", HalfTones = 3},
            new Chroma(){ Description = "E", HalfTones = 4},
            new Chroma(){ Description = "F", HalfTones = 5},
            new Chroma(){ Description = "F#", HalfTones = 6},
            new Chroma(){ Description = "G", HalfTones = 7},
            new Chroma(){ Description = "G#", HalfTones = 8},
            new Chroma(){ Description = "A", HalfTones = 9},
            new Chroma(){ Description = "A#", HalfTones = 10},
            new Chroma(){ Description = "B", HalfTones = 11},
        };



        /// <summary>
        /// Function generates list of tones in 9 octaves: from sub-contra to five-lined octave.
        /// Overall there are 108 tones. One tone has String description, octave number, 
        /// chroma, frequency and frequency deviation. 
        /// </summary>
        /// <returns>The list of 108 tones in interval between sub-contra and five-lined octaves.</returns>
        public static List<Tone> GetListOfTones()
        {
            List<Tone> tones = new List<Tone> { };
            double previous_border = (BaseFrequencies[0] + BaseFrequencies[11] / 2) / 2;
            // iterate over all octaves
            for (int i = 0; i < 9; i++)
            {
                // iterate over single tones
                for (int j = 0; j < Chroma.Count; j++)
                {
                    // add new Tone
                    tones.Add(new Tone()
                    {
                        Description = Chroma[j].Description,
                        Octave = i,
                        Chroma = Chroma[j],
                        Frequency = Math.Pow(2, i) * BaseFrequencies[j],
                        L_FreqBorder = previous_border, 
                        R_FreqBorder = (j == Chroma.Count - 1) ? 
                                       (Math.Pow(2, i) * BaseFrequencies[j] + Math.Pow(2, i + 1) * BaseFrequencies[(j + 1) % Chroma.Count]) / 2 :
                                       (Math.Pow(2, i) * BaseFrequencies[j] + Math.Pow(2, i) * BaseFrequencies[(j + 1) % Chroma.Count]) / 2
                    });
                    previous_border = tones[tones.Count - 1].R_FreqBorder;
                }
            }
            return tones;
        }
    }
}