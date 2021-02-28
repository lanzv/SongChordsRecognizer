using System;
using System.Collections.Generic;

namespace SongChordsRecognizer.MusicFeatures
{
    /// <summary>
    /// Static class that provides important sets connected with Chord class. 
    /// </summary>
    public static class ChordsGenerator
    {
        #region Public core methods

        /// <summary>
        /// Function generates list of all possible triad chords.
        /// Overall there are 48 chords, 12 possible root (12 chroma) and for each root we have
        /// major, minor, augmented and diminished chord.
        /// One chord has String description, root chroma, third chroma and fifth chroma. 
        /// </summary>
        /// <returns>The list of all 48 triad chords.</returns>
        public static List<Triad> GetListOfTriads()
        {
            List<Triad> chords = new List<Triad>();

            List<Chroma> chroma = TonesGenerator.Chroma;
            
            // All root tones
            foreach(Chroma ch in chroma)
            {
                // All thirds - minor, major
                for(int i = 3; i <= 4; i++)
                {
                    // All triads - diminished, minor, major, augmented
                    for(int j = 3; j <= 4; j++)
                    {
                        chords.Add(
                            new Triad()
                            {
                                Description = ch.Description + "" + TriadDescription(i, j), 
                                Root = ch,
                                Third = chroma[(ch.HalfTones + i)%chroma.Count],
                                Fifth = chroma[(ch.HalfTones + i + j)%chroma.Count]
                            }
                        );
                    }
                }
            }
            return chords;
        }



        /// <summary>
        /// Function generates list of all possible Seventh chords.
        /// Overall there are 84 chords, 12 possible root (12 chroma) and for each root we have
        /// major seventh, minor seventh, dominant sevneth, diminished seventh, half-diminished seventh, 
        /// minor major seventh, augmented major seventh.
        /// One chord has String description, root chroma, third chroma, fifth chroma and seventh chroma. 
        /// </summary>
        /// <returns>The list of all 84 seventh chords.</returns>
        public static List<SeventhChord> GetListOfSeventh()
        {
            List<SeventhChord> chords = new List<SeventhChord>();

            List<Chroma> chroma = TonesGenerator.Chroma;

            // All root tones
            foreach (Chroma ch in chroma)
            {
                // All thirds - minor, major
                for (int i = 3; i <= 4; i++)
                {
                    // All triads - diminished, minor, major, augmented
                    for (int j = 3; j <= 4; j++)
                    {
                        // All seventh chords
                        for(int k = 3; k <= 4; k++)
                        {
                            if(i + j + k == 12) { continue; } // Seventh is same as a root tone. 
                            chords.Add(
                                new SeventhChord()
                                {
                                    Description = ch.Description + "" + SeventhDescription(i, j, k),
                                    Root = ch,
                                    Third = chroma[(ch.HalfTones + i) % chroma.Count],
                                    Fifth = chroma[(ch.HalfTones + i + j) % chroma.Count],
                                    Seventh = chroma[(ch.HalfTones + i + j) % chroma.Count]
                                }
                            );
                        }
                    }
                }
            }
            return chords;
        }



        #endregion


        #region Private methods

        /// <summary>
        /// The private function that creates a triad description based on its intervals.
        /// </summary>
        /// <param name="third">Number of half tones between root and third tones. Basically 3 or 4.</param>
        /// <param name="fifth">Number of half tones between third and fifth tones. Basically 3 or 4.</param>
        /// <returns>Chord specification, one of these options: {, m, aug, dim}. </returns>
        private static String TriadDescription(int third, int fifth)
        {
            if(third == 3)
            {
                if(fifth == 3){ return "dim"; } // diminished
                else if(fifth == 4) { return "m"; } // minor
            }
            else if(third == 4)
            {
                if(fifth == 3) { return ""; } // major
                else if(fifth == 4) { return "aug";  } // augmented
            }
            throw new Exception(ErrorMessages.ErrorMessages.ChordGenerator_InvalidIntervalFormat);
        }



        /// <summary>
        /// The private function that creates a seventh chord description based on its intervals.
        /// </summary>
        /// <param name="third">Number of half tones between root and third tones. Basically 3 or 4.</param>
        /// <param name="fifth">Number of half tones between third and fifth tones. Basically 3 or 4.</param>
        /// <param name="seventh">Number of half tones between fifth and seventh tones. Basically 3 or 4.</param>
        /// <returns>Chord specification, one of these options: {dim 7, m 7 b5, m 7, m Maj7, 7, Maj7, Maj7 #5}. </returns>
        private static String SeventhDescription(int third, int fifth, int seventh)
        {
            if (third == 3)
            {
                if (fifth == 3)
                {
                    if (seventh == 3) return "dim 7"; //  Diminished seventh
                    if (seventh == 4) return "m 7 b5"; // Half-diminished seventh
                }
                else if (fifth == 4)
                {
                    if (seventh == 3) return "m 7"; // Minor seventh
                    if (seventh == 4) return "m Maj7"; // Minor major seventh
                }
            }
            else if (third == 4)
            {
                if (fifth == 3)
                {
                    if (seventh == 3) return "7"; // Dominant seventh
                    if (seventh == 4) return "Maj7"; // Major seventh
                }
                else if (fifth == 4 && seventh == 3)
                {
                    return "Maj7 #5"; // Augmented major seventh
                }
            }
            throw new Exception(ErrorMessages.ErrorMessages.ChordGenerator_InvalidIntervalFormat);
        }



        #endregion
    }
}