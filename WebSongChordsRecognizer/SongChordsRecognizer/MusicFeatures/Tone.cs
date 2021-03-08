using System;

namespace SongChordsRecognizer.MusicFeatures
{
    #region Tone descriptions, classes, structs

    /// <summary>
    /// Structure of tones. We have 108 tones (9 octaves and 12 chroma).
    /// One Tone is characterized by
    /// Description (e.g. C, C#, ...), 
    /// Octave (e.g. 0 - sub contra, 1 - contra, .., 8 - five lined), 
    /// Chroma (instances of Chroma - C, C#, ..., B)
    /// Frequency (default accurate frequency of that Tone), 
    /// L_FreqBorder, R_FreqBorder (borders of deviation from the Frequency value)
    /// </summary>
    public struct Tone
    {
        public String Description;
        public int Octave;
        public Chroma Chroma;
        public double Frequency;
        public double L_FreqBorder;
        public double R_FreqBorder;
    }



    /// <summary>
    /// Structure of Chroma. We have 12 Chroma:
    /// C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    /// One Chroma is characterized by
    /// Description and HalfTones which is number of half tones
    /// between this Chroma and Chroma 'C' 
    /// </summary>
    public struct Chroma
    {
        public String Description;
        public int HalfTones;
    }



    #endregion
}