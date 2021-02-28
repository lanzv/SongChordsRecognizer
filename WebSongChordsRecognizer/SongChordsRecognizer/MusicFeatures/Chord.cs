using System;

namespace SongChordsRecognizer.MusicFeatures
{
    #region General Chord classes

    /// <summary>
    /// Class of chords,
    /// One Chord is characterized by
    /// Description and at least three tones:
    /// Root ( Root Chroma, base chord tone ),
    /// Third ( Interval of minor third or major third by Root tone ),
    /// Fifth ( Interval of minor third or major third by Third tone)
    /// </summary>
    public class Chord
    {
        public String Description;
        public Chroma Root;
        public Chroma Third;
        public Chroma Fifth;
    }



    #endregion


    #region Chord types and extensions

    /// <summary>
    /// Class of triads - chord of three tones / chroma.
    /// We have only 48 triads ( 12 root Chroma, two thirds and two fifths .. diminished, minor, major, augmented ).
    /// </summary>
    public class Triad : Chord
    {
    }



    /// <summary>
    /// Class of Seventh Chord - three tone chord with extra seventh.
    /// We have only 84 Seventh Chord ( 12 root Chroma, seven Seventh:
    ///     Major seventh, Minor seventh, Dominant seventh, Diminished seventh, 
    ///     Half-diminished seventh, Minor major seventh, Augmented major seventh
    /// </summary>
    public class SeventhChord : Chord
    {
        public Chroma Seventh;
    }



    #endregion
}