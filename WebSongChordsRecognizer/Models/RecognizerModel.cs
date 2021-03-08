using Microsoft.AspNetCore.Http;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.IO;

namespace WebSongChordsRecognizer.Models
{
    /// <summary>
    /// ASP.NET Model to store information about generated chord sequence.
    /// </summary>
    public class RecognizerModel
    {
        #region Properties

        /// <summary>
        /// Boolean value. true if ChordSequence has been generated. Otherwise false.
        /// </summary>
        public Boolean ChordsPrepared { get; private set; }
        /// <summary>
        /// Sequence of chords played in an audio.
        /// </summary>
        public List<Chord> ChordSequence { get; private set; }



        #endregion


        #region Initialization

        /// <summary>
        /// Initialize RecognizerModel and its properties.
        /// </summary>
        public RecognizerModel()
        {
            ChordsPrepared = false;
            ChordSequence = new List<Chord>();
        }



        #endregion


        #region Public core methods

        /// <summary>
        /// Whole process of the Song Chords Recognizer algorithm.
        /// audio file -> Spectrogram -> Chromagram -> Chord Classification
        /// </summary>
        /// <param name="audio">IFormFile of an audio we want to process.</param>
        /// <param name="window">STFT window.</param>
        /// <param name="filtration">Spectrogram filtration type.</param>
        /// <param name="sampleLengthLevel">Level of sample time for one chord.</param>
        public void ProcessAudio(IFormFile audio, IWindow window, ISpectrogramFiltration filtration, int sampleLengthLevel, int bpm)
        {
            // Generate chords
            using (var ms = new MemoryStream())
            {
                // Get audio data
                audio.CopyTo(ms);
                var audioBytes = ms.ToArray();
                // Parse audio data
                AudioSourceWav wav = new AudioSourceWav(audioBytes, audio.FileName);
                // Generate SPECTROGRAM
                Spectrogram spectrogram = new Spectrogram(wav, sampleLengthLevel, window);
                // Generate CHROMAGRAM
                Chromagram chromagram = new Chromagram(spectrogram, filtration);
                // Classify chords
                ChordSequence = ChordClassifier.GetChords(chromagram, bpm);

                ChordsPrepared = true;
            }
        }



        /// <summary>
        /// Reset all class data.
        /// </summary>
        public void ResetRecognizer()
        {
            ChordsPrepared = false;
            ChordSequence = new List<Chord>();
        }



        /// <summary>
        /// Convert ChordSequence List to justified and readable string.
        /// </summary>
        /// <returns>String of song's chord sequence.</returns>
        public override string ToString()
        {
            String result = "";
            String newLine = Environment.NewLine;
            if (ChordsPrepared)
            {
                // ----------------- PRINT CHORDS -----------------
                result += newLine;
                result += new String('-', 56) + " CHORDS " + new String('-', 56) + newLine;
                result += newLine;
                for (int i = 0; i < ChordSequence.Count; i++)
                {
                    result += ChordSequence[i].Description.PadRight(10);
                    if ((i + 1) % 12 == 0) result += newLine;
                }
                result += newLine;
                result += new String('-', 56) + " CHORDS " + new String('-', 56) + newLine;
                result += newLine;
                // -------------------------------------------------
            }
            else
            {
                result = "There is no processed audio.";
            }
            Console.WriteLine(result);
            return result;
        }



        #endregion
    }
}