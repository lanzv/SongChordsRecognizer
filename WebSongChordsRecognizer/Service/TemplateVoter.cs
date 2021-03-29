using Microsoft.AspNetCore.Http;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using WebSongChordsRecognizer.Models;

namespace WebSongChordsRecognizer.Service
{
    public class TemplateVoter
    {
        #region Service public methods

        /// <summary>
        /// Whole process of the Song Chords Recognizer algorithm based on the template voting (the .NET part).
        /// audio file -> Spectrogram -> Chromagram -> Chord Classification
        /// </summary>
        /// <param name="audio">IFormFile of an audio we want to process.</param>
        /// <param name="window">STFT window.</param>
        /// <param name="filtration">Spectrogram filtration type.</param>
        /// <param name="sampleLengthLevel">Level of sample time for one chord.</param>
        /// <param name="bpm">Beats per minute value.</param>
        /// <returns>TemplateVoterResponse response, result of the process.</returns>
        public TemplateVoterResponse GetChords(IFormFile audio, IWindow window, ISpectrogramFiltration filtration, int sampleLengthLevel, int bpm)
        {
            TemplateVoterResponse response = new TemplateVoterResponse();

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
                List<Chord> chordSequence = ChordClassifier.GetChords(chromagram, bpm);

                response.ChordSequence = chordSequence;
            }

            return response;
        }



        #endregion
    }
}
