using Microsoft.AspNetCore.Http;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using WebSongChordsRecognizer.Models;

namespace WebSongChordsRecognizer.Service
{
    public class StatisticalModel
    {
        #region Service Public Methods

        /// <summary>
        /// Whole process of the Song Chords Recognizer algorithm based on the deep learning (the python part).
        /// audio file -> Spectrogram sequences -> key -> Chord sequence prediction -> Beat voting -> Chord Classification
        /// </summary>
        /// <param name="audio">IFormFile of an audio we want to process.</param>
        /// <returns>StatisticalModelResponse response, result of the process.</returns>
        public StatisticalModelResponse GetChords(IFormFile audio)
        {
            throw new NotImplementedException();
        }



        #endregion
    }
}
