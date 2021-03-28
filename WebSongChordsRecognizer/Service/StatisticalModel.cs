using Microsoft.AspNetCore.Http;
using SongChordsRecognizer.ErrorMessages;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using WebSongChordsRecognizer.Models;

namespace WebSongChordsRecognizer.Service
{
    public class StatisticalModel
    {
        #region Fields

        /// <summary>
        /// Destination of the ACR SongChordRecognizer python pipeline script.
        /// </summary>
        private static readonly string script = @"C:\Users\vojte\source\repos\SongChordsRecognizer\ACR_Pipeline\SongChordsRecognizer.py";



        /// <summary>
        /// Destionation of the python.exe file.
        /// </summary>
        private static readonly string python = @"C:\Users\vojte\AppData\Local\Microsoft\WindowsApps\python.exe";



        #endregion


        #region Service Public Methods

        /// <summary>
        /// Whole process of the Song Chords Recognizer algorithm based on the deep learning (the python part).
        /// audio file -> Spectrogram sequences -> key -> Chord sequence prediction -> Beat voting -> Chord Classification
        /// </summary>
        /// <param name="audio">IFormFile of an audio we want to process.</param>
        /// <returns>StatisticalModelResponse response, result of the process.</returns>
        public StatisticalModelResponse GetChords(IFormFile audio)
        {
            StatisticalModelResponse response = new StatisticalModelResponse();

            // Generate chords
            using (var ms = new MemoryStream())
            {
                // Get audio data
                audio.CopyTo(ms);
                Byte[] audioBytes = ms.ToArray();


                // Initialize Process
                ProcessStartInfo python_SongChordRecognizer = new ProcessStartInfo();
                python_SongChordRecognizer.FileName = python;


                // Prepare command with arguments
                python_SongChordRecognizer.Arguments = script;


                // Python process configuration
                python_SongChordRecognizer.UseShellExecute = false;
                python_SongChordRecognizer.CreateNoWindow = true;
                python_SongChordRecognizer.RedirectStandardOutput = true;
                python_SongChordRecognizer.RedirectStandardError = true;


                // Execute process
                string results = "";
                string errors = "";
                using (Process process = Process.Start(python_SongChordRecognizer))
                {
                    StreamWriter streamWriter = process.StandardInput;
                    streamWriter.Write(audioBytes);
                    results = process.StandardOutput.ReadToEnd();
                    errors = process.StandardError.ReadToEnd();
                }
                if(errors != "")
                {
                    // ToDo Better python executing error handling.
                    throw new Exception(ErrorMessages.StatisticalModel_ExecutingError);
                }



                // Parse console output
                (response.ChordSequence, response.Key, response.BPM) = parseConsoleOutput(results);
            }

            return response;
        }



        #endregion


        #region Private Methods

        /// <summary>
        /// 
        /// </summary>
        /// <param name="results"></param>
        /// <returns></returns>
        private static (List<Chord>, string, int) parseConsoleOutput(String results)
        {
            List<Chord> chordSequence = new List<Chord>();
            String key = "";
            int bpm = 0;
            // ToDo
            return (chordSequence, key, bpm);
        }



        #endregion
    }
}
