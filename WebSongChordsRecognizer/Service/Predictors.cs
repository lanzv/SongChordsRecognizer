using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.Configuration;
using SongChordsRecognizer.ErrorMessages;
using SongChordsRecognizer.Logger;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using WebSongChordsRecognizer.Models;

namespace WebSongChordsRecognizer.Service
{
    public class Predictors
    {
        #region Fields

        /// <summary>
        /// Destination of the ACR SongChordRecognizer python pipeline script.
        /// </summary>
        private static string script_path;

        /// <summary>
        /// Destination of the python.exe file.
        /// </summary>
        private static string python_path;

        /// <summary>
        /// Destination of the model predicting original songs.
        /// </summary>
        private static string original_model_path;

        /// <summary>
        /// Destination of the model predicting transposed songs.
        /// </summary>
        private static string transposed_model_path;

        /// <summary>
        /// Destination of the preprocessor that preprocesses original songs.
        /// </summary>
        private static string original_preprocessor_path;

        /// <summary>
        /// Destination of the preprocessor that preprocesses transposed songs.
        /// </summary>
        private static string transposed_preprocessor_path;

        /// <summary>
        /// Configuration file that contains data from appsettings.json config file.
        /// </summary>
        private static readonly IConfiguration configuration = ApplicationConfiguring.CreateConfiguration();

        /// <summary>
        /// Logger of the Predictors class.
        /// </summary>
        private readonly ILogger _logger = ApplicationLogging.CreateLogger<Predictors>();



        #endregion


        #region Initialization

        /// <summary>
        /// Predictors constructor that loads path to the script of Predictors SongChordsRecognizer python pipeline
        /// and path to the python.exe file from appsettings.json config file.
        /// </summary>
        public Predictors()
        {
            script_path = Path.GetFullPath(configuration["Predictors:ACRScriptPath"]);
            python_path = Path.GetFullPath(configuration["Predictors:PythonPath"]);
            original_model_path = Path.GetFullPath(configuration["Predictors:OriginalModelPath"]);
            transposed_model_path = Path.GetFullPath(configuration["Predictors:TransposedModelPath"]);
            original_preprocessor_path = Path.GetFullPath(configuration["Predictors:OriginalPreprocessorPath"]);
            transposed_preprocessor_path = Path.GetFullPath(configuration["Predictors:TransposedPreprocessorPath"]);
        }



        #endregion


        #region Service public methods

        /// <summary>
        /// Whole process of the Song Chords Recognizer algorithm based on the deep learning (the python part).
        /// audio file -> Spectrogram sequences -> key -> Chord sequence prediction -> Beat voting -> Chord Classification
        /// </summary>
        /// <param name="audio">IFormFile of an audio we want to process.</param>
        /// <returns>PredictorsResponse response, result of the process.</returns>
        public PredictorsResponse GetChords(IFormFile audio)
        {
            PredictorsResponse response = new PredictorsResponse();

            // Generate chords
            using (var ms = new MemoryStream())
            {
                // Get audio data
                audio.CopyTo(ms);
                Byte[] audioBytes = ms.ToArray();
                AudioSourceWav wav = new AudioSourceWav(audioBytes, audio.FileName);
                double[] waveform = wav.GetMonoWaveform();
                double sample_rate = wav.SampleRate;

                // Initialize Process
                ProcessStartInfo python_SongChordRecognizer = new ProcessStartInfo();
                python_SongChordRecognizer.FileName = python_path;


                // Prepare command with arguments
                python_SongChordRecognizer.Arguments = script_path +
                    " --original_model_path=" + original_model_path +
                    " --transposed_model_path=" + transposed_model_path +
                    " --original_preprocessor_path=" + original_preprocessor_path +
                    " --transposed_preprocessor_path=" + transposed_preprocessor_path;


                // Python process configuration
                python_SongChordRecognizer.UseShellExecute = false;
                python_SongChordRecognizer.CreateNoWindow = true;
                python_SongChordRecognizer.RedirectStandardInput = true;
                python_SongChordRecognizer.RedirectStandardOutput = true;
                python_SongChordRecognizer.RedirectStandardError = true;


                // Execute process
                string json_response = "";
                string errors = "";
                using (Process process = Process.Start(python_SongChordRecognizer))
                {
                    StreamWriter streamWriter = process.StandardInput;
                    // Send Json request
                    streamWriter.WriteLine(createJsonRequestBody(waveform, sample_rate));
                    streamWriter.Close();
                    // Get Json response
                    json_response = process.StandardOutput.ReadToEnd();
                    errors = process.StandardError.ReadToEnd();
                }
                if(errors != "")
                {
                    // ToDo Better python executing error handling.
                    _logger.LogError(errors);
                    throw new Exception(ErrorMessages.Predictors_ExecutingError);
                }



                // Parse console output
                (response.ChordSequence, response.BeatTimes, response.Key, response.BPM, response.BarQuarters) = parseJsonResponse(json_response);
            }

            return response;
        }



        #endregion


        #region Private methods

        /// <summary>
        /// Create Json Request for python ACR Pipeline.
        /// </summary>
        /// <param name="waveform">Waveform of audio we want to process.</param>
        /// <param name="sample_rate">Sample rate of audio we want to process.</param>
        /// <returns>Json body.</returns>
        private static string createJsonRequestBody(double[] waveform, double sample_rate)
        {
            // Serialize waveform and sample rate
            string json = JsonConvert.SerializeObject(new {
                Waveform = waveform,
                SampleRate = sample_rate
            });

            return json;
        }



        /// <summary>
        /// Parse Json Response from python ACR Pipeline.
        /// </summary>
        /// <param name="json_response">JSON string that contains Key, BPM and ChordSequence keys.</param>
        /// <returns>List of played chords with times, song's key, the bpm value and number of quarter tones in one bar.</returns>
        private static (List<Chord>, List<double>, string, string, int) parseJsonResponse(string json_response)
        {
            // Initialization
            List<Chord> chordSequence = new List<Chord>();
            List<double> beatTimes = new List<double>();
            string[] chordSequenceStr;
            string key;
            string bpm;
            int barQuarters;

            // Parse Json response
            var acrDefinition = new
            {
                Key = "",
                BPM = "",
                ChordSequence = new string[] {},
                BeatTimes = new double[] {},
                BarQuarters = 0
            };
            var acrObj = JsonConvert.DeserializeAnonymousType(json_response, acrDefinition);

            // Get Json values
            key = acrObj.Key;
            bpm = acrObj.BPM;
            chordSequenceStr = acrObj.ChordSequence;
            beatTimes.AddRange(acrObj.BeatTimes);
            barQuarters = acrObj.BarQuarters;

            // Get chord dictionary and add the None chord to it.
            Dictionary<String, Triad> allChords = ChordsGenerator.GetDictionaryOfTriads();
            allChords.Add("N", new Triad() { Description = "N" });

            // Parse and process chords
            foreach (string chordStr in chordSequenceStr)
            {
                // Add chord to a chordSequence list, if N is found, take the last predicted chord.
                chordSequence.Add(allChords[chordStr.Replace(":min", "m")]);
            }


            return (chordSequence, beatTimes, key, bpm, barQuarters);
        }



        #endregion
    }
}
