using Microsoft.AspNetCore.Http;
using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.ErrorMessages;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using System.Xml.XPath;
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


        #region Service public methods

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
                AudioSourceWav wav = new AudioSourceWav(audioBytes, audio.FileName);
                string waveform = String.Join(";", wav.GetMonoWaveform());
                double sample_rate = wav.SampleRate;

                // Initialize Process
                ProcessStartInfo python_SongChordRecognizer = new ProcessStartInfo();
                python_SongChordRecognizer.FileName = python;


                // Prepare command with arguments
                python_SongChordRecognizer.Arguments = script;


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
                    // Send audio waveform
                    streamWriter.WriteLine(waveform);
                    // Send sample rate
                    streamWriter.WriteLine(sample_rate);
                    // Get the output, chord sequence
                    json_response = process.StandardOutput.ReadToEnd();
                    errors = process.StandardError.ReadToEnd();
                }
                if(errors != "")
                {
                    // ToDo Better python executing error handling.
                    Console.WriteLine(errors);
                    throw new Exception(ErrorMessages.StatisticalModel_ExecutingError);
                }



                // Parse console output
                (response.ChordSequence, response.Key, response.BPM) = parseConsoleOutput(json_response);
            }

            return response;
        }



        #endregion


        #region Private methods

        /// <summary>
        /// Parse Console Output from python ACR Pipeline in a json format.
        /// </summary>
        /// <param name="json_response">JSON string that contains Key, BPM and ChordSequence keys.</param>
        /// <returns>List of played chords, song's key and the bpm value.</returns>
        private static (List<Chord>, string, string) parseConsoleOutput(String json_response)
        {
            // Initialization
            List<Chord> chordSequence = new List<Chord>();
            String key = "";
            String bpm = "";

            // Parse Json response
            var jsonReader = JsonReaderWriterFactory.CreateJsonReader(
                Encoding.UTF8.GetBytes(json_response),
                new System.Xml.XmlDictionaryReaderQuotas()
                );

            // Get Json values
            var root = XElement.Load(jsonReader);
            key = root.XPathSelectElement("//Key").Value;
            bpm = root.XPathSelectElement("//BPM").Value;
            String chordSequenceStr = root.XPathSelectElement("//ChordSequence").Value;


            // Get chord dictionary and add the None chord to it.
            Dictionary<String, Triad> allChords = ChordsGenerator.GetDictionaryOfTriads();
            allChords.Add("N", new Triad() { Description = "N" });

            // Parse and process chords
            foreach (String notParsedChord in chordSequenceStr.Split(','))
            {
                // Parse chord annotation, trim the mess arround
                String parsedChord = notParsedChord.Replace(" ", "");
                parsedChord = parsedChord.Replace(",", "");
                parsedChord = parsedChord.Replace("\'", "");
                parsedChord = parsedChord.Replace("\"", "");
                parsedChord = parsedChord.Replace("[", "");
                parsedChord = parsedChord.Replace("]", "");
                parsedChord = parsedChord.Replace(":min", "m");

                // Add chord to a chordSequence list, if N is found, take the last predicted chord.
                chordSequence.Add(allChords[parsedChord]);
            }


            return (chordSequence, key, bpm);
        }



        #endregion
    }
}
