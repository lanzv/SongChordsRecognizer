using SongChordsRecognizer.AudioSource;
using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;

namespace SongChordsRecognizer
{
    class Program
    {
        /// <summary>
        /// Main function that will start whole process, takes four arguments:
        ///     1) audio file path
        ///     2) stft window type {Rectangular, Triangular, Parzen, Welch, Nuttal}
        ///     3) filtration type {Identity, AFAM, WO, FNH}
        ///     4) time level of one sample for chord ( non negative integer )
        /// </summary>
        /// <param name="args">Arguments that are needed for classification of song chords.</param>
        static void Main(string[] args)
        {
            Console.WriteLine("[INFO] The program has started.");

            try
            {
                // Parse arguments
                if (args.Length != 4) { throw new Exception(ErrorMessages.ErrorMessages.Program_WrongNumberOfArguments); }

                String audioPath = args[0];
                IWindow window = ParseSTFTWindow(args[1]);
                ISpectrogramFiltration filtration = ParseFiltration(args[2]);
                int sampleLengthLevel = ParseSampleLengthLevel(args[3]);

                Console.WriteLine("[INFO] Arguments was parsed successfully.");

                // Process audio and print chords
                PrintChordsOfSong(audioPath, window, filtration, sampleLengthLevel);

                Console.WriteLine("[INFO] The program ended successfully.");
            }
            catch (Exception e)
            {
                Console.WriteLine();
                Console.WriteLine(e.Message);
                Console.WriteLine();
            }
            

            Console.WriteLine("Press Enter to exit!");
            Console.Read();
        }


        
        /// <summary>
        /// Parse stft window argument or throw exception.
        /// </summary>
        /// <param name="stftWindow">STFT window type in string.</param>
        /// <returns>IWindow object that corresponds to stftWindow string.</returns>
        static IWindow ParseSTFTWindow(String stftWindow)
        {
            switch (stftWindow)
            {
                case "Rectangular":
                    return new RectangularWindow();
                case "Triangular":
                    return new TriangularWindow();
                case "Parzen":
                    return new ParzenWindow();
                case "Welch":
                    return new WelchWindow();
                case "Nuttall":
                    return new NuttallWindow();
                default:
                    throw new Exception(ErrorMessages.ErrorMessages.Program_NotKnownSTFTWindowType);
            }
        }



        /// <summary>
        /// Parse spectrogram filtration argument or throw exception.
        /// </summary>
        /// <param name="filtration">Spectrogram filtration type in string.</param>
        /// <returns>ISpectrogramFiltration object that corresponds to filtration string.</returns>
        static ISpectrogramFiltration ParseFiltration(String filtration)
        {
            switch (filtration)
            {
                case "Identity":
                    return new Identity();
                case "AFAM":
                    return new AccompanimentFrequencyAreaMask();
                case "WO":
                    return new WeightedOctaves();
                case "FNH":
                    return new FilterNthHarmonics();
                default:
                    throw new Exception(ErrorMessages.ErrorMessages.Program_NotKnownFiltrationType);
            }
        }



        /// <summary>
        /// Parse sample length argument or throw exception.
        /// </summary>
        /// <param name="sampleLengthLevel">Logarithm (base 2) of STFT samples. Non negative integer.</param>
        /// <returns>Non negative integer that corresponds to sampleLength string.</returns>
        static int ParseSampleLengthLevel(String sampleLengthLevel)
        {
            if(Int32.TryParse(sampleLengthLevel, out int result) && result >= 0)
            {
                return result;
            }
            throw new Exception(ErrorMessages.ErrorMessages.Program_InvalidSampleLengthLevel);
        }



        /// <summary>
        /// Whole process of the program in one place.
        /// audio file -> Spectrogram -> Chromagram -> Chord Classification -> Print Chords
        /// </summary>
        /// <param name="audioPath">Path of audio file.</param>
        /// <param name="window">STFT window.</param>
        /// <param name="filtration">Spectrogram filtration type.</param>
        /// <param name="sampleLengthLevel">Level of sample time for one chord.</param>
        static void PrintChordsOfSong(String audioPath, IWindow window, ISpectrogramFiltration filtration, int sampleLengthLevel)
        {
            // Generate chords
            AudioSourceWav wav = new AudioSourceWav(audioPath);
            // Generate Spectrogram
            Spectrogram spectrogram = new Spectrogram(wav, sampleLengthLevel, window);
            // Print Spectrogram
            PrintGraphToTextFile spectrogramPrinter = new PrintGraphToTextFile();
            spectrogramPrinter.SetFileName("Spectrogram_first20.txt");
            spectrogram.Print(spectrogramPrinter, 0, 20);
            // Generate Chromagram
            Chromagram chromagram = new Chromagram(spectrogram, filtration);
            // Print Chromagram
            PrintGraphToTextFile chromagramPrinter = new PrintGraphToTextFile();
            chromagramPrinter.SetFileName("Chromagram_first20.txt");
            chromagram.Print(chromagramPrinter, 0, 20);
            // Chord Classifing
            List<Chord> chords = ChordClassifier.GetChords(chromagram);



            // Print chors
            Console.WriteLine();
            Console.WriteLine(new String('-', 56) + " CHORDS " + new String('-', 56));
            Console.WriteLine();
            for (int i = 0; i < chords.Count; i++)
            {
                Console.Write(chords[i].Description.PadRight(10));
                if ((i + 1) % 12 == 0) Console.WriteLine();
            }
            Console.WriteLine();
            Console.WriteLine(new String('-', 56) + " CHORDS " + new String('-', 56));
            Console.WriteLine();
        }
    }
}