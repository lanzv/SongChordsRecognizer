using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.Graphs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SongChordsRecognizer.Parsers
{
    public static class InputArgsParser
    {

        /// <summary>
        /// Parse stft window argument or throw exception.
        /// </summary>
        /// <param name="stftWindow">STFT window type in string.</param>
        /// <returns>IWindow object that corresponds to stftWindow string.</returns>
        public static IWindow ParseSTFTWindow(String stftWindow)
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
        public static ISpectrogramFiltration ParseFiltration(String filtration)
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
        public static int ParseSampleLengthLevel(String sampleLengthLevel)
        {
            if (Int32.TryParse(sampleLengthLevel, out int result) && result >= 0)
            {
                return result;
            }
            throw new Exception(ErrorMessages.ErrorMessages.Program_InvalidSampleLengthLevel);
        }



        /// <summary>
        /// Parse bpm argument or throw exception.
        /// </summary>
        /// <param name="bpm">BPM of the song. Non negative integer.</param>
        /// <returns>Non negative integer that corresponds to bpm string.</returns>
        public static int ParseBPM(String bpm)
        {
            if (Int32.TryParse(bpm, out int result) && result >= 0)
            {
                return result;
            }
            throw new Exception(ErrorMessages.ErrorMessages.Program_InvalidBPM);
        }
    }
}