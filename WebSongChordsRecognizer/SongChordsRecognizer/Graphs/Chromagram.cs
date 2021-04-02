using Microsoft.Extensions.Logging;
using SongChordsRecognizer.Logger;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;

namespace SongChordsRecognizer.Graphs
{

    /// <summary>
    /// Data about chroma (all 12 tones) intensities in time for specific Audio waveform.
    /// Result data are stored in double[][] ChromagramData. You can get them
    /// by calling GetData() function.
    /// </summary>
    public class Chromagram :IGraph
    {
        #region Properties

        /// <summary>
        /// Chromagram data.
        /// First index corresponds to specific time frames ( one time frame correspond to 'TimeForSample' time of audio )
        /// Second index corresponds to chroma (all twelve tones).
        /// </summary>
        private double[][] ChromagramData { get; }

        /// <summary>
        /// Time duration of one chromagram sample in seconds.
        /// </summary>
        public readonly double SampleLength;



        #endregion


        #region Fields

        /// <summary>
        /// Logger of the Chromagram class.
        /// </summary>
        private readonly ILogger _logger = ApplicationLogging.CreateLogger<Chromagram>();



        #endregion


        #region Initialization

        /// <summary>
        /// Chromagram constructor,
        /// generates chromagram datas from passed spectrogram.
        /// For now, the FilterNthHarmonics filtration on spectrogram is used.
        /// </summary>
        /// <param name="spectrogram">Generated spectrogram of some music audio.</param>
        /// <param name="filtration">Filtration contains specific algorithm of filtration we want to do on spectrogram data.</param>
        public Chromagram(Spectrogram spectrogram, ISpectrogramFiltration filtration)
        {
            //Generate Chromagram data

            // Get filtered spectrogramData
            double[][] spectrogramData = spectrogram.GetFilteredSpectogramData(filtration);

            ChromagramData = new double[spectrogramData.Length][];

            // Get tones(with frequency borders)
            List<Tone> tones = TonesGenerator.GetListOfTones();

            // Key .. Chroma index,
            // Value .. array of two elements: 
            //       1) sum of chroma tones intensity peaks over all octaves
            //       2) number of octaves
            Dictionary<int, double[]> intensity_sums = new Dictionary<int, double[]>();

            // Init intensity_sums
            for (int i= 0; i < TonesGenerator.Chroma.Count; i++)
            {
                intensity_sums.Add(i, new double[] { 0, 0 });
            }


            // Iterate over all Spectrogram samples
            for (int i = 0; i < spectrogramData.Length; i++)
            {
                for (int j = 0; j < TonesGenerator.Chroma.Count; j++)
                {
                    intensity_sums[j] = new double[] { 0, 0 };
                }
                double[] avgs = new double[intensity_sums.Count];

                // Iterate over all tones and sum peak values of single tone intensities together
                foreach (Tone tone in tones)
                {
                    double max_peak = 0;
                    for (int j = (int)(tone.L_FreqBorder * spectrogram.FrequencyToBinConst); j < (int)(tone.R_FreqBorder * spectrogram.FrequencyToBinConst); j++)
                    {
                        if (max_peak < Math.Abs(spectrogramData[i][j]))
                        {
                            max_peak = Math.Abs(spectrogramData[i][j]);
                        }
                    }
                    if(max_peak != 0) // Not interesting peaks, probabily manually changed to zero
                    {
                        intensity_sums[tone.Chroma.HalfTones][0] += max_peak;
                        intensity_sums[tone.Chroma.HalfTones][1] += 1;
                    }
                }

                // Average all sums of chroma
                for (int j = 0; j < TonesGenerator.Chroma.Count; j++)
                {
                    avgs[j] = intensity_sums[j][0] / intensity_sums[j][1];
                }

                ChromagramData[i] = avgs;
            }


            SampleLength = spectrogram.SampleLength;

            // log
            _logger.LogInformation("The chromagram was successfuly generated.");
        }



        #endregion


        #region Public core methods

        /// <summary>
        /// Compute intensity for specific chroma in specific time.
        /// </summary>
        /// <param name="time">Time in seconds that corresponds to some chromagram sample.</param>
        /// <param name="chroma">Chroma we are interested in.</param>
        /// <returns>Intensity of chroma feature and time passed as arguments.</returns>
        public double GetIntensity(double time, Chroma chroma)
        {
            int chromagramSample = (int)(time / SampleLength);
            if (chromagramSample >= ChromagramData.Length || chromagramSample < 0) return 0;

            if (chroma.HalfTones >= ChromagramData[chromagramSample].Length || chroma.HalfTones < 0) return 0;

            return ChromagramData[chromagramSample][chroma.HalfTones];
        }



        /// <summary>
        /// Print and visualize data of chromagram with specific printer type.
        /// </summary>
        /// <param name="printer">IGraphPrinter printer that says how we want to visualize data.</param>
        /// <param name="startingSample">Starting index of chromagram samples we want to print.</param>
        /// <param name="length">Number of chromagram samples we want to print.</param>
        public void Print(IGraphPrinter printer, int startingSample, int length)
        {
            List<Chroma> chroma = TonesGenerator.Chroma;

            List<(int Index, String Description)> indicesToPrint = new List<(int Index, String Description)>();
            for(int i = 0; i < chroma.Count; i++)
            {
                indicesToPrint.Add((i, chroma[i].Description));
            }
            printer.Print(this.GetData(), startingSample, length, indicesToPrint, this.SampleLength);

            _logger.LogInformation("Chromagram graph was printed.");
        }



        /// <summary>
        /// GetData creates the copy of Chromagram Data and returns it as a result.
        /// </summary>
        /// <returns>Chromagram Data as a new copy.</returns>
        public double[][] GetData()
        {
            double[][] dataCopy = new double[ChromagramData.Length][];
            for (int i = 0; i < dataCopy.Length; i++)
            {
                dataCopy[i] = new double[ChromagramData[i].Length];
                Array.Copy(ChromagramData[i], dataCopy[i], ChromagramData[i].Length);
            }
            return dataCopy;
        }



        #endregion
    }
}