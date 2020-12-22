using SongChordsRecognizer.FourierTransform;
using SongChordsRecognizer.MusicFeatures;
using System;
using System.Collections.Generic;

namespace SongChordsRecognizer.Graphs
{
    /// <summary>
    /// Data about frequency intensities in time for specific Audio waveform.
    /// Result data are stored in double[][] SpectrogramData. You can get them
    /// by calling GetData() function.
    /// </summary>
    class Spectrogram:IGraph
    {
        /// <summary>
        /// Spectrogram data.
        /// First index corresponds to specific time frames ( that corresponds to 'TimeForSample' time of audio ).
        /// Second index specifies one stft sample ( that corresponds to 'i/FrequencyToBinConst' frequency ).
        /// </summary>
        private readonly double[][] SpectrogramData;

        /// <summary>
        /// Time duration of one spectrogram sample in seconds.
        /// </summary>
        public readonly double SampleLength;

        /// <summary>
        /// Number of spectrogram samples.
        /// </summary>
        public readonly int NumberOfSamples;

        /// <summary>
        /// Constant for converting frequency to STFT indices and backward.
        /// i = FrequencyToBinConst * frequency
        /// frequency = i / FrequencyToBinConst
        /// </summary>
        public readonly double FrequencyToBinConst;

        /// <summary>
        /// The AudioSource that its waveform generates Spectrogram.
        /// </summary>
        private AudioSource.AudioSource source;

        /// <summary>
        /// Audio waveform of AudioSource. 
        /// </summary>
        private readonly double[] Waveform;



        /// <summary>
        /// Spectrogram constructor,
        /// generates spectrogram datas.
        /// </summary>
        /// <param name="source">AudioSource of some music sample.</param>
        /// <param name="log2_waveform_length_for_sample">Number of waveform samples for one Spectrogram sample is equal to 2^{'log2_waveform_length_for_sample'}.</param>
        /// <param name="STFTwindow">Convolution window we want to apply on waveform function in STFT algorithm.</param>
        public Spectrogram(AudioSource.AudioSource source, int log2_waveform_length_for_sample, IWindow STFTwindow)
        {
            int numberOfSourceSamples = (int)Math.Pow(2, log2_waveform_length_for_sample);
            this.source = source;
            this.Waveform = source.GetMonoWaveform();
            this.NumberOfSamples = Waveform.Length / numberOfSourceSamples;
            this.SampleLength = numberOfSourceSamples * source.SampleLength;
            this.FrequencyToBinConst = numberOfSourceSamples * source.SampleLength;
            // Generates Spectrogram data
            SpectrogramData = new double[NumberOfSamples][];
            for (int i = 0; i < NumberOfSamples; i++)
            {
                SpectrogramData[i] = FourierTransform.FourierTransform.STFT(Waveform, i * numberOfSourceSamples, log2_waveform_length_for_sample, STFTwindow).Real();
            }
            // log
            Console.WriteLine("[INFO] The spectrogram was successfuly generated.");
        }



        /// <summary>
        /// Compute intensity for specific frequency in specific time.
        /// </summary>
        /// <param name="time">Time in seconds of music sample.</param>
        /// <param name="frequency">Frequency of any pitch.</param>
        /// <returns>Intensity of frequency and time passed as arguments.</returns>
        public double GetIntensity(double time, double frequency)
        {
            int spectogramStep = (int)(time / SampleLength);
            if (spectogramStep >= SpectrogramData.Length || spectogramStep < 0) return 0;

            int frequencyBin = (int)(frequency * FrequencyToBinConst);
            if (frequencyBin >= SpectrogramData[spectogramStep].Length || frequencyBin < 0) return 0;

            return SpectrogramData[spectogramStep][frequencyBin];
        }



        /// <summary>
        /// Filter this spectrogram from misleading results.
        /// </summary>
        /// <param name="filtration">Filtration contains specific algorithm of filtration we want to do.</param>
        /// <returns>Copy of filtered spectrogram data for given filtration algorithm.</returns>
        public double[][] GetFilteredSpectogramData(ISpectrogramFiltration filtration)
        {
            return filtration.Filter(this);
        }



        /// <summary>
        /// Print spectrogram result of 'length' samples starting on 'startingSample'.
        /// This function prints only intensities for existing tone frequencies.
        /// </summary>
        /// <param name="printer">IGraphPrinter printer that prints Spectrogram in some specific way (to file, console, image ... ).</param>
        /// <param name="startingSample">Starting index of Spectrogram samples we want to print.</param>
        /// <param name="length">Number of Spectrogram samples we want to print.</param>
        public void Print(IGraphPrinter printer, int startingSample, int length)
        {
            List<Tone> tones = TonesGenerator.GetListOfTones();
            List<(int Index, String Description)> indicesToPrint = new List<(int Index, String Description)>();
            foreach(Tone tone in tones)
            {
                indicesToPrint.Add(((int)(tone.Frequency * FrequencyToBinConst), tone.Description));
            }
            printer.Print(this.GetData(), startingSample, length, indicesToPrint, this.SampleLength);
            Console.WriteLine("[INFO] Spectrogram graph was printed.");
        }



        /// <summary>
        /// GetData creates the copy of Spectrogram Data and returns it as a result.
        /// </summary>
        /// <returns>Spectrogram Data as a new copy.</returns>
        public double[][] GetData()
        {
            double[][] dataCopy = new double[SpectrogramData.Length][];
            for(int i = 0; i < dataCopy.Length; i++)
            {
                dataCopy[i] = new double[SpectrogramData[i].Length];
                Array.Copy(SpectrogramData[i], dataCopy[i], SpectrogramData[i].Length);
            }
            return dataCopy;
        }
    }
}