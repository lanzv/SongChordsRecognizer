using System;
using System.IO;

namespace SongChordsRecognizer.AudioSource
{
    /// <summary>
    /// AudioSource contains parsed data of music file that was passed as an argument to constructor.
    /// The waveform of audio is stored in 'AudioWaveform', the length of one waveform sample
    /// in seconds is stored in 'SampleLength'.
    /// </summary>
    public abstract class AudioSource
    {
        /// <summary>
        /// Audio waveform of AudioSource, where first index indicates number of channel, second one indicates number of sample.
        /// </summary>
        public double[,] AudioWaveform { get; protected set; }
        /// <summary>
        /// Time duration of one waveform sample in seconds.
        /// </summary>
        public double SampleLength { get; protected set; }
        /// <summary>
        /// Number of waveform samples for each channel.
        /// </summary>
        public int NumberOfSamples { get; protected set; }
        /// <summary>
        /// Absolute path to your music file.
        /// </summary>
        protected String audioPath { get; }
        /// <summary>
        /// Content of the music file in bytes.
        /// </summary>
        protected byte[] AudioData { get; }



        /// <summary>
        /// AudioSource constructor,
        /// parses data from music file in 'audioPath'.
        /// </summary>
        /// <param name="audioPath">Absolute path to your music file.</param>
        public AudioSource(String audioPath)
        {
            this.audioPath = audioPath;
            AudioData = File.ReadAllBytes(audioPath);
            ParseAudioData();
            // log
            Console.WriteLine("[INFO] The AudioSource of file \"" + audioPath.Split('\\')[audioPath.Split('\\').Length - 1] + "\" was successfuly parsed.");
        }



        /// <summary>
        /// AudioSource constructor,
        /// get data from audioData and name of audioName.
        /// The audio file doesn't exist so there is no audio path.
        /// </summary>
        /// <param name="audioData">Byte array of audio data.</param>
        /// <param name="audioName">The name of uploaded file.</param>
        public AudioSource(byte[] audioData, String audioName)
        {
            this.audioPath = audioName + " was uploaded from browser!";
            AudioData = audioData;
            ParseAudioData();
            // log
            Console.WriteLine("[INFO] The AudioSource of file \"" + audioName + "\" was successfuly parsed.");
        }



        /// <summary>
        /// ParseAudioData parases AudioData from bytes into variables:
        /// AudioWaveform, SampleLength and NumberOfSamples
        /// </summary>
        protected abstract void ParseAudioData();



        /// <summary>
        /// Function averages waveforms over all channels and returns mono audio waveform. 
        /// </summary>
        /// <returns>Averaged mono audio waveform.</returns>
        public double[] GetMonoWaveform()
        {
            double[] averaged = new double[AudioWaveform.GetLength(1)];
            int numChannels = AudioWaveform.GetLength(0);
            for (int i = 0; i < AudioWaveform.GetLength(1); i++)
            {
                double sum = 0;
                for (int j = 0; j < numChannels; j++)
                {
                    sum += AudioWaveform[j, i];
                }
                averaged[i] = sum / (double)numChannels;
            }
            return averaged;
        }
    }
}