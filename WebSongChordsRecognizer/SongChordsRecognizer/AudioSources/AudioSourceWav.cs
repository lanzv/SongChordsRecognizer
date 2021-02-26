using System;

namespace SongChordsRecognizer.AudioSource
{
    /// <summary>
    /// AudioSource of any music file in format '.WAV', contains parsed data of that music file:
    /// For instance Waveform, Number of channels, Sample rate, Byte rate and Sample Length.
    /// AudioSourceWav extends AudioSource abstract class.
    /// </summary>
    public class AudioSourceWav : AudioSource
    {
        /// <summary>
        /// Size of first subchunk, art of FMT subchunk.
        /// </summary>
        public int Chunk1Size { get; private set; }
        /// <summary>
        /// Number of channels, part of FMT subchunk.
        /// </summary>
        public int NumChannels { get; private set; }
        /// <summary>
        /// Sample rate (in other words, number of samples per second), part of 'FMT' subchunk.
        /// </summary>
        public int SampleRate { get; private set; }
        /// <summary>
        /// Byte rate, (who knows the functionality ... probably nothing important), part of 'FMT' subchunk.
        /// </summary>
        public int ByteRate { get; private set; }
        /// <summary>
        /// Block align, (who knows the functionality ... probably nothing important), part of 'FMT' subchunk. 
        /// </summary>
        public int BlockAlign { get; private set; }
        /// <summary>
        /// Number of bits that correspond to one sample, part of 'FMT' subchunk.
        /// </summary>
        public int BitsPerSample { get; private set; }
        /// <summary>
        /// Size of data in datachunk, part of 'DATA' subchunk.
        /// </summary>
        public int DataSize { get; private set; }



        /// <summary>
        /// AudioSourceWav constructor,
        /// parses data from music file in 'audioPath' using constructor of AudioSource.
        /// </summary>
        /// <param name="audioPath">Absolute path to your music file.</param>
        public AudioSourceWav(String audioPath) : base(audioPath) { }




        /// <summary>
        /// AudioSourceWav constructor,
        /// parses data from music file in 'audioData' using constructor of AudioSource.
        /// </summary>
        /// <param name="audioData">Byte array of audio data.</param>
        /// <param name="audioName">The name of uploaded file.</param>
        public AudioSourceWav(byte[] audioData, String audioName) : base(audioData, audioName) { }


        /// <summary>
        /// ParseAudioData parases AudioData from bytes into variables:
        /// AudioWaveform, SampleLength, NumberOfSamples and others, that WAV subchunks contain.
        /// </summary>
        protected override void ParseAudioData()
        {
            // Get Data from Subchunk1 of wave format
            Chunk1Size = BitConverter.ToInt16(AudioData, 16);
            NumChannels = BitConverter.ToInt16(AudioData, 22);
            SampleRate = BitConverter.ToInt32(AudioData, 24);
            ByteRate = BitConverter.ToInt32(AudioData, 28);
            BlockAlign = BitConverter.ToInt16(AudioData, 32);
            BitsPerSample = BitConverter.ToInt16(AudioData, 34);
            //ToDo support BitsPerSample indivisible by 8 either
            if (BitsPerSample % 8 != 0) throw new NotImplementedException(ErrorMessages.ErrorMessages.AudioSource_NotSupportedBitsPerSample);

            // Get Data from 'data' subchunk of wave format
            DataSize = BitConverter.ToInt32(AudioData, GetDatachunkOffset() + 4);
            NumberOfSamples = (DataSize / (BitsPerSample / 8)) / NumChannels;
            SampleLength = 1.0f / SampleRate;
            AudioWaveform = GetParsedWaveform();
        }



        /// <summary>
        /// GetParsedWaveform parses 'data' subchunk from 'AudioData' bytes to array of channels and values representing audio waveform.
        /// </summary>
        /// <returns>Waveform in array double[,], where first index indicates channel a second index indicates specific value in range between -1 and 1.</returns>
        private double[,] GetParsedWaveform()
        {
            double[,] waveform = new double[NumChannels, NumberOfSamples];
            int offset = GetDatachunkOffset() + 8;
            for (int i = 0; i < NumberOfSamples; i++)
            {
                for (int j = 0; j < NumChannels; j++)
                {
                    waveform[j, i] = GetSampleValue(offset + (i + j) * (BitsPerSample / 8)); ;
                }
            }
            return waveform;
        }



        /// <summary>
        /// GetSampleValue parses 'AudioData' bytes from 'data' subchunk sample
        /// at position 'position' to float value in range between -1 and 1.
        /// </summary>
        /// <param name="position">Position of 'AudioData' bytes where starts sample that contains value we want.</param>
        /// <returns>Value of audio waveform at specific position normalized to range between -1 and 1.</returns>
        private float GetSampleValue(int position)
        {
            // Get value from bytes
            double sample = 0;
            for (int i = (BitsPerSample / 8) - 1; i >= 0; i--)
            {
                sample = sample * 256 + AudioData[position + i];
            }
            // Correcting unsigned number with negative sign
            if (sample >= Math.Pow(2, (BitsPerSample - 1)))
            {
                sample = sample - (double)Math.Pow(2, BitsPerSample);
            }
            // Returns the result in range between -1 and 1
            return (float)sample / (float)Math.Pow(2, (BitsPerSample - 1));
        }



        /// <summary>
        /// GetDatachunkOffset iterates over all subchunks until the string 'data' in subchunkID block is found.
        /// </summary>
        /// <returns>Index of 'AudioData' bytes where the 'data' subchunk begins.</returns>
        private int GetDatachunkOffset()
        {
            // offset of second chunk, the one after 'fmt' chunk 
            int i = 20 + BitConverter.ToInt16(AudioData, 16);
            int newChunkSize;
            String str = System.Text.Encoding.UTF8.GetString(AudioData, i, 4);

            // Iterate over all Chunks until we find the 'data' chunk
            // ChunkSize is always on 4th byte of that chunk. 
            while (!str.Equals("data"))
            {
                i += 4;
                newChunkSize = BitConverter.ToInt32(AudioData, i);
                i += newChunkSize + 4;
                str = System.Text.Encoding.UTF8.GetString(AudioData, i, 4);
            }
            return i;
        }
    }
}