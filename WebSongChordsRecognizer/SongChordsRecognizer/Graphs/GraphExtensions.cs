using System;
using System.Linq;

namespace SongChordsRecognizer.Graphs
{
    /// <summary>
    /// Static class of Graph (Spectrogram, Chromagram, ect..) extensions.
    /// </summary>
    public static class GraphExtensions
    {
        /// <summary>
        /// Sum specified subarrays in sequence of given array.
        /// </summary>
        /// <param name="samples">Array of arrays. Each array includes graph data for one graph sample.</param>
        /// <param name="offset">First graph sample.</param>
        /// <param name="length">Number of samples for summing.</param>
        /// <returns>New double array which contains sum of normalized values of specified subarrays in sequenace.</returns>
        public static double[] SumSamples(this double[][] samples, int offset, int length)
        {
            if (length <= 0) throw new Exception(ErrorMessages.ErrorMessages.GraphSumSamples_InvalidLength);
            if (offset < 0 || offset >= samples.Length) throw new Exception(ErrorMessages.ErrorMessages.GraphSumSamples_InvalidOffset);

            // Find the smallest array length
            double[] maxValues = new double[length];
            int minLength = Int32.MaxValue;
            for (int j = offset; j < offset+length && j < samples.Length; j++)
            {
                if (minLength > samples[j].Length) minLength = samples[j].Length;
                maxValues[j-offset] = samples[j].Max();
            }

            // Sum all arrays together
            double[] result = new double[minLength];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = 0;
                for(int j = offset; j < offset+length && j < samples.Length; j++)
                {
                    result[i] += (samples[j][i]/maxValues[j-offset]);
                }
            }
            return result;
        }
    }
}