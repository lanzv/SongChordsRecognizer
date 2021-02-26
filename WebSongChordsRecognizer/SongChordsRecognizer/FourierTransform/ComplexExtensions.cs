using System.Numerics;

namespace SongChordsRecognizer.FourierTransform
{
    /// <summary>
    /// Static class of Complex number extensions.
    /// </summary>
    public static class ComplexExtensions
    {
        /// <summary>
        /// Convert Complex array to the new array of its Real values.
        /// </summary>
        /// <param name="array">Array of Complex numbers.</param>
        /// <returns>New array that contains only Real numbers (double). </returns>
        public static double[] Real(this Complex[] array)
        {
            double[] result = new double[array.Length/2];
            for(int i = 0; i < result.Length; i++)
            {
                result[i] = array[i].Real;
            }
            return result;
        }
    }
}