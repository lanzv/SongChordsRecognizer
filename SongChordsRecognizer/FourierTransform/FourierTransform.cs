using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;

namespace SongChordsRecognizer.FourierTransform
{
    static class FourierTransform
    {
        /// <summary>
        /// Generate all STFT values for waveform 'g' asynchronously. 
        /// </summary>
        /// <param name="g">Waveform of some music sample.</param>
        /// <param name="log2_fft_length">Number of STFT samples for this specific part of audio waveform is equal to N = 2^{nth_power_of_two}</param>
        /// <param name="window">Window that is applied on tthis specific part of audio waveform.</param>
        /// <returns>All STFT values of 'g' waveform.</returns>
        public static async Task<double[][]> GetFrequencyIntensitiesAsync(double[] g, int log2_fft_length, IWindow window)
        {
            int fft_length = (int)Math.Pow(2, log2_fft_length);
            int numberOfSamples = g.Length / fft_length;
            double[][] intensities = new double[numberOfSamples][];

            // Run Tasks to compute STFT
            Task<double[]>[] tasks = new Task<double[]>[numberOfSamples];
            for(int i = 0; i < numberOfSamples; i++)
            {
                int index = i;
                tasks[index] = Task<double[]>.Factory.StartNew(
                    () => STFT(g, index * fft_length, log2_fft_length, window).Real()
                    ); 
            }

            // Collect results from tasks
            for (int i = 0; i < numberOfSamples; i++)
            {
                int index = i;
                intensities[index] = await tasks[index];
            }

            return intensities;
        }



        /// <summary>
        /// Short Time Fourier Transform.
        /// Uses FFT to compute frequency intensities from part of function 'g' multiplied by 'window' function.
        /// </summary>
        /// <param name="g">Waveform of some music sample.</param>
        /// <param name="offset">Starting index of 'g' of this specific part of audio waveform.</param>
        /// <param name="log2_fft_length">Number of STFT samples for this specific part of audio waveform is equal to N = 2^{nth_power_of_two}</param>
        /// <param name="window">Window that is applied on tthis specific part of audio waveform.</param>
        /// <returns>STFT part for specified parameters.</returns>
        public static Complex[] STFT(double[] g, int offset, int log2_fft_length, IWindow window)
        {
            double[] covolutioned_g = window.Apply(g, offset, (int)Math.Pow(2, log2_fft_length));
            return FastFourierTransform(covolutioned_g, 0, log2_fft_length);
        }



        /// <summary>
        /// Fast Fourier Transform for waveform 'g'.
        /// </summary>
        /// <param name="g">Waveform of some music sample.</param>
        /// <param name="offset">Starting index of 'g'</param>
        /// <param name="log2_fft_length">Number of FFT samples is equal to N = 2^{nth_power_of_two}</param>
        /// <returns>
        /// Array of complex numbers representing intensities of frequencies in waveform 'g'.
        /// The number of specific frequency in HZ for index 'i' is computed as
        /// i / (intensities.Length * timeForOneSample)
        /// </returns>
        public static Complex[] FastFourierTransform( double[] g, int offset, int log2_fft_length)
        {
            int N = (int)(Math.Pow(2, log2_fft_length));
            return FFTRecursion(N, g, offset, N);
        }



        /// <summary>
        /// Discrete Fourier Transform for waveform 'g'.
        /// </summary>
        /// <param name="g">Waveform of some music sample.</param>
        /// <returns>
        /// Array of complex numbers representing intensities of frequencies in waveform 'g'.
        /// The number of specific frequency in HZ for index 'i' is
        /// i / (intensities.Length * timeForOneSample)
        /// </returns>
        public static Complex[] DiscreteFourierTransform(double[] g)
        {
            Complex[] intensities = new Complex[g.Length];
            int N = g.Length;
            for(int k = 0; k < g.Length; k++)
            {
                Complex sum = new Complex();
                for (int t = 0; t < g.Length; t++)
                {
                    sum += new Complex(g[t], 0) * Complex.Exp(new Complex(0, (-2.0f * Math.PI * (double)k * (double)t)/N));
                }
                intensities[k] = sum;
            }
            return intensities;
        }



        /// <summary>
        /// Recursive function for FastFourierTransform to get the result in N*log(N)
        /// </summary>
        /// <param name="N">Number of samples in the current recursion stage.</param>
        /// <param name="g">Waveform of some music sample.</param>
        /// <param name="offset">Starting index of function 'g' in this specific part of recursion.</param>
        /// <param name="fft_length">Number of samples in the first stage of recursion.</param>
        /// <returns>Array of complex numbers representing intensities of frequencies in waveform 'g'.</returns>
        private static Complex[] FFTRecursion(int N, double[] g, int offset, int fft_length)
        {
            Complex[] result = new Complex[N];
            if (N == 1)
            {
                result[0] = g[offset];
            }
            else
            {
                Complex w = Complex.Exp(new Complex(0, (-2.0f * Math.PI) / N));
                Complex[] even = FFTRecursion(N / 2, g, offset, fft_length);
                Complex[] odd = FFTRecursion(N / 2, g, offset + fft_length / N, fft_length);
                for(int i = 0; i < even.Length; i++)
                {
                    result[i] = even[i] + Complex.Pow(w, i) * odd[i];
                    result[i + N / 2] = even[i] - Complex.Pow(w, i) * odd[i];
                }
            }
            return result;
        }
    }
}