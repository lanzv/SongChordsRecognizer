using System;

namespace SongChordsRecognizer.FourierTransform
{
    /// <summary>
    /// Interface for all STFT window functions. 
    /// Each of window should contain function 'Apply' that returns
    /// result of multiplication of two functions:
    /// the specific part of audio waveform and the specific window function
    /// </summary>
    public interface IWindow
    {
        /// <summary>
        /// Apply specific window function on whole audio waveform 'g'.
        /// </summary>
        /// <param name="g">Waveform of some music sample.</param>
        /// <returns></returns>
        public double[] Apply(double[] g);
        /// <summary>
        /// Apply specific window function on part of audio waveform 'g'.
        /// </summary>
        /// <param name="g">Waveform of some music sample.</param>
        /// <param name="offset">Starting index of 'g' of this specific part of audio waveform.</param>
        /// <param name="length">Length of 'g' part (and also length of window).</param>
        /// <returns></returns>
        public double[] Apply(double[] g, int offset, int length);
    }






    /// <summary>
    /// Window function, in shape of rectangle.
    /// w[n] = 1
    /// </summary>
    public class RectangularWindow : IWindow
    {
        public double[] Apply(double[] g)
        {
            double[] result = new double[g.Length];
            for(int i = 0; i < result.Length; i++)
            {
                result[i] = g[i];
            }
            return result;
        }
        public double[] Apply(double[] g, int offset, int length)
        {
            if (g.Length - offset < length) throw new Exception(ErrorMessages.ErrorMessages.IWindow_NotCorrespondingLength);
            double[] result = new double[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = g[offset + i];
            }
            return result;
        }
    }



    /// <summary>
    /// Window function, in shape of triangle.
    /// w[n] = 1 - | (n - N/2) / (N/2) |, where N is length of window (or also 'g' part)
    /// </summary>
    public class TriangularWindow : IWindow
    {
        public double[] Apply(double[] g)
        {
            double[] result = new double[g.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = TriangularFunction(i, g.Length) * g[i];
            }
            return result;
        }
        public double[] Apply(double[] g, int offset, int length)
        {
            if (g.Length - offset < length) throw new Exception(ErrorMessages.ErrorMessages.IWindow_NotCorrespondingLength);
            double[] result = new double[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = TriangularFunction(i, length) * g[offset + i];
            }
            return result;
        }
        private double TriangularFunction(int n, int length)
        {
            return 1.0 - Math.Abs(((double)n - ((double)length / 2.0)) / ((double)length / 2.0));
        }
    }



    /// <summary>
    /// window function in shape of narrow hill.
    /// (n shuffled to left by N/2, N is length of window (or also 'g' part))
    /// for |n| between 0 and (N-1)/4: 
    ///     w[n] = 1 - 6 * (|n|/(N/2))^{2} + 6 * (|n|/(N/2))^{3} 
    /// for |n| between (N-1)/4 and (N-1)/2: 
    ///     w[n] = 2 * (1 - (|n|/(N/2))^{3}
    /// </summary>
    public class ParzenWindow : IWindow
    {
        public double[] Apply(double[] g)
        {
            double[] result = new double[g.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = ParzenFunction(i, g.Length) * g[i];
            }
            return result;
        }
        public double[] Apply(double[] g, int offset, int length)
        {
            if (g.Length - offset < length) throw new Exception(ErrorMessages.ErrorMessages.IWindow_NotCorrespondingLength);
            double[] result = new double[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = ParzenFunction(i, length) * g[offset + i];
            }
            return result;
        }
        private double ParzenFunction(int n, int length)
        {
            n -= length/2;
            if( 0 <= Math.Abs(n) && Math.Abs(n) <= ((double)length - 1.0)/4.0)
            {
                return 1.0 - 6.0 * Math.Pow( (Math.Abs(n) / ((double)length / 2.0)) , 2) + 6.0 * Math.Pow( (Math.Abs(n) / ((double)length / 2.0)) , 3);
            }
            else
            {
                return 2.0 * Math.Pow( 1.0 - (Math.Abs(n) / ((double)length / 2.0)) , 3);
            }
        }
    }



    /// <summary>
    /// Window function, in shape of wide hill.
    /// w[n] = 1 - ((n - N/2) / (N/2))^{2}, where N is length of window (or also 'g' part)
    /// </summary>
    public class WelchWindow : IWindow
    {
        public double[] Apply(double[] g)
        {
            double[] result = new double[g.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = WelchFunction(i, g.Length) * g[i];
            }
            return result;
        }
        public double[] Apply(double[] g, int offset, int length)
        {
            if (g.Length - offset < length) throw new Exception(ErrorMessages.ErrorMessages.IWindow_NotCorrespondingLength);
            double[] result = new double[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = WelchFunction(i, length) * g[offset + i];
            }
            return result;
        }
        private double WelchFunction(int n, int length)
        {
            return 1 - Math.Pow( ( (double)n - ((double)length) / 2.0 ) / ( (double)length / 2.0 ), 2);
        }
    }



    /// <summary>
    /// Window function, in shape of narrow hill, simular to Parzen window.
    /// w[n] = a0 - a1 * cos(2*pi*n/N) + a2 * cos(4*pi*n/N) - a3 * cos(6*pi*n/N),
    /// where N is length of window (or also 'g' part)
    /// </summary>
    public class NuttallWindow : IWindow
    {
        private static readonly double a0 = 0.355768;
        private static readonly double a1 = 0.487396;
        private static readonly double a2 = 0.144232;
        private static readonly double a3 = 0.012604;

        public double[] Apply(double[] g)
        {
            double[] result = new double[g.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NuttallFunction(i, g.Length) * g[i];
            }
            return result;
        }
        public double[] Apply(double[] g, int offset, int length)
        {
            if (g.Length - offset < length) throw new Exception(ErrorMessages.ErrorMessages.IWindow_NotCorrespondingLength);
            double[] result = new double[length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NuttallFunction(i, length) * g[offset + i];
            }
            return result;
        }
        private double NuttallFunction(int n, int length)
        {
            return a0 - a1 * Math.Cos((2.0 * Math.PI * (double)n) / (double)length) + a2 * Math.Cos((4.0 * Math.PI * (double)n) / (double)length) - a3 * Math.Cos((6.0 * Math.PI * (double)n) / (double)length);
        }
    }
}