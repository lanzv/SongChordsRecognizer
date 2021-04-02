using Microsoft.Extensions.Logging;

namespace SongChordsRecognizer.Logger
{
    /// <summary>
    /// Static class that provides ILogger and ILoggerFactory generations.
    /// </summary>
    public static class ApplicationLogging
    {
        #region Factory methods

        public static ILoggerFactory LoggerFactory { get; } = new LoggerFactory();



        #endregion


        #region Logger methods

        public static ILogger CreateLogger<T>() =>
          LoggerFactory.CreateLogger<T>();



        #endregion
    }
}
