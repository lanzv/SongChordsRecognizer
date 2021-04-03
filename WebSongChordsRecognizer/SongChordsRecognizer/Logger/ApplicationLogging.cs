using Microsoft.Extensions.Logging;

namespace SongChordsRecognizer.Logger
{
    /// <summary>
    /// Static class that provides ILogger and ILoggerFactory creations.
    /// </summary>
    public static class ApplicationLogging
    {
        #region Factory

        public static ILoggerFactory Factory { get; } = LoggerFactory.Create(builder => builder.AddConsole().AddDebug());



        #endregion


        #region Logger

        public static ILogger CreateLogger<T>() =>
          Factory.CreateLogger<T>();



        #endregion
    }
}
