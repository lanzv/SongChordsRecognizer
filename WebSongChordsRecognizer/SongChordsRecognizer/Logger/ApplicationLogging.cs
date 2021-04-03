using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using SongChordsRecognizer.Configuration;

namespace SongChordsRecognizer.Logger
{
    /// <summary>
    /// Static class that provides ILogger and ILoggerFactory creations.
    /// </summary>
    public static class ApplicationLogging
    {
        #region Fields

        /// <summary>
        /// Configuration file that contains data from appsettings.json config file. For instance, information about Logger levels ect..
        /// </summary>
        private static readonly IConfiguration configuration = ApplicationConfiguring.CreateConfiguration();



        #endregion


        #region Factory

        public static ILoggerFactory Factory { get; } = LoggerFactory.Create(builder => builder.AddConsole().AddDebug());



        #endregion


        #region Logger

        public static ILogger CreateLogger<T>() =>
          Factory.CreateLogger<T>();



        #endregion
    }
}
