using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.EnvironmentVariables;
using Microsoft.Extensions.Configuration.Json;
using System.Collections.Generic;
using System.IO;

namespace SongChordsRecognizer.Configuration
{
    /// <summary>
    /// Static class that provides IConfiguration of appsettings.json creations.
    /// </summary>
    public static class ApplicationConfiguring
    {
        #region Configuration

        public static IConfiguration CreateConfiguration()
        {
            // Create a JsonConfigurationProvider of appsettings.json
            var appsettingsSource = new JsonConfigurationSource { Path = Path.GetFullPath("./appsettings.json") };
            appsettingsSource.ResolveFileProvider();
            var appsettingsProvider = new JsonConfigurationProvider(appsettingsSource);
            appsettingsProvider.Load();

            // Create a JsonConfigurationProvider of appsettings.Development.json
            var appsettingsDeveloperSource = new JsonConfigurationSource { Path = Path.GetFullPath("./appsettings.Development.json") };
            appsettingsDeveloperSource.ResolveFileProvider();
            var appsettingsDeveloperProvider = new JsonConfigurationProvider(appsettingsDeveloperSource);
            appsettingsDeveloperProvider.Load();

            // Create a Configuration Provider list, Add appsetings.json, appsestings.Development.json and Env variables configuration.
            IList<IConfigurationProvider> cofigurationList = new List<IConfigurationProvider>();
            cofigurationList.Add(appsettingsProvider);
            cofigurationList.Add(appsettingsDeveloperProvider);
            cofigurationList.Add(new EnvironmentVariablesConfigurationProvider());

            return new ConfigurationRoot(cofigurationList);
        }



        #endregion
    }
}