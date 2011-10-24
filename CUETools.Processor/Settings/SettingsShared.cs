using System;
using System.IO;

namespace CUETools.Processor.Settings
{
    static class SettingsShared
    {
        public static string GetMyAppDataDir(string appName)
        {
            string appDataDir = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            string myAppDataDir = Path.Combine(appDataDir, appName);

            if (Directory.Exists(myAppDataDir) == false)
            {
                Directory.CreateDirectory(myAppDataDir);
            }

            return myAppDataDir;
        }

        public static string GetProfileDir(string appName, string appPath)
        {
            bool userProfilesEnabled = (appPath == null || File.Exists(Path.Combine(Path.GetDirectoryName(appPath), "user_profiles_enabled")));
            string appDataDir = userProfilesEnabled ?
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData) :
                Path.GetDirectoryName(appPath);
            string myAppDataDir = Path.Combine(appDataDir, appName);
            if (!Directory.Exists(myAppDataDir))
                Directory.CreateDirectory(myAppDataDir);
            return myAppDataDir;
        }
    }
}
