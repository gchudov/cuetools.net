using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace CUETools.Processor.Settings
{
    public class SettingsReader
    {
        Dictionary<string, string> _settings;
        string profilePath;

        public string ProfilePath
        {
            get
            {
                return profilePath;
            }
        }

        public SettingsReader(string appName, string fileName, string appPath)
        {
            _settings = new Dictionary<string, string>();
            profilePath = SettingsShared.GetProfileDir(appName, appPath);
            string path = Path.Combine(profilePath, fileName);
            if (!File.Exists(path))
                return;

            using (StreamReader sr = new StreamReader(path, Encoding.UTF8))
            {
                string line, name = null, val;
                int pos;

                while ((line = sr.ReadLine()) != null)
                {
                    pos = line.IndexOf('=');
                    if (pos != -1)
                    {
                        if (pos > 0)
                        {
                            name = line.Substring(0, pos);
                            val = line.Substring(pos + 1);
                            if (!_settings.ContainsKey(name))
                                _settings.Add(name, val);
                        }
                        else
                        {
                            val = line.Substring(pos + 1);
                            if (_settings.ContainsKey(name))
                                _settings[name] += "\r\n" + val;
                        }
                    }
                }
            }
        }

        public string Load(string name)
        {
            return _settings.ContainsKey(name) ? _settings[name] : null;
        }

        public bool? LoadBoolean(string name)
        {
            string val = Load(name);
            if (val == "0") return false;
            if (val == "1") return true;
            return null;
        }

        public int? LoadInt32(string name, int? min, int? max)
        {
            int val;
            if (!Int32.TryParse(Load(name), out val)) return null;
            if (min.HasValue && (val < min.Value)) return null;
            if (max.HasValue && (val > max.Value)) return null;
            return val;
        }

        public uint? LoadUInt32(string name, uint? min, uint? max)
        {
            uint val;
            if (!UInt32.TryParse(Load(name), out val)) return null;
            if (min.HasValue && (val < min.Value)) return null;
            if (max.HasValue && (val > max.Value)) return null;
            return val;
        }

        public long? LoadLong(string name, long? min, long? max)
        {
            long val;
            if (!long.TryParse(Load(name), out val)) return null;
            if (min.HasValue && (val < min.Value)) return null;
            if (max.HasValue && (val > max.Value)) return null;
            return val;
        }

        public DateTime? LoadDate(string name)
        {
            long? val = LoadLong(name, null, null);
            if (!val.HasValue) return null;
            return DateTime.FromBinary(val.Value);
        }
    }
}
