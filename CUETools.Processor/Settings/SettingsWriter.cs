using System;
using System.IO;
using System.Text;

namespace CUETools.Processor.Settings
{
    public class SettingsWriter
    {
        StreamWriter _sw;

        public SettingsWriter(string appName, string fileName, string appPath)
        {
            string path = Path.Combine(SettingsShared.GetProfileDir(appName, appPath), fileName);
            _sw = new StreamWriter(path, false, Encoding.UTF8);
        }

        public void Save(string name, string value)
        {
            _sw.WriteLine(name + "=" + value);
        }

        public void SaveText(string name, string value)
        {
            _sw.Write(name);
            if (value == "")
            {
                _sw.WriteLine("=");
                return;
            }
            using (StringReader sr = new StringReader(value))
            {
                string lineStr;
                while ((lineStr = sr.ReadLine()) != null)
                    _sw.WriteLine("=" + lineStr);
            }
        }

        public void Save(string name, bool value)
        {
            Save(name, value ? "1" : "0");
        }

        public void Save(string name, int value)
        {
            Save(name, value.ToString());
        }

        public void Save(string name, uint value)
        {
            Save(name, value.ToString());
        }

        public void Save(string name, long value)
        {
            Save(name, value.ToString());
        }

        public void Save(string name, DateTime value)
        {
            Save(name, value.ToBinary());
        }

        public void Close()
        {
            _sw.Close();
        }
    }
}
