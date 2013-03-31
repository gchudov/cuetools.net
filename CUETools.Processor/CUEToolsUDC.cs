using System;
using System.ComponentModel;
using System.Xml.Serialization;
using CUETools.Codecs;

namespace CUETools.Processor
{
    public class CUEToolsUDC : INotifyPropertyChanged
    {
        public string name = "";
        public string extension = "wav";
        public string path = "";
        public string parameters = "";
        public Type type = null;
        public object settings = null;
        public XmlSerializer settingsSerializer = null;
        public string supported_modes = "";
        public string default_mode = "";
        public bool lossless = false;
        public int priority = 0;

        public event PropertyChangedEventHandler PropertyChanged;

        public CUEToolsUDC(
            string _name,
            string _extension,
            bool _lossless,
            string _supported_modes,
            string _default_mode,
            string _path,
            string _parameters
            )
        {
            name = _name;
            extension = _extension;
            lossless = _lossless;
            supported_modes = _supported_modes;
            default_mode = _default_mode;
            priority = 0;
            path = _path;
            parameters = _parameters;
            type = null;
        }

        public CUEToolsUDC(AudioEncoderClass enc, Type enctype)
        {
            name = enc.EncoderName;
            extension = enc.Extension;
            lossless = enc.Lossless;
            supported_modes = enc.SupportedModes;
            default_mode = enc.DefaultMode;
            priority = enc.Priority;
            path = null;
            parameters = "";
            type = enctype;
            settingsSerializer = null;
            settings = null;
            if (enc.Settings != null && enc.Settings != typeof(object))
            {
                settingsSerializer = new XmlSerializer(enc.Settings);
                settings = Activator.CreateInstance(enc.Settings);
            }
        }

        public CUEToolsUDC(AudioDecoderClass dec, Type dectype)
        {
            name = dec.DecoderName;
            extension = dec.Extension;
            lossless = true;
            supported_modes = "";
            default_mode = "";
            priority = 1;
            path = null;
            parameters = null;
            type = dectype;
        }

        public override string ToString()
        {
            return name;
        }

        public string Name
        {
            get { return name; }
            set { name = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Name")); }
        }
        public string FullName
        {
            get { return name + " [" + extension + "]"; }
            //set { name = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Name")); }
        }
        public string Path
        {
            get { return path; }
            set { path = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Path")); }
        }
        public string Parameters
        {
            get { return parameters; }
            set { parameters = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Parameters")); }
        }
        public bool Lossless
        {
            get { return lossless; }
            set { lossless = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Lossless")); }
        }
        public string Extension
        {
            get { return extension; }
            set { extension = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Extension")); }
        }
        public string DotExtension
        {
            get { return "." + extension; }
        }
        public string SupportedModesStr
        {
            get { return supported_modes; }
            set { supported_modes = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("SupportedModesStr")); }
        }

        public string[] SupportedModes
        {
            get
            {
                return supported_modes.Split(' ');
            }
        }

        public int DefaultModeIndex
        {
            get
            {
                string[] modes = supported_modes.Split(' ');
                if (modes == null || modes.Length < 2)
                    return -1;
                for (int i = 0; i < modes.Length; i++)
                    if (modes[i] == default_mode)
                        return i;
                return -1;
            }
        }

        public bool CanBeDeleted
        {
            get
            {
                return path != null;
            }
        }
    }
}
