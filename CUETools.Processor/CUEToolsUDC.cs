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
        public AudioEncoderSettings settings = null;
        public XmlSerializer settingsSerializer = null;
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
            priority = 0;
            path = null;
            parameters = null;
            type = typeof(UserDefinedWriter);
            settingsSerializer = new XmlSerializer(typeof(UserDefinedEncoderSettings));
            settings = new UserDefinedEncoderSettings() { SupportedModes = _supported_modes, DefaultMode = _default_mode, Path = _path, Parameters = _parameters };
        }

        public CUEToolsUDC(AudioEncoderClassAttribute enc, Type enctype)
        {
            name = enc.EncoderName;
            extension = enc.Extension;
            lossless = enc.Lossless;
            priority = enc.Priority;
            path = null;
            parameters = null;
            type = enctype;
            settingsSerializer = new XmlSerializer(enc.Settings);
            settings = Activator.CreateInstance(enc.Settings) as AudioEncoderSettings;
            if (settings == null)
                throw new InvalidOperationException("invalid codec");
        }

        public CUEToolsUDC(
            string _name,
            string _extension,
            string _path,
            string _parameters
            )
        {
            name = _name;
            extension = _extension;
            lossless = true;
            priority = 0;
            path = _path;
            parameters = _parameters;
            type = null;
        }

        public CUEToolsUDC(AudioDecoderClass dec, Type dectype)
        {
            name = dec.DecoderName;
            extension = dec.Extension;
            lossless = true;
            priority = dec.Priority;
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
            get
            {
                var settings = this.settings as UserDefinedEncoderSettings;
                return settings == null ? path : settings.Path;
            }
            set
            {
                var settings = this.settings as UserDefinedEncoderSettings;
                if (settings == null) path = value;
                else settings.Path = value;
                if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Path"));
            }
        }
        public string Parameters
        {
            get
            {
                var settings = this.settings as UserDefinedEncoderSettings;
                return settings == null ? parameters : settings.Parameters;
            }
            set
            {
                var settings = this.settings as UserDefinedEncoderSettings;
                if (settings == null) parameters = value;
                else settings.Parameters = value;
                if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("Parameters"));
            }
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
            get
            {
                string defaultMode;
                return this.settings.GetSupportedModes(out defaultMode);
            }
            set
            {
                var settings = this.settings as UserDefinedEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.SupportedModes = value;
                if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("SupportedModesStr"));
            }
        }

        public string[] SupportedModes
        {
            get
            {
                return this.SupportedModesStr.Split(' ');
            }
        }

        public int EncoderModeIndex
        {
            get
            {
                string[] modes = this.SupportedModes;
                if (modes == null || modes.Length < 2)
                    return -1;
                for (int i = 0; i < modes.Length; i++)
                    if (modes[i] == this.settings.EncoderMode)
                        return i;
                return -1;
            }
        }

        public bool CanBeDeleted
        {
            get
            {
                return type == null || type == typeof(UserDefinedWriter);
            }
        }
    }
}
