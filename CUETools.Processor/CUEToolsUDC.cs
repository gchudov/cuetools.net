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

        private string supported_modes = "";
        private string default_mode = "";

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

        public CUEToolsUDC(AudioEncoderClassAttribute enc, Type enctype)
        {
            name = enc.EncoderName;
            extension = enc.Extension;
            lossless = enc.Lossless;
            priority = enc.Priority;
            path = null;
            parameters = "";
            type = enctype;
            settingsSerializer = null;
            settings = null;
            if (enc.Settings != null && typeof(AudioEncoderSettings).IsAssignableFrom(enc.Settings))
            {
                settingsSerializer = new XmlSerializer(enc.Settings);
                settings = Activator.CreateInstance(enc.Settings) as AudioEncoderSettings;
            }
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
            get
            {
                string defaultMode;
                return this.settings == null ? this.supported_modes : this.settings.GetSupportedModes(out defaultMode);
            }
            set
            {
                if (this.settings != null) throw new NotSupportedException();
                supported_modes = value; if (PropertyChanged != null) PropertyChanged(this, new PropertyChangedEventArgs("SupportedModesStr"));
            }
        }

        public string EncoderMode
        {
            get
            {
                if (this.settings != null) return this.settings.EncoderMode;
                else return this.default_mode;
            }
            set
            {
                if (this.settings != null) this.settings.EncoderMode = value;
                else this.default_mode = value;
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
                    if (modes[i] == this.EncoderMode)
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
