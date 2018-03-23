using Newtonsoft.Json;
using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    [JsonObject(MemberSerialization.OptIn)]
    public class AudioEncoderSettingsViewModel : INotifyPropertyChanged
    {
        [JsonProperty]
        public AudioEncoderSettings settings = null;

        public event PropertyChangedEventHandler PropertyChanged;

        [JsonConstructor]
        private AudioEncoderSettingsViewModel()
        {
        }

        public AudioEncoderSettingsViewModel(
            string _name,
            string _extension,
            bool _lossless,
            string _supported_modes,
            string _default_mode,
            string _path,
            string _parameters
            )
        {
            settings = new CommandLineEncoderSettings() { name = _name, extension = _extension, SupportedModes = _supported_modes, EncoderMode = _default_mode, Path = _path, Parameters = _parameters, lossless = _lossless };
        }

        public AudioEncoderSettingsViewModel(AudioEncoderClassAttribute enc)
        {
            settings = Activator.CreateInstance(enc.Settings) as AudioEncoderSettings;
            if (settings == null)
                throw new InvalidOperationException("invalid codec");
        }

        public AudioEncoderSettingsViewModel Clone()
        {
            var res = this.MemberwiseClone() as AudioEncoderSettingsViewModel;
            if (settings != null) res.settings = settings.Clone();
            return res;
        }

        public override string ToString()
        {
            return Name;
        }

        public string FullName => Name + " [" + Extension + "]";

        public string Path
        {
            get
            {
                if (settings is CommandLineEncoderSettings)
                    return (settings as CommandLineEncoderSettings).Path;
                return "";
            }
            set
            {
                var settings = this.settings as CommandLineEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Path = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Path"));
            }
        }

        public string Parameters
        {
            get
            {
                if (settings is CommandLineEncoderSettings)
                    return (settings as CommandLineEncoderSettings).Parameters;
                return "";
            }
            set
            {
                var settings = this.settings as CommandLineEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Parameters = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Parameters"));
            }
        }

        public bool Lossless
        {
            get => settings.Lossless;
            set
            {
                var settings = this.settings as CommandLineEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.lossless = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Lossless"));
            }
        }

        
        public string Name
        {
            get => settings.Name;
            set
            {
                var settings = this.settings as CommandLineEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.name = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Name"));
            }
        }

        public string Extension
        {
            get => settings.Extension;
            set
            {
                var settings = this.settings as CommandLineEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.extension = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Extension"));
            }
        }

        public string DotExtension => "." + Extension;

        public string SupportedModesStr
        {
            get
            {
                string defaultMode;
                return this.settings.GetSupportedModes(out defaultMode);
            }
            set
            {
                var settings = this.settings as CommandLineEncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.SupportedModes = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("SupportedModesStr"));
            }
        }

        public string[] SupportedModes => this.SupportedModesStr.Split(' ');

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

        public bool CanBeDeleted => settings is CommandLineEncoderSettings;

        public bool IsValid =>
               (settings != null)
            && (settings is CommandLineEncoderSettings ? (settings as CommandLineEncoderSettings).Path != "" : true);
    }

    [JsonObject(MemberSerialization.OptIn)]
    public class AudioDecoderSettingsViewModel : INotifyPropertyChanged
    {
        [JsonProperty]
        public AudioDecoderSettings decoderSettings = null;

        public event PropertyChangedEventHandler PropertyChanged;

        [JsonConstructor]
        private AudioDecoderSettingsViewModel()
        {
        }

        public AudioDecoderSettingsViewModel(AudioDecoderSettings settings)
        {
            decoderSettings = settings;
        }

        public AudioDecoderSettingsViewModel Clone()
        {
            var res = this.MemberwiseClone() as AudioDecoderSettingsViewModel;
            if (decoderSettings != null) res.decoderSettings = decoderSettings.Clone();
            return res;
        }

        public override string ToString()
        {
            return Name;
        }

        public string FullName => Name + " [" + Extension + "]";

        public string Path
        {
            get
            {
                if (decoderSettings is CommandLineDecoderSettings)
                    return (decoderSettings as CommandLineDecoderSettings).Path;
                return "";
            }
            set
            {
                if (decoderSettings is CommandLineDecoderSettings)
                    (decoderSettings as CommandLineDecoderSettings).Path = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Path"));
            }
        }
        public string Parameters
        {
            get
            {
                if (decoderSettings is CommandLineDecoderSettings)
                    return (decoderSettings as CommandLineDecoderSettings).Parameters;
                return "";
            }
            set
            {
                if (decoderSettings is CommandLineDecoderSettings)
                    (decoderSettings as CommandLineDecoderSettings).Parameters = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Parameters"));
            }
        }

        public bool Lossless
        {
            get => true;
            set {
                throw new InvalidOperationException();
            }
        }

        public string Name
        {
            get => decoderSettings.Name;
            set
            {
                if (decoderSettings is CommandLineDecoderSettings)
                    (decoderSettings as CommandLineDecoderSettings).name = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Name"));
            }
        }

        public string Extension
        {
            get => decoderSettings.Extension;
            set
            {
                if (decoderSettings is CommandLineDecoderSettings)
                    (decoderSettings as CommandLineDecoderSettings).extension = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Extension"));
            }
        }

        public string DotExtension => "." + Extension;

        public bool CanBeDeleted => decoderSettings is CommandLineDecoderSettings;

        public bool IsValid =>
               (decoderSettings != null)
            && (decoderSettings is CommandLineDecoderSettings ? (decoderSettings as CommandLineDecoderSettings).Path != "" : true);
    }
}
