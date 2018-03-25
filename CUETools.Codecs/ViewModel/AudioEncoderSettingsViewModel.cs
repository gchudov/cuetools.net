using Newtonsoft.Json;
using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    [JsonObject(MemberSerialization.OptIn)]
    public class AudioEncoderSettingsViewModel : INotifyPropertyChanged
    {
        [JsonProperty]
        public IAudioEncoderSettings Settings = null;

        public event PropertyChangedEventHandler PropertyChanged;

        [JsonConstructor]
        private AudioEncoderSettingsViewModel()
        {
        }

        public AudioEncoderSettingsViewModel(IAudioEncoderSettings settings)
        {
            this.Settings = settings;
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
                if (Settings is CommandLine.EncoderSettings)
                    return (Settings as CommandLine.EncoderSettings).Path;
                return "";
            }
            set
            {
                var settings = this.Settings as CommandLine.EncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Path = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Path"));
            }
        }

        public string Parameters
        {
            get
            {
                if (Settings is CommandLine.EncoderSettings)
                    return (Settings as CommandLine.EncoderSettings).Parameters;
                return "";
            }
            set
            {
                var settings = this.Settings as CommandLine.EncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Parameters = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Parameters"));
            }
        }

        public bool Lossless
        {
            get => Settings.Lossless;
            set
            {
                var settings = this.Settings as CommandLine.EncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Lossless = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Lossless"));
            }
        }

        
        public string Name
        {
            get => Settings.Name;
            set
            {
                var settings = this.Settings as CommandLine.EncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Name = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Name"));
            }
        }

        public string Extension
        {
            get => Settings.Extension;
            set
            {
                var settings = this.Settings as CommandLine.EncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.Extension = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Extension"));
            }
        }

        public string DotExtension => "." + Extension;

        public string SupportedModes
        {
            get => Settings.SupportedModes;
            set
            {
                var settings = this.Settings as CommandLine.EncoderSettings;
                if (settings == null) throw new InvalidOperationException();
                settings.SupportedModes = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("SupportedModes"));
            }
        }

        public int EncoderModeIndex
        {
            get
            {
                string[] modes = this.SupportedModes.Split(' ');
                if (modes == null || modes.Length < 2)
                    return -1;
                for (int i = 0; i < modes.Length; i++)
                    if (modes[i] == this.Settings.EncoderMode)
                        return i;
                return -1;
            }
        }

        public bool CanBeDeleted => Settings is CommandLine.EncoderSettings;

        public bool IsValid =>
               (Settings != null)
            && (Settings is CommandLine.EncoderSettings ? (Settings as CommandLine.EncoderSettings).Path != "" : true);
    }
}
