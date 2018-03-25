using Newtonsoft.Json;
using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    [JsonObject(MemberSerialization.OptIn)]
    public class AudioDecoderSettingsViewModel : INotifyPropertyChanged
    {
        [JsonProperty]
        public IAudioDecoderSettings Settings = null;

        public event PropertyChangedEventHandler PropertyChanged;

        [JsonConstructor]
        private AudioDecoderSettingsViewModel()
        {
        }

        public AudioDecoderSettingsViewModel(IAudioDecoderSettings settings)
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
                if (Settings is CommandLine.DecoderSettings)
                    return (Settings as CommandLine.DecoderSettings).Path;
                return "";
            }
            set
            {
                if (Settings is CommandLine.DecoderSettings)
                    (Settings as CommandLine.DecoderSettings).Path = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Path"));
            }
        }
        public string Parameters
        {
            get
            {
                if (Settings is CommandLine.DecoderSettings)
                    return (Settings as CommandLine.DecoderSettings).Parameters;
                return "";
            }
            set
            {
                if (Settings is CommandLine.DecoderSettings)
                    (Settings as CommandLine.DecoderSettings).Parameters = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Parameters"));
            }
        }

        public string Name
        {
            get => Settings.Name;
            set
            {
                if (Settings is CommandLine.DecoderSettings)
                    (Settings as CommandLine.DecoderSettings).Name = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Name"));
            }
        }

        public string Extension
        {
            get => Settings.Extension;
            set
            {
                if (Settings is CommandLine.DecoderSettings)
                    (Settings as CommandLine.DecoderSettings).Extension = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Extension"));
            }
        }

        public string DotExtension => "." + Extension;

        public bool CanBeDeleted => Settings is CommandLine.DecoderSettings;

        public bool IsValid =>
               (Settings != null)
            && (Settings is CommandLine.DecoderSettings ? (Settings as CommandLine.DecoderSettings).Path != "" : true);
    }
}
