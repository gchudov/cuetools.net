using Newtonsoft.Json;
using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    [JsonObject(MemberSerialization.OptIn)]
    public class AudioDecoderSettingsViewModel : INotifyPropertyChanged
    {
        [JsonProperty]
        public AudioDecoderSettings Settings = null;

        public event PropertyChangedEventHandler PropertyChanged;

        [JsonConstructor]
        private AudioDecoderSettingsViewModel()
        {
        }

        public AudioDecoderSettingsViewModel(AudioDecoderSettings settings)
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

        public bool Lossless
        {
            get => true;
            set {
                throw new InvalidOperationException();
            }
        }

        public string Name
        {
            get => Settings.Name;
            set
            {
                if (Settings is CommandLine.DecoderSettings)
                    (Settings as CommandLine.DecoderSettings).name = value;
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
                    (Settings as CommandLine.DecoderSettings).extension = value;
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
