using Newtonsoft.Json;
using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
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
                if (decoderSettings is CommandLine.DecoderSettings)
                    return (decoderSettings as CommandLine.DecoderSettings).Path;
                return "";
            }
            set
            {
                if (decoderSettings is CommandLine.DecoderSettings)
                    (decoderSettings as CommandLine.DecoderSettings).Path = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Path"));
            }
        }
        public string Parameters
        {
            get
            {
                if (decoderSettings is CommandLine.DecoderSettings)
                    return (decoderSettings as CommandLine.DecoderSettings).Parameters;
                return "";
            }
            set
            {
                if (decoderSettings is CommandLine.DecoderSettings)
                    (decoderSettings as CommandLine.DecoderSettings).Parameters = value;
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
                if (decoderSettings is CommandLine.DecoderSettings)
                    (decoderSettings as CommandLine.DecoderSettings).name = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Name"));
            }
        }

        public string Extension
        {
            get => decoderSettings.Extension;
            set
            {
                if (decoderSettings is CommandLine.DecoderSettings)
                    (decoderSettings as CommandLine.DecoderSettings).extension = value;
                else throw new InvalidOperationException();
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs("Extension"));
            }
        }

        public string DotExtension => "." + Extension;

        public bool CanBeDeleted => decoderSettings is CommandLine.DecoderSettings;

        public bool IsValid =>
               (decoderSettings != null)
            && (decoderSettings is CommandLine.DecoderSettings ? (decoderSettings as CommandLine.DecoderSettings).Path != "" : true);
    }
}
