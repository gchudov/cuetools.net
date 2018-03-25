using System;
using System.Collections.Generic;
using System.Text;
using System.ComponentModel;
using Newtonsoft.Json;

namespace CUETools.Codecs.CommandLine
{
    [JsonObject(MemberSerialization.OptIn)]
    public class EncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [DefaultValue("")]
        [JsonProperty]
        public string Name { get; set; }

        [DefaultValue("")]
        [JsonProperty]
        public string Extension { get; set; }

        [Browsable(false)]
        public Type EncoderType => typeof(AudioEncoder);

        [JsonProperty]
        public bool Lossless { get; set; }

        [Browsable(false)]
        public int Priority => 0;

        [DefaultValue("")]
        [JsonProperty]
        public string SupportedModes { get; set; }

        public string DefaultMode => EncoderMode;

        [Browsable(false)]
        [DefaultValue("")]
        [JsonProperty]
        public string EncoderMode { get; set; }

        [Browsable(false)]
        public AudioPCMConfig PCM { get; set; }

        [Browsable(false)]
        public int BlockSize { get; set; }

        [Browsable(false)]
        [DefaultValue(4096)]
        public int Padding { get; set; }

        public IAudioEncoderSettings Clone()
        {
            return MemberwiseClone() as IAudioEncoderSettings;
        }
        #endregion

        public EncoderSettings()
        {
            this.Init();
        }

        public EncoderSettings(
            string name,
            string extension,
            bool lossless,
            string supportedModes,
            string defaultMode,
            string path,
            string parameters
            )
        {
            this.Init();
            Name = name;
            Extension = extension;
            Lossless = lossless;
            SupportedModes = supportedModes;
            Path = path;
            EncoderMode = defaultMode;
            Parameters = parameters;
        }

        [DefaultValue("")]
        [JsonProperty]
        public string Path
        {
            get;
            set;
        }

        [DefaultValue("")]
        [JsonProperty]
        public string Parameters
        {
            get;
            set;
        }
    }
}
