using System;
using System.Collections.Generic;
using System.Text;
using System.ComponentModel;
using Newtonsoft.Json;

namespace CUETools.Codecs.CommandLine
{
    [JsonObject(MemberSerialization.OptIn)]
    public class DecoderSettings : IAudioDecoderSettings
    {
        #region IAudioDecoderSettings implementation
        [DefaultValue("")]
        [JsonProperty]
        public string Name { get; set; }

        [DefaultValue("")]
        [JsonProperty]
        public string Extension { get; set; }

        [Browsable(false)]
        public Type DecoderType => typeof(AudioDecoder);

        [Browsable(false)]
        public int Priority => 2;

        public IAudioDecoderSettings Clone()
        {
            return MemberwiseClone() as IAudioDecoderSettings;
        }
        #endregion

        public DecoderSettings()
        {
            this.Init();
        }

        public DecoderSettings(
            string name,
            string extension,
            string path,
            string parameters)
            : base()
        {
            Name = name;
            Extension = extension;
            Path = path;
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
