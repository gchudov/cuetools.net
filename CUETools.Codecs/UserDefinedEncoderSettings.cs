using System;
using System.Collections.Generic;
using System.Text;
using System.ComponentModel;
using Newtonsoft.Json;

namespace CUETools.Codecs
{
    [JsonObject(MemberSerialization.OptIn)]
    public class CommandLineEncoderSettings : AudioEncoderSettings
    {
        public override string Name => name;

        public override string Extension => extension;

        public override Type EncoderType => typeof(CommandLineEncoder);

        public override bool Lossless => lossless;

        public CommandLineEncoderSettings()
            : base()
        {
        }

        [JsonProperty]
        public string name;

        [JsonProperty]
        public string extension;

        [JsonProperty]
        public bool lossless;

        [DefaultValue(null)]
        [JsonProperty]
        public string Path
        {
            get;
            set;
        }

        [DefaultValue(null)]
        [JsonProperty]
        public string Parameters
        {
            get;
            set;
        }

        [JsonProperty]
        public string SupportedModes
        {
            get
            {
                return m_supported_modes;
            }
            set
            {
                m_supported_modes = value;
            }
        }
    }
}
