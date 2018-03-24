using System;
using System.Collections.Generic;
using System.Text;
using System.ComponentModel;
using Newtonsoft.Json;

namespace CUETools.Codecs.CommandLine
{
    [JsonObject(MemberSerialization.OptIn)]
    public class EncoderSettings : AudioEncoderSettings
    {
        public override string Name => name;

        public override string Extension => extension;

        public override Type EncoderType => typeof(AudioEncoder);

        public override bool Lossless => lossless;

        public EncoderSettings()
            : base()
        {
        }

        public EncoderSettings(
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
            SupportedModes = _supported_modes;
            EncoderMode = _default_mode;
            Path = _path;
            Parameters = _parameters;
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
