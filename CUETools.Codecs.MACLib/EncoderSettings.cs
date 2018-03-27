using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using CUETools.Codecs;
using Newtonsoft.Json;

namespace CUETools.Codecs.MACLib
{
    [JsonObject(MemberSerialization.OptIn)]
    public class EncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [Browsable(false)]
        public string Extension => "ape";

        [Browsable(false)]
        public string Name => "MACLib";

        [Browsable(false)]
        public Type EncoderType => typeof(AudioEncoder);

        [Browsable(false)]
        public bool Lossless => true;

        [Browsable(false)]
        public int Priority => 1;

        [Browsable(false)]
        public string SupportedModes => "fast normal high extra insane";

        [Browsable(false)]
        public string DefaultMode => "high";

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

        [DisplayName("Version")]
        [Description("Library version")]
        public string Version => Marshal.PtrToStringAnsi(MACLibDll.GetVersionString());
    };
}
