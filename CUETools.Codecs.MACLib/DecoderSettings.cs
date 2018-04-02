using Newtonsoft.Json;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.MACLib
{
    [JsonObject(MemberSerialization.OptIn)]
    public class DecoderSettings : IAudioDecoderSettings
    {
        #region IAudioDecoderSettings implementation
        [Browsable(false)]
        public string Extension => "ape";

        [Browsable(false)]
        public string Name => "MACLib";

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

        [DisplayName("Version")]
        [Description("Library version")]
        public string Version => Marshal.PtrToStringAnsi(MACLibDll.GetVersionString());
    }
}