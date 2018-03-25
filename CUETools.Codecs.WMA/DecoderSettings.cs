using Newtonsoft.Json;
using System;
using System.ComponentModel;

namespace CUETools.Codecs.WMA
{
    [JsonObject(MemberSerialization.OptIn)]
    public class DecoderSettings : IAudioDecoderSettings
    {
        #region IAudioDecoderSettings implementation
        [Browsable(false)]
        public string Extension => "wma";

        [Browsable(false)]
        public string Name => "windows";

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
    }
}
